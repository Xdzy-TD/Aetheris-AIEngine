"""
Artifact Store — persistent, hash-verified evidence.

Every mask, overlay, change map and report an analysis produces is written
here with its SHA-256 and the model version that produced it, so a result can
be re-examined after the process that made it is gone. Deliberately not
``/tmp``: evidence that evaporates on reboot cannot be audited.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import uuid
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from controller.model_registry import version_for_task

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps Pillow out of import path
    import numpy as np
    from PIL import Image

logger = structlog.get_logger(__name__)

_DEFAULT_ROOT = Path(os.environ.get("AETHERIS_ARTIFACT_DIR", "./artifacts")).expanduser()

# Every artifact_id this store hands out is uuid.uuid4().hex[:16] (put(), below).
# get_record()/verify() build a filesystem path by string-joining a caller-
# supplied artifact_id (GET /artifacts/{artifact_id} passes the URL segment
# straight through) — without this check, an id like "../../etc/passwd" walks
# the path outside self.root and reads any *.meta.json file the process can see.
_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{1,32}$")


class ArtifactStore:
    """Persistent store for analysis artifacts, each recorded with its SHA-256."""

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root else _DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        data: bytes | Path | str,
        artifact_type: str,
        model_version: str,
        query_id: str = "",
        suffix: str = "",
    ) -> dict[str, object]:
        """Store bytes (or copy a file) and return its evidence record.

        Returns a record carrying ``artifact_id``, ``sha256``, ``type`` and
        ``model_version`` — the four fields needed to cite it later.
        """
        if isinstance(data, bytes):
            payload = data
        else:
            src = Path(data)
            payload, suffix = src.read_bytes(), suffix or src.suffix

        sha = hashlib.sha256(payload).hexdigest()
        artifact_id = uuid.uuid4().hex[:16]
        dest = self.root / f"{artifact_id}{suffix}"
        dest.write_bytes(payload)

        record: dict[str, object] = {
            "artifact_id": artifact_id,
            "sha256": sha,
            "type": artifact_type,
            "model_version": model_version,
            "query_id": query_id,
            "path": str(dest),
            "size_bytes": len(payload),
            "created_at": datetime.now(UTC).isoformat(),
        }
        # ``.meta.json``, not ``.json``: a stored *.json* artifact would otherwise
        # overwrite its own metadata sidecar and fail verification.
        (self.root / f"{artifact_id}.meta.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        logger.info("artifact_stored", artifact_id=artifact_id, type=artifact_type)
        return record

    def get_record(self, artifact_id: str) -> dict[str, object] | None:
        """Return a stored artifact's record, or ``None`` if unknown."""
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            return None
        meta = self.root / f"{artifact_id}.meta.json"
        if not meta.exists():
            return None
        return json.loads(meta.read_text(encoding="utf-8"))

    def verify(self, artifact_id: str) -> bool:
        """Re-hash the stored bytes and compare against the recorded digest."""
        record = self.get_record(artifact_id)
        if record is None:
            return False
        path = Path(str(record["path"]))
        if not path.exists():
            return False
        return hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


@lru_cache(maxsize=1)
def default_store() -> ArtifactStore:
    """Process-wide store, rooted at ``$AETHERIS_ARTIFACT_DIR`` (default ``./artifacts``)."""
    return ArtifactStore()


def save_image(image: "Image.Image", artifact_type: str, task: str) -> dict[str, object]:
    """Persist a PIL image as evidence and return its record.

    Specialists call this instead of writing to ``tempfile.gettempdir()``: a
    mask referenced by a report has to still exist when the report is read.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return default_store().put(
        buf.getvalue(), artifact_type, version_for_task(task), suffix=".png"
    )


def save_geotiff(
    mask: "np.ndarray",
    geo_meta: dict[str, Any],
    native_shape: tuple[int, int],
    artifact_type: str,
    task: str,
) -> dict[str, object]:
    """Persist a single-band mask as evidence, as a real GeoTIFF when the
    source raster carried a CRS, else falling back to :func:`save_image`.

    A mask computed on a resized processing grid (``mask.shape``) does not
    share the source raster's transform (``native_shape``) — writing the
    original transform unscaled would silently claim the output covers the
    wrong extent at the wrong resolution. The transform is rescaled to the
    output grid so the GeoTIFF this writes actually georeferences the pixels
    it contains, not the ones it was resampled from.
    """
    if not geo_meta.get("georeferenced") or not geo_meta.get("crs"):
        from PIL import Image

        return save_image(Image.fromarray(mask), artifact_type, task)

    try:
        from rasterio.io import MemoryFile
        from rasterio.transform import Affine
    except ImportError:
        from PIL import Image

        return save_image(Image.fromarray(mask), artifact_type, task)

    h0, w0 = native_shape[:2]
    ht, wt = mask.shape[:2]
    transform = Affine(*geo_meta["transform"]) * Affine.scale(w0 / wt, h0 / ht)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff", height=ht, width=wt, count=1,
            dtype=str(mask.dtype), crs=geo_meta["crs"], transform=transform,
        ) as dst:
            dst.write(mask, 1)
        payload = memfile.read()

    return default_store().put(payload, artifact_type, version_for_task(task), suffix=".tif")
