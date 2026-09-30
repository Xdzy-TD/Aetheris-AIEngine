"""
SNAP gpt CLI pipeline — SAR radiometric calibration, terrain correction,
and speckle filtering via ESA's Sentinel Application Platform.

This module shells out to the ``gpt`` (Graph Processing Tool) command-line
interface that ships with ESA SNAP.  If SNAP is not installed, callers
should fall back to :mod:`preprocessing.fallback_filters`.

Typical SNAP processing chain for SAR:
    Apply-Orbit-File → Calibration → Speckle-Filter → Terrain-Correction
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

import structlog

logger = structlog.get_logger(__name__)


def is_snap_available() -> bool:
    """Check whether ESA SNAP ``gpt`` CLI is on PATH."""
    return shutil.which("gpt") is not None


def run_snap_despeckle(
    input_path: str | Path,
    output_path: str | Path | None = None,
    filter_type: Literal["Lee", "Frost", "Refined Lee", "Lee Sigma"] = "Refined Lee",
    window_size: int = 7,
    calibrate: bool = True,
    terrain_correct: bool = False,
) -> Path:
    """Run SAR despeckle (and optional calibration) via SNAP gpt.

    Args:
        input_path:       Path to input SAR product (SAFE dir or GeoTIFF).
        output_path:      Where to write the result. Defaults to
                          ``<input_stem>_despeckled.tif``.
        filter_type:      SNAP speckle filter name.
        window_size:      Filter window size.
        calibrate:        Whether to apply radiometric calibration first.
        terrain_correct:  Whether to apply Range-Doppler terrain correction.

    Returns:
        Path to the output GeoTIFF.

    Raises:
        RuntimeError: If ``gpt`` is not found or the subprocess fails.
    """
    if not is_snap_available():
        raise RuntimeError(
            "ESA SNAP 'gpt' CLI not found on PATH. Install SNAP or use "
            "preprocessing.fallback_filters as an alternative."
        )

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_despeckled.tif")
    output_path = Path(output_path)

    # Build SNAP XML processing graph
    graph_xml = _build_graph_xml(
        input_path=str(input_path),
        output_path=str(output_path),
        filter_type=filter_type,
        window_size=window_size,
        calibrate=calibrate,
        terrain_correct=terrain_correct,
    )

    # Write graph to temp file and execute
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", delete=False, prefix="aetheris_snap_"
    ) as f:
        f.write(graph_xml)
        graph_path = f.name

    logger.info(
        "snap_pipeline_starting",
        input=str(input_path),
        filter=filter_type,
        calibrate=calibrate,
        terrain_correct=terrain_correct,
    )

    try:
        result = subprocess.run(
            ["gpt", graph_path],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            logger.error("snap_pipeline_failed", stderr=result.stderr[:500])
            raise RuntimeError(f"SNAP gpt failed: {result.stderr[:500]}")

        logger.info("snap_pipeline_complete", output=str(output_path))
        return output_path

    finally:
        Path(graph_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# SNAP XML graph builder
# ---------------------------------------------------------------------------

def _build_graph_xml(
    input_path: str,
    output_path: str,
    filter_type: str,
    window_size: int,
    calibrate: bool,
    terrain_correct: bool,
) -> str:
    """Build an ESA SNAP processing graph XML string.

    The graph chains operators in order:
        Read → [Calibration] → Speckle-Filter → [Terrain-Correction] → Write
    """
    graph = ET.Element("graph", id="aetheris-despeckle")
    ET.SubElement(graph, "version").text = "1.0"

    nodes: list[str] = []

    # Read node
    _add_node(graph, "Read", "Read", source=None, params={"file": input_path})
    nodes.append("Read")

    # Calibration (optional)
    if calibrate:
        _add_node(
            graph, "Calibration", "Calibration",
            source=nodes[-1],
            params={
                "outputSigmaBand": "true",
                "selectedPolarisations": "",
            },
        )
        nodes.append("Calibration")

    # Speckle filter
    _add_node(
        graph, "Speckle-Filter", "Speckle-Filter",
        source=nodes[-1],
        params={
            "filter": filter_type,
            "filterSizeX": str(window_size),
            "filterSizeY": str(window_size),
        },
    )
    nodes.append("Speckle-Filter")

    # Terrain correction (optional)
    if terrain_correct:
        _add_node(
            graph, "Terrain-Correction", "Terrain-Correction",
            source=nodes[-1],
            params={
                "demName": "SRTM 1Sec HGT",
                "pixelSpacingInMeter": "10.0",
            },
        )
        nodes.append("Terrain-Correction")

    # Write node
    _add_node(
        graph, "Write", "Write",
        source=nodes[-1],
        params={
            "file": output_path,
            "formatName": "GeoTIFF",
        },
    )

    return ET.tostring(graph, encoding="unicode", xml_declaration=True)


def _add_node(
    graph: ET.Element,
    node_id: str,
    operator: str,
    source: str | None,
    params: dict[str, str],
) -> None:
    """Add a processing node to the SNAP graph XML."""
    node = ET.SubElement(graph, "node", id=node_id)
    ET.SubElement(node, "operator").text = operator

    if source is not None:
        sources = ET.SubElement(node, "sources")
        ET.SubElement(sources, "sourceProduct", refid=source)

    parameters = ET.SubElement(node, "parameters")
    for key, value in params.items():
        ET.SubElement(parameters, key).text = value
