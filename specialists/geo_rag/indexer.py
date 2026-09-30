"""
Geospatial RAG Index — builds and queries a vector index of geographic
context (OpenStreetMap tags, DEM elevation, land-cover statistics).

The index grounds VLM answers against authoritative geospatial data,
reducing hallucination on real Cartosat/RISAT tiles where the VLM has
never seen the sensor characteristics.

When Qdrant is available, uses it as the vector store. Otherwise falls
back to a simple in-memory NumPy cosine-similarity index.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import structlog
from numpy.typing import NDArray

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = structlog.get_logger(__name__)


def _is_qdrant_available() -> bool:
    try:
        import qdrant_client  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class GeoDocument:
    """A geographic context document for the RAG index."""
    doc_id: str
    text: str
    layer: str  # "osm", "dem", or "landcover"
    bbox: list[float]  # [west, south, east, north]
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: NDArray[np.floating] | None = None


class GeoIndex:
    """Vector index for geospatial context retrieval.

    Supports two backends:
    1. Qdrant (when available): full-featured vector DB.
    2. In-memory NumPy: simple cosine-similarity search.

    Args:
        collection_name: Name of the Qdrant collection.
        embed_dim:       Embedding dimensionality.
        qdrant_url:      Qdrant server URL (default: localhost:6333).
    """

    def __init__(
        self,
        collection_name: str = "aetheris_geo",
        embed_dim: int = 384,
        qdrant_url: str = "http://localhost:6333",
    ) -> None:
        self.collection_name = collection_name
        self.embed_dim = embed_dim
        self.qdrant_url = qdrant_url

        self._qdrant_client: QdrantClient | None = None
        self._memory_store: list[GeoDocument] = []
        self._use_qdrant = False

        self._init_backend()

    def _init_backend(self) -> None:
        """Try to connect to Qdrant; fall back to in-memory."""
        if _is_qdrant_available():
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams

                self._qdrant_client = QdrantClient(url=self.qdrant_url, timeout=5)
                # Ensure collection exists
                collections = [c.name for c in self._qdrant_client.get_collections().collections]
                if self.collection_name not in collections:
                    self._qdrant_client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(
                            size=self.embed_dim,
                            distance=Distance.COSINE,
                        ),
                    )
                self._use_qdrant = True
                logger.info("geo_index_qdrant_connected", collection=self.collection_name)
            except Exception as exc:
                logger.warning("geo_index_qdrant_failed_using_memory", error=str(exc))
                self._use_qdrant = False
        else:
            logger.info("geo_index_using_memory_backend")

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(self, doc: GeoDocument) -> None:
        """Add a document to the index."""
        if doc.embedding is None:
            doc.embedding = self._embed_text(doc.text)

        if self._use_qdrant:
            self._add_qdrant(doc)
        else:
            self._memory_store.append(doc)

    def add_osm_tags(
        self,
        tags: list[dict[str, Any]],
        bbox: list[float],
    ) -> int:
        """Index a batch of OSM tags for an AOI.

        Args:
            tags: List of OSM tag dicts (e.g., {"name": "...", "amenity": "..."}).
            bbox: Bounding box of the AOI.

        Returns:
            Number of documents added.
        """
        count = 0
        for tag in tags:
            text = " ".join(f"{k}={v}" for k, v in tag.items() if v)
            doc = GeoDocument(
                doc_id=hashlib.md5(text.encode()).hexdigest()[:12],
                text=text,
                layer="osm",
                bbox=bbox,
                metadata=tag,
            )
            self.add_document(doc)
            count += 1
        logger.info("geo_index_osm_added", count=count)
        return count

    def add_elevation_stats(
        self,
        stats: dict[str, float],
        bbox: list[float],
    ) -> None:
        """Index DEM elevation statistics."""
        text = (
            f"Elevation statistics: min={stats.get('min', 0):.1f}m, "
            f"max={stats.get('max', 0):.1f}m, mean={stats.get('mean', 0):.1f}m, "
            f"std={stats.get('std', 0):.1f}m"
        )
        doc = GeoDocument(
            doc_id=f"dem_{hashlib.md5(str(bbox).encode()).hexdigest()[:8]}",
            text=text,
            layer="dem",
            bbox=bbox,
            metadata=stats,
        )
        self.add_document(doc)

    def add_landcover(
        self,
        distribution: dict[str, float],
        bbox: list[float],
    ) -> None:
        """Index land-cover class distribution."""
        parts = [f"{cls}: {pct:.1f}%" for cls, pct in distribution.items()]
        text = "Land cover distribution: " + ", ".join(parts)
        doc = GeoDocument(
            doc_id=f"lc_{hashlib.md5(str(bbox).encode()).hexdigest()[:8]}",
            text=text,
            layer="landcover",
            bbox=bbox,
            metadata=distribution,
        )
        self.add_document(doc)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        layer_filter: list[str] | None = None,
        bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Search the index for relevant geospatial context.

        Args:
            query_text:   Natural language query.
            top_k:        Number of results to return.
            layer_filter: Restrict to specific layers (e.g., ["osm"]).
            bbox:         [west, south, east, north] — restrict to documents
                          whose own bbox intersects this AOI. ``None`` (or a
                          zero-area box) disables spatial filtering; text
                          similarity alone otherwise happily returns context
                          indexed for a completely different AOI.

        Returns:
            List of dicts with ``text``, ``layer``, ``score``, ``metadata``.
        """
        query_emb = self._embed_text(query_text)
        if bbox is not None and bbox[0] == bbox[2] and bbox[1] == bbox[3]:
            bbox = None  # degenerate/placeholder box — treat as "no AOI given"

        if self._use_qdrant:
            return self._query_qdrant(query_emb, top_k, layer_filter, bbox)
        else:
            return self._query_memory(query_emb, top_k, layer_filter, bbox)

    @staticmethod
    def _bbox_intersects(a: list[float], b: list[float]) -> bool:
        """True if two [west, south, east, north] boxes overlap (touching counts)."""
        return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]

    # ------------------------------------------------------------------
    # Qdrant backend
    # ------------------------------------------------------------------

    def _add_qdrant(self, doc: GeoDocument) -> None:
        from qdrant_client.models import PointStruct

        assert self._qdrant_client is not None  # only called when _use_qdrant is True
        assert doc.embedding is not None  # add_document() always embeds before this call

        self._qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=doc.doc_id,
                    vector=doc.embedding.tolist(),
                    payload={
                        "text": doc.text,
                        "layer": doc.layer,
                        "bbox": doc.bbox,
                        **doc.metadata,
                    },
                )
            ],
        )

    def _query_qdrant(
        self,
        query_emb: NDArray,
        top_k: int,
        layer_filter: list[str] | None,
        bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        qfilter = None
        if layer_filter:
            qfilter = Filter(
                must=[FieldCondition(key="layer", match=MatchAny(any=layer_filter))]
            )

        assert self._qdrant_client is not None  # only called when _use_qdrant is True
        # No native bbox-intersection index here, so over-fetch and filter
        # client-side using the bbox we already store in each point's payload.
        results = self._qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_emb.tolist(),
            limit=top_k * 4 if bbox else top_k,
            query_filter=qfilter,
        )
        if bbox is not None:
            results = [
                r for r in results if self._bbox_intersects(r.payload.get("bbox", bbox), bbox)
            ][:top_k]

        return [
            {
                "text": r.payload.get("text", ""),
                "layer": r.payload.get("layer", ""),
                "score": round(r.score, 4),
                "metadata": {k: v for k, v in r.payload.items() if k not in ("text", "layer")},
            }
            for r in results
        ]

    # ------------------------------------------------------------------
    # In-memory backend
    # ------------------------------------------------------------------

    def _query_memory(
        self,
        query_emb: NDArray,
        top_k: int,
        layer_filter: list[str] | None,
        bbox: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        candidates = self._memory_store
        if layer_filter:
            candidates = [d for d in candidates if d.layer in layer_filter]
        if bbox is not None:
            candidates = [d for d in candidates if self._bbox_intersects(d.bbox, bbox)]

        if not candidates:
            return []

        # Cosine similarity (skip any legacy doc that was never embedded)
        embeddings = np.stack([d.embedding for d in candidates if d.embedding is not None])
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        scores = emb_norms @ query_norm.T
        scores = np.atleast_1d(scores.squeeze())  # squeeze() alone collapses a 1-candidate match to 0-d

        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [
            {
                "text": candidates[i].text,
                "layer": candidates[i].layer,
                "score": round(float(scores[i]), 4),
                "metadata": candidates[i].metadata,
            }
            for i in top_indices
        ]

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed_text(self, text: str) -> NDArray[np.floating]:
        """Embed text using sentence-transformers or a hash fallback."""
        try:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer("all-MiniLM-L6-v2")
            embedding: NDArray[np.floating] = encoder.encode([text])[0]
            return embedding
        except ImportError:
            # Deterministic pseudo-embedding from hash
            hash_bytes = hashlib.sha256(text.encode()).digest()
            pseudo = np.array([b / 255.0 for b in hash_bytes], dtype=np.float32)
            pseudo = np.tile(pseudo, self.embed_dim // len(pseudo) + 1)[:self.embed_dim]
            return pseudo

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        if self._use_qdrant:
            assert self._qdrant_client is not None  # only True when client was set
            info = self._qdrant_client.get_collection(self.collection_name)
            return {
                "backend": "qdrant",
                "total_documents": info.points_count,
            }
        else:
            return {
                "backend": "memory",
                "total_documents": len(self._memory_store),
            }
