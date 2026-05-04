"""
Episodic Memory store — Memory-R1 style CRUD operations on a persistent
trajectory collection.

Architecture lineage:
  v1 (2023, deprecated): Reflexion-style append-only — only knew ADD.
                         Suffered from duplicate accumulation, stale wrong
                         root_causes lingering forever, retrieval noise.
  v2 (current, 2025+):   Memory-R1 style — supports {ADD, UPDATE, DELETE,
                         NOOP} operations driven by an LLM Memory Manager
                         node, with soft delete for safety.

Backed by a local-persistent ChromaDB collection. Embeddings come from
Chroma's bundled sentence-transformers (all-MiniLM-L6-v2, ~80MB,
auto-downloaded on first use).

Why local + sentence-transformers:
  - Zero API cost (no OpenAI embedding calls)
  - Works offline once the model is downloaded
  - The texts we embed (error signatures, ~200 chars) are short, MiniLM
    is more than adequate; OpenAI text-embedding-3 would be overkill.

Lifecycle:
  - The Reflector node calls `retrieve_similar(...)` BEFORE its LLM call
    (memory-augmented self-healing)
  - The Memory Manager node calls add/update/mark_deprecated AFTER each
    Reflector run, based on the LLM's CRUD operation decision
  - retrieve_similar() filters out deprecated rows by default, but the
    underlying data is preserved (soft delete)
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from auto_sdet.models.schemas import MemoryTrajectory

logger = logging.getLogger(__name__)


# Default location: project-root/.auto_sdet_memory/
# Add to .gitignore — these are runtime artifacts, not source.
DEFAULT_PERSIST_DIR = Path(".auto_sdet_memory")
COLLECTION_NAME = "reflector_trajectories"

# Below this similarity, the retrieved memory is too unrelated to be useful
# and would only add noise. Threshold tuned conservatively — it's easy to
# inject 0 memories (no harm), but injecting an unrelated one can mislead.
SIMILARITY_THRESHOLD = 0.4


class MemoryStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Lazily initializes the chromadb client on first use so that import-time
    overhead stays small (chromadb pulls in numpy + sentence-transformers
    which together can take ~1s to load).
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        self._persist_dir = persist_dir or DEFAULT_PERSIST_DIR
        self._client = None       # lazy
        self._collection = None   # lazy

    # ──────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────

    # ─── ADD ──────────────────────────────────────────────

    def add_trajectory(self, traj: MemoryTrajectory) -> None:
        """ADD: persist a brand-new trajectory. Idempotent on trajectory_id."""
        coll = self._get_collection()
        try:
            coll.add(
                ids=[traj.trajectory_id],
                documents=[traj.error_signature],     # what gets embedded
                metadatas=[self._traj_to_metadata(traj)],
            )
        except Exception as e:
            # Memory failures should never break the agent — log and move on
            logger.warning(f"MemoryStore.add_trajectory failed: {e}")

    # ─── UPDATE ───────────────────────────────────────────

    def update_trajectory(
        self,
        target_id: str,
        new_traj: MemoryTrajectory,
    ) -> bool:
        """
        UPDATE: replace the contents of an existing trajectory in place.

        Keeps the same trajectory_id (so any external references stay valid).
        Re-embeds the document with new_traj.error_signature so future
        retrieval reflects the corrected understanding.

        Returns True on success, False if the target doesn't exist.
        """
        coll = self._get_collection()
        try:
            existing = coll.get(ids=[target_id])
            if not existing.get("ids"):
                logger.warning(f"UPDATE failed: trajectory_id {target_id} not found")
                return False
            coll.update(
                ids=[target_id],
                documents=[new_traj.error_signature],
                metadatas=[self._traj_to_metadata(new_traj)],
            )
            return True
        except Exception as e:
            logger.warning(f"MemoryStore.update_trajectory failed: {e}")
            return False

    # ─── DELETE (soft) ────────────────────────────────────

    def mark_deprecated(self, target_id: str, reason: str) -> bool:
        """
        SOFT DELETE: flip the `deprecated` flag, keep the data.

        Why soft instead of hard delete:
          - The Memory Manager LLM can be wrong; recoverability matters.
          - Audit trail: the `deprecated_reason` records WHY this memory
            was removed, useful for debugging the Manager itself.
          - retrieve_similar() filters deprecated rows by default, so
            soft-deleted rows are functionally invisible to consumers.
        """
        coll = self._get_collection()
        try:
            existing = coll.get(ids=[target_id])
            if not existing.get("ids"):
                logger.warning(f"DELETE failed: trajectory_id {target_id} not found")
                return False
            current_meta = existing["metadatas"][0]
            current_meta["deprecated"] = True
            current_meta["deprecated_reason"] = reason
            coll.update(ids=[target_id], metadatas=[current_meta])
            return True
        except Exception as e:
            logger.warning(f"MemoryStore.mark_deprecated failed: {e}")
            return False

    # ─── READ ─────────────────────────────────────────────

    def get_by_id(self, trajectory_id: str) -> Optional[MemoryTrajectory]:
        """READ a single trajectory by id, including deprecated ones."""
        coll = self._get_collection()
        try:
            result = coll.get(ids=[trajectory_id])
            if not result.get("ids"):
                return None
            return self._metadata_to_traj(trajectory_id, result["metadatas"][0])
        except Exception as e:
            logger.warning(f"MemoryStore.get_by_id failed: {e}")
            return None

    def retrieve_similar(
        self,
        query_text: str,
        k: int = 3,
        include_deprecated: bool = False,
    ) -> list[MemoryTrajectory]:
        """
        Return up to `k` past trajectories whose error_signature is most
        semantically similar to `query_text`.

        Filters:
          - Below SIMILARITY_THRESHOLD: noise, drop
          - deprecated=True: drop unless caller explicitly opts in
            (Memory Manager may want to see all neighbors including
            deprecated ones to make a CRUD decision).

        Over-fetches by 2x to compensate for post-query filtering.
        """
        coll = self._get_collection()
        try:
            count = coll.count()
        except Exception as e:
            logger.warning(f"MemoryStore.count failed: {e}")
            return []

        if count == 0:
            return []   # cold start — no history yet

        # Over-fetch: filtering out deprecated/noise might leave us short
        try:
            result = coll.query(
                query_texts=[query_text],
                n_results=min(k * 2, count),
            )
        except Exception as e:
            logger.warning(f"MemoryStore.retrieve_similar failed: {e}")
            return []

        # Chroma returns parallel arrays under the first batch (we sent 1 query)
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        keep: list[MemoryTrajectory] = []
        for tid, meta, dist in zip(ids, metadatas, distances):
            # Chroma returns L2 distance by default; smaller = more similar.
            # Convert to a rough similarity score in [0, 1] for thresholding.
            similarity = 1.0 / (1.0 + dist)
            if similarity < SIMILARITY_THRESHOLD:
                continue
            if not include_deprecated and meta.get("deprecated", False):
                continue
            try:
                keep.append(self._metadata_to_traj(tid, meta))
            except Exception as e:
                logger.warning(f"Skipping malformed memory {tid}: {e}")
            if len(keep) >= k:
                break
        return keep

    def count(self, include_deprecated: bool = False) -> int:
        """
        Trajectory count.

        By default counts only LIVE trajectories (deprecated excluded), so
        the number reflects what consumers actually see via retrieval.
        """
        try:
            coll = self._get_collection()
            if include_deprecated:
                return coll.count()
            # Active count requires fetching metadata
            all_records = coll.get()
            return sum(
                1 for m in all_records.get("metadatas", [])
                if not m.get("deprecated", False)
            )
        except Exception:
            return 0

    # ──────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────

    def _get_collection(self):
        """Lazy-init the chromadb collection."""
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "chromadb is required for episodic memory. "
                "Install it with: pip install chromadb"
            ) from e

        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            # Use chroma's default embedding (sentence-transformers
            # all-MiniLM-L6-v2). It's downloaded the first time chroma
            # needs it and cached locally afterward.
        )
        return self._collection

    @staticmethod
    def _traj_to_metadata(traj: MemoryTrajectory) -> dict:
        """
        Convert MemoryTrajectory -> chromadb metadata dict.

        Chroma supports str / int / float / bool values in metadata.
        None must be coerced (chroma rejects None values), so
        deprecated_reason is normalized to empty string when absent.
        """
        return {
            "timestamp": traj.timestamp,
            "target_file": traj.target_file,
            "error_signature": traj.error_signature,
            "error_classification": traj.error_classification,
            "root_cause": traj.root_cause,
            "fix_strategy": traj.fix_strategy,
            "fix_summary": traj.fix_summary,
            "outcome": traj.outcome,
            "deprecated": traj.deprecated,
            "deprecated_reason": traj.deprecated_reason or "",
        }

    @staticmethod
    def _metadata_to_traj(tid: str, meta: dict) -> MemoryTrajectory:
        """Reverse of _traj_to_metadata. Raises Pydantic ValidationError on bad data."""
        return MemoryTrajectory(
            trajectory_id=tid,
            timestamp=meta["timestamp"],
            target_file=meta["target_file"],
            error_signature=meta["error_signature"],
            error_classification=meta["error_classification"],
            root_cause=meta["root_cause"],
            fix_strategy=meta["fix_strategy"],
            fix_summary=meta["fix_summary"],
            outcome=meta["outcome"],
            deprecated=meta.get("deprecated", False),
            deprecated_reason=meta.get("deprecated_reason") or None,
        )


# ──────────────────────────────────────────────────────────────────
# Helpers used by the Reflector node
# ──────────────────────────────────────────────────────────────────

def build_error_signature(
    error_classification: str,
    root_cause: str,
    target_file: str,
) -> str:
    """
    Deterministically construct the embedding-friendly signature string.

    Uses just enough of root_cause to be discriminative without dragging
    in pytest header noise. Target file is included so retrieval can boost
    same-file follow-up runs slightly.
    """
    short_cause = root_cause.split(".", 1)[0][:200]
    return f"{error_classification}: {short_cause} [in {target_file}]"


def build_trajectory(
    *,
    error_classification: str,
    root_cause: str,
    fix_strategy: str,
    fix_summary: str,
    target_file: str,
    outcome: str = "fix_applied",
) -> MemoryTrajectory:
    """Construct a MemoryTrajectory with auto-filled id/timestamp/signature."""
    return MemoryTrajectory(
        trajectory_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        target_file=target_file,
        error_signature=build_error_signature(
            error_classification, root_cause, target_file
        ),
        error_classification=error_classification,
        root_cause=root_cause,
        fix_strategy=fix_strategy,
        fix_summary=fix_summary,
        outcome=outcome,   # type: ignore[arg-type]  # Literal narrowed at runtime
    )