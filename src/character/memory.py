"""Local, structured scene memory -- demo moment 4, and the recall half of
demo moment 5's remember_object/recall_memory Live tools.

The challenge only requires memory to survive within one continuous demo
(CHALLENGE.md: "retains useful information... later answers a spoken
question about it"), not across restarts -- disk persistence here is a
design choice for observability/demonstrability, not a challenge
requirement: it means the last run's memory is directly inspectable
(cat var/scene_memory.json) and a demo can show "it still knew this from
a minute ago" without needing to keep one process running the whole time.
Tests construct SceneMemory with persist_path=None and never touch disk.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.protocol.observation import DetectedObject

MAX_STRING_LEN = 200
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _sanitize(text: str, max_len: int = MAX_STRING_LEN) -> str:
    """Strips control characters and clamps length -- memory records come
    from model output (either a vision call's JSON or a tool-call
    argument), and while both are schema-validated, this is a second,
    cheap guard against anything weird making it into a persisted file."""
    cleaned = _CONTROL_CHARS_RE.sub("", text).strip()
    return cleaned[:max_len]


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1, max_length=80)
    color: str | None = Field(default=None, max_length=40)
    attributes: str = Field(default="", max_length=MAX_STRING_LEN)
    image_x: float = Field(ge=0.0, le=1.0)
    image_y: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    source_observation_id: str = Field(min_length=1, max_length=64)
    notes: str | None = Field(default=None, max_length=MAX_STRING_LEN)
    first_seen: float
    last_seen: float

    def dedup_key(self) -> tuple[str, str]:
        return (self.label.strip().lower(), (self.color or "").strip().lower())


@dataclass
class _LoadResult:
    records: list[MemoryRecord] = field(default_factory=list)
    backed_up_corrupt_file: Path | None = None


class SceneMemory:
    def __init__(
        self,
        max_entries: int = 50,
        persist_path: Path | None = None,
        on_debug: callable | None = None,
    ):
        self._max_entries = max_entries
        self._persist_path = Path(persist_path) if persist_path is not None else None
        self._on_debug = on_debug or (lambda msg: None)
        self._lock = threading.Lock()
        self._records: list[MemoryRecord] = []
        if self._persist_path is not None:
            result = self._load()
            self._records = result.records
            if result.backed_up_corrupt_file:
                self._on_debug(
                    f"scene memory: existing file was corrupt, backed up to "
                    f"{result.backed_up_corrupt_file}, starting fresh"
                )

    # -- writes ----------------------------------------------------------------

    def remember(self, objects: list[DetectedObject], source_observation_id: str) -> bool:
        """Bulk upsert from a scene scan -- returns whether anything
        actually changed (new label+color, or an existing one's
        attributes/position changed), so a caller can decide whether it's
        worth telling the model about (see fsm.py)."""
        changed = False
        for obj in objects:
            _, was_new_or_changed = self._upsert(
                label=obj.label,
                color=obj.color,
                attributes=obj.attributes,
                image_x=obj.image_x,
                image_y=obj.image_y,
                confidence=obj.confidence,
                source_observation_id=source_observation_id,
                notes=obj.notes,
            )
            changed = changed or was_new_or_changed
        if changed:
            self._save()
        return changed

    def remember_one(
        self,
        *,
        label: str,
        color: str | None,
        attributes: str,
        image_x: float,
        image_y: float,
        confidence: float,
        source_observation_id: str,
        notes: str | None,
    ) -> MemoryRecord:
        """The remember_object Live tool's path -- one object, explicitly
        named by the model, tied to a specific (caller-verified-fresh)
        observation_id."""
        record, _ = self._upsert(
            label=label,
            color=color,
            attributes=attributes,
            image_x=image_x,
            image_y=image_y,
            confidence=confidence,
            source_observation_id=source_observation_id,
            notes=notes,
        )
        self._save()
        return record

    def _upsert(
        self,
        *,
        label: str,
        color: str | None,
        attributes: str,
        image_x: float,
        image_y: float,
        confidence: float,
        source_observation_id: str,
        notes: str | None,
    ) -> tuple[MemoryRecord, bool]:
        now = time.time()
        try:
            candidate = MemoryRecord(
                label=_sanitize(label, 80),
                color=_sanitize(color, 40) if color else None,
                attributes=_sanitize(attributes),
                image_x=image_x,
                image_y=image_y,
                confidence=confidence,
                source_observation_id=_sanitize(source_observation_id, 64),
                notes=_sanitize(notes) if notes else None,
                first_seen=now,
                last_seen=now,
            )
        except ValidationError as exc:
            raise ValueError(f"malformed memory record rejected: {exc}") from exc

        with self._lock:
            key = candidate.dedup_key()
            for i, existing in enumerate(self._records):
                if existing.dedup_key() == key:
                    changed = (
                        existing.attributes != candidate.attributes
                        or existing.image_x != candidate.image_x
                        or existing.image_y != candidate.image_y
                    )
                    updated = existing.model_copy(
                        update={
                            "attributes": candidate.attributes,
                            "image_x": candidate.image_x,
                            "image_y": candidate.image_y,
                            "confidence": candidate.confidence,
                            "source_observation_id": candidate.source_observation_id,
                            "notes": candidate.notes,
                            "last_seen": now,
                        }
                    )
                    self._records[i] = updated
                    return updated, changed
            self._records.append(candidate)
            self._records = self._records[-self._max_entries :]
            return candidate, True

    # -- reads -------------------------------------------------------------------

    def recall(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        """Concise token-overlap matching against label/color/attributes/
        notes -- no embeddings/vector search, matching this project's
        "no infrastructure the demo doesn't need" stance (see the old
        as_context_text() docstring this replaces). Falls back to the
        most recently seen records if nothing matches any token, so
        recall_memory("what did you see") still returns something
        sensible rather than an empty list."""
        tokens = {t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 2}
        with self._lock:
            records = list(self._records)
        if tokens:
            scored = []
            for r in records:
                haystack = f"{r.label} {r.color or ''} {r.attributes} {r.notes or ''}".lower()
                score = sum(1 for t in tokens if t in haystack)
                if score > 0:
                    scored.append((score, r.last_seen, r))
            if scored:
                scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
                return [r for _, _, r in scored[:limit]]
        records.sort(key=lambda r: r.last_seen, reverse=True)
        return records[:limit]

    def all_records(self) -> list[MemoryRecord]:
        with self._lock:
            return list(self._records)

    def as_context_text(self) -> str:
        """Human-readable summary -- used only for debug logging and the
        reconnect recap (see live_client.py), never injected as a live
        conversational nudge (that's what caused unsolicited narration;
        recall now goes through the recall_memory tool instead)."""
        records = self.all_records()
        if not records:
            return "No objects remembered yet."
        lines = [f"- {r.label}" + (f" ({r.color})" if r.color else "") + f": {r.attributes}" for r in records]
        return "Remembered objects:\n" + "\n".join(lines)

    def __len__(self) -> int:
        return len(self._records)

    # -- persistence -------------------------------------------------------------

    def _load(self) -> _LoadResult:
        path = self._persist_path
        assert path is not None
        if not path.exists():
            return _LoadResult(records=[])
        try:
            raw = path.read_text()
            payload = json.loads(raw)
            records = [MemoryRecord.model_validate(item) for item in payload.get("records", [])]
            return _LoadResult(records=records)
        except (json.JSONDecodeError, ValidationError, OSError, KeyError, TypeError) as exc:
            backup_path = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            try:
                shutil.copy2(path, backup_path)
            except OSError:
                backup_path = None
            self._on_debug(f"scene memory: failed to load {path}: {exc}")
            return _LoadResult(records=[], backed_up_corrupt_file=backup_path)

    def _save(self) -> None:
        path = self._persist_path
        if path is None:
            return
        with self._lock:
            payload = {"records": [r.model_dump(mode="json") for r in self._records]}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2))
            os.replace(tmp_path, path)  # atomic on POSIX -- no half-written file on crash
        except OSError as exc:
            self._on_debug(f"scene memory: failed to persist to {path}: {exc}")
