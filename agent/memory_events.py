from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SECRET_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]+", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9][A-Za-z0-9._\-]{6,}\b"),
    re.compile(r"\b(api[_-]?key\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"\b(password\s*[:=]\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"\b((?:access[_-]?token|refresh[_-]?token|token|secret|client[_-]?secret)\s*=\s*)([^\s,;]+)", re.IGNORECASE),
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:@/]+:[^\s@/]+@[^\s]+", re.IGNORECASE),
)

_HONCHO_SOURCE_KINDS = {
    "canonical_user",
    "workspace_user",
    "user_profile",
}
_HINDSIGHT_EVENT_TYPES = {"memory", "fact", "decision", "pattern", "history", "incident", "gotcha"}
_HONCHO_EVENT_TYPES = {"user", "preference", "goal", "identity", "style"}
_HONCHO_KEYWORDS = ("user preference", "prefers", "likes", "dislikes", "working style", "communication style")
_EXPLICIT_USER_MARKERS = (
    "user",
    "user.md",
    "user_profile",
    "user profile",
    "ashwin",
    "ashooo",
    "preference correction",
)


def _stable_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def redact_text(text: str) -> str:
    redacted = text
    redacted = _SECRET_PATTERNS[0].sub("Bearer [REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[1].sub("[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[2].sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[3].sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[4].sub(lambda m: f"{m.group(1)}[REDACTED]", redacted)
    redacted = _SECRET_PATTERNS[5].sub("[REDACTED]", redacted)
    return redacted


def canonicalize_text(text: str) -> str:
    normalized = redact_text(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def compute_canonical_hash(
    *,
    event_type: str,
    source_kind: str,
    source_uri: str,
    source_locator: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> str:
    canonical_text = canonicalize_text(text)
    canonical_text = re.sub(r"\n{2,}", "\n", canonical_text)
    payload = {
        "event_type": event_type,
        "source_kind": source_kind,
        "source_uri": source_uri,
        "source_locator": source_locator,
        "text": canonical_text,
        "metadata": metadata or {},
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def deterministic_source_uid(
    *,
    profile: str,
    fleet: str,
    platform: str,
    session_id: str,
    source_kind: str,
    source_uri: str,
    source_locator: str,
    canonical_hash: str,
) -> str:
    payload = {
        "canonical_hash": canonical_hash,
        "fleet": fleet,
        "platform": platform,
        "profile": profile,
        "session_id": session_id,
        "source_kind": source_kind,
        "source_locator": source_locator,
        "source_uri": source_uri,
    }
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def route_hints_for_source(
    source_kind: str,
    text: str = "",
    metadata: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    metadata = metadata or {}
    lowered_text = canonicalize_text(text).lower()
    event_type = str(metadata.get("event_type") or metadata.get("kind") or "").lower()
    metadata_blob = _stable_json(metadata).lower() if metadata else ""

    def _has_explicit_user_context() -> bool:
        if source_kind in _HONCHO_SOURCE_KINDS:
            return True
        return any(marker in lowered_text or marker in metadata_blob for marker in _EXPLICIT_USER_MARKERS)

    wants_hindsight = source_kind not in _HONCHO_SOURCE_KINDS or event_type in _HINDSIGHT_EVENT_TYPES
    wants_honcho = source_kind in _HONCHO_SOURCE_KINDS

    if _has_explicit_user_context() and (
        event_type in _HONCHO_EVENT_TYPES or any(keyword in lowered_text for keyword in _HONCHO_KEYWORDS)
    ):
        wants_honcho = True
    if any(token in lowered_text for token in ("decision:", "fact:", "history:", "incident:", "gotcha:")):
        wants_hindsight = True

    hints: list[str] = []
    if wants_hindsight:
        hints.append("hindsight")
    if wants_honcho:
        hints.append("honcho")
    return tuple(hints)


@dataclass(frozen=True)
class MemoryEvent:
    event_id: str
    event_type: str
    profile: str
    fleet: str
    platform: str
    session_id: str
    source_kind: str
    source_uri: str
    source_locator: str
    canonical_hash: str
    actor_peer: str
    target_peer: str | None
    occurred_at: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    privacy_class: str = "internal"
    routing_hints: tuple[str, ...] = field(default_factory=tuple)


def create_memory_event(
    *,
    event_type: str,
    profile: str,
    fleet: str,
    platform: str,
    session_id: str,
    source_kind: str,
    source_uri: str,
    source_locator: str,
    occurred_at: str,
    text: str,
    metadata: dict[str, Any] | None = None,
    actor_peer: str = "user",
    target_peer: str | None = None,
    privacy_class: str = "internal",
) -> MemoryEvent:
    metadata = dict(metadata or {})
    redacted_text = canonicalize_text(text)
    routing_hints = route_hints_for_source(
        source_kind,
        text=redacted_text,
        metadata={**metadata, "event_type": event_type, "source_uri": source_uri},
    )
    canonical_hash = compute_canonical_hash(
        event_type=event_type,
        source_kind=source_kind,
        source_uri=source_uri,
        source_locator=source_locator,
        text=redacted_text,
        metadata=metadata,
    )
    event_id = deterministic_source_uid(
        profile=profile,
        fleet=fleet,
        platform=platform,
        session_id=session_id,
        source_kind=source_kind,
        source_uri=source_uri,
        source_locator=source_locator,
        canonical_hash=canonical_hash,
    )
    return MemoryEvent(
        event_id=event_id,
        event_type=event_type,
        profile=profile,
        fleet=fleet,
        platform=platform,
        session_id=session_id,
        source_kind=source_kind,
        source_uri=source_uri,
        source_locator=source_locator,
        canonical_hash=canonical_hash,
        actor_peer=actor_peer,
        target_peer=target_peer,
        occurred_at=occurred_at,
        text=redacted_text,
        metadata=metadata,
        privacy_class=privacy_class,
        routing_hints=routing_hints,
    )


@dataclass(frozen=True)
class SourceManifestEntry:
    source_kind: str
    source_uri: str
    source_path: str
    importable: bool
    discovered_count: int
    importable_count: int
    skipped_count: int
    quarantined_count: int
    content_hash: str
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_hash": self.content_hash,
            "discovered_count": self.discovered_count,
            "importable": self.importable,
            "importable_count": self.importable_count,
            "notes": list(self.notes),
            "quarantined_count": self.quarantined_count,
            "skipped_count": self.skipped_count,
            "source_kind": self.source_kind,
            "source_path": self.source_path,
            "source_uri": self.source_uri,
        }


def hash_file_content(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
