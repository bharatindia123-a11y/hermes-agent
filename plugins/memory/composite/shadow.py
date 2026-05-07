"""Fake-client-first shadow-write harness for memory-vnext staging.

This module is deliberately backend-free. It proves the shape of Hindsight and
Honcho shadow writes against injected fake clients before any later wave is
allowed to touch real services.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Sequence

from agent.memory_events import MemoryEvent


SHADOW_STAGING_HINDSIGHT_BANK_ID = "hermes-thindi-memory-vnext-staging"
SHADOW_STAGING_HONCHO_WORKSPACE = "hermes-thindi-memory-vnext-staging"
SHADOW_STAGING_HONCHO_SESSION_PREFIX = "hermes-thindi-memory-vnext-staging"
SHADOW_STAGING_APPROVAL_KEY = "memory-shadow-staging-canary"
SHADOW_STAGING_APPROVAL_PHRASE = (
    "APPROVE WAVE 7 REAL-STAGING CANARY: authorize one tiny shadow-write batch only, "
    "with fake_clients_required=false for this run only, Hindsight bank "
    "hermes-thindi-memory-vnext-staging, Honcho workspace/session namespace "
    "hermes-thindi-memory-vnext-staging, retain_async=false, injectInferred=false, "
    "fail_open=false, no read cutover, no provider cutover, Rasputin remains live."
)


class ShadowWriteKillSwitchError(RuntimeError):
    """Raised when shadow writes are disabled or a client is not fake-marked."""


@dataclass(frozen=True)
class ShadowWriteResult:
    target: str
    event_id: str
    ok: bool
    skipped: bool = False
    error: str = ""
    response: Any = None


def _event_metadata(event: MemoryEvent) -> dict[str, str]:
    metadata = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "source_uri": event.source_uri,
        "source_locator": event.source_locator,
        "canonical_hash": event.canonical_hash,
        "source_kind": event.source_kind,
        "privacy_class": event.privacy_class,
        "profile": event.profile,
        "fleet": event.fleet,
        "platform": event.platform,
        "session_id": event.session_id,
        "shadow_mode": "fake-client",
    }
    for key, value in event.metadata.items():
        if value is None:
            continue
        metadata.setdefault(str(key), str(value))
    return metadata


def build_hindsight_shadow_payload(
    event: MemoryEvent,
    *,
    bank_id: str,
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the exact fake-client payload for a Hindsight retain call."""

    merged_tags: list[str] = []
    for tag in [*tags, event.event_type, "route:hindsight"]:
        tag = str(tag).strip()
        if tag and tag not in merged_tags:
            merged_tags.append(tag)

    return {
        "bank_id": bank_id,
        "content": event.text,
        "context": event.event_type,
        "document_id": event.event_id,
        "metadata": _event_metadata(event),
        "tags": merged_tags,
        "retain_async": False,
    }


def build_honcho_shadow_payload(
    event: MemoryEvent,
    *,
    workspace: str,
    session_prefix: str = "shadow",
) -> dict[str, Any]:
    """Build the exact fake-client payload for a Honcho message write."""

    session_name = f"{session_prefix}:{event.session_id}" if session_prefix else event.session_id
    metadata = _event_metadata(event)
    metadata.update(
        {
            "source_uid": event.event_id,
            "idempotency_key": event.event_id,
        }
    )
    return {
        "workspace": workspace,
        "session_name": session_name,
        "actor_peer": event.actor_peer,
        "target_peer": event.target_peer,
        "inject_inferred": False,
        "message": {
            "role": event.actor_peer or "user",
            "content": event.text,
        },
        "metadata": metadata,
    }


def _require_fake_client(client: Any, target: str) -> None:
    if client is None:
        raise ShadowWriteKillSwitchError(f"Missing fake {target} shadow client")
    if getattr(client, "is_fake_shadow_client", False) is not True:
        raise ShadowWriteKillSwitchError(
            f"Refusing {target} shadow write: client is not marked is_fake_shadow_client=True"
        )


def _assert_exact_staging_value(name: str, value: str, expected: str) -> None:
    if value != expected:
        raise ShadowWriteKillSwitchError(
            f"Refusing staging shadow write: {name} must be exactly {expected!r}, got {value!r}"
        )


def _known_targets(targets: Iterable[str]) -> set[str]:
    return {str(target) for target in targets if target in {"hindsight", "honcho"}}


def approval_phrase_matches(phrase: str | None) -> bool:
    return phrase == SHADOW_STAGING_APPROVAL_PHRASE


def _memory_provider_from_yaml_text(text: str) -> str:
    in_memory = False
    memory_indent = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "memory:":
            in_memory = True
            memory_indent = indent
            continue
        if in_memory and indent <= memory_indent:
            in_memory = False
        if in_memory and stripped.startswith("provider:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def assert_rasputin_live_configs(
    config_texts: Mapping[str, str],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> None:
    """Pure no-cutover guard for later real-staging canary pre/post checks."""

    for path, text in config_texts.items():
        provider = _memory_provider_from_yaml_text(text)
        if provider != "rasputin":
            raise ShadowWriteKillSwitchError(
                f"Refusing staging canary: {path} memory.provider must remain 'rasputin', got {provider!r}"
            )
        if expected_sha256 is not None:
            digest = hashlib.sha256(text.encode()).hexdigest()
            expected = expected_sha256.get(path)
            if expected is not None and digest != expected:
                raise ShadowWriteKillSwitchError(
                    f"Refusing staging canary: {path} sha256 drifted from expected live config hash"
                )


class ShadowWriteHarness:
    """Run fake-only shadow writes for canonical memory events.

    The harness is intentionally not wired into live turns. Callers must pass
    injected fake clients and set enabled=True; otherwise no write method is
    reached.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        hindsight_client: Any = None,
        honcho_client: Any = None,
        hindsight_bank_id: str = SHADOW_STAGING_HINDSIGHT_BANK_ID,
        honcho_workspace: str = SHADOW_STAGING_HONCHO_WORKSPACE,
        honcho_session_prefix: str = SHADOW_STAGING_HONCHO_SESSION_PREFIX,
        hindsight_tags: Sequence[str] = ("memory-vnext", "shadow"),
        fail_open: bool = True,
        allow_non_fake_staging_clients: bool = False,
        staging_approval_check: Callable[[str], bool] | None = None,
        staging_approval_key: str = SHADOW_STAGING_APPROVAL_KEY,
        staging_approval_phrase: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.hindsight_client = hindsight_client
        self.honcho_client = honcho_client
        self.hindsight_bank_id = hindsight_bank_id
        self.honcho_workspace = honcho_workspace
        self.honcho_session_prefix = honcho_session_prefix
        self.hindsight_tags = tuple(hindsight_tags)
        self.fail_open = bool(fail_open)
        self.allow_non_fake_staging_clients = bool(allow_non_fake_staging_clients)
        self.staging_approval_check = staging_approval_check
        self.staging_approval_key = staging_approval_key
        self.staging_approval_phrase = staging_approval_phrase

    def write_events(
        self,
        events: Iterable[MemoryEvent],
        *,
        targets: Sequence[str] | None = None,
    ) -> list[ShadowWriteResult]:
        if not self.enabled:
            raise ShadowWriteKillSwitchError("Shadow-write harness disabled by kill switch")

        results: list[ShadowWriteResult] = []
        target_filter = tuple(targets) if targets is not None else None
        events = list(events)
        self._preflight(events, target_filter)
        for event in events:
            event_targets = target_filter or tuple(event.routing_hints)
            for target in event_targets:
                if target == "hindsight":
                    results.append(self._write_hindsight(event))
                elif target == "honcho":
                    results.append(self._write_honcho(event))
                else:
                    results.append(
                        ShadowWriteResult(
                            target=str(target),
                            event_id=event.event_id,
                            ok=True,
                            skipped=True,
                        )
                    )
        return results

    def _preflight(self, events: Sequence[MemoryEvent], target_filter: Sequence[str] | None) -> None:
        selected_targets: set[str] = set()
        for event in events:
            selected_targets.update(_known_targets(target_filter or event.routing_hints))

        if "hindsight" in selected_targets:
            _assert_exact_staging_value(
                "hindsight_bank_id",
                self.hindsight_bank_id,
                SHADOW_STAGING_HINDSIGHT_BANK_ID,
            )
            self._require_allowed_client(self.hindsight_client, "hindsight")

        if "honcho" in selected_targets:
            _assert_exact_staging_value(
                "honcho_workspace",
                self.honcho_workspace,
                SHADOW_STAGING_HONCHO_WORKSPACE,
            )
            _assert_exact_staging_value(
                "honcho_session_prefix",
                self.honcho_session_prefix,
                SHADOW_STAGING_HONCHO_SESSION_PREFIX,
            )
            self._require_allowed_client(self.honcho_client, "honcho")

    def _require_allowed_client(self, client: Any, target: str) -> None:
        if client is None:
            raise ShadowWriteKillSwitchError(f"Missing {target} shadow client")
        if getattr(client, "is_fake_shadow_client", False) is True:
            return
        if not self.allow_non_fake_staging_clients:
            raise ShadowWriteKillSwitchError(
                f"Refusing {target} shadow write: client is not marked is_fake_shadow_client=True"
            )
        if not approval_phrase_matches(self.staging_approval_phrase):
            raise ShadowWriteKillSwitchError(
                "Refusing staging shadow write: exact Wave 7 real-staging canary approval phrase missing"
            )
        if self.staging_approval_check is not None and self.staging_approval_check(self.staging_approval_key) is not True:
            raise ShadowWriteKillSwitchError(
                f"Refusing {target} staging shadow write: explicit approval {self.staging_approval_key!r} missing"
            )

    def _write_hindsight(self, event: MemoryEvent) -> ShadowWriteResult:
        try:
            _assert_exact_staging_value(
                "hindsight_bank_id",
                self.hindsight_bank_id,
                SHADOW_STAGING_HINDSIGHT_BANK_ID,
            )
            self._require_allowed_client(self.hindsight_client, "hindsight")
            payload = build_hindsight_shadow_payload(
                event,
                bank_id=self.hindsight_bank_id,
                tags=self.hindsight_tags,
            )
            response = self.hindsight_client.retain(**payload)
            return ShadowWriteResult("hindsight", event.event_id, True, response=response)
        except ShadowWriteKillSwitchError:
            raise
        except Exception as exc:
            if not self.fail_open:
                raise
            return ShadowWriteResult("hindsight", event.event_id, False, error=str(exc))

    def _write_honcho(self, event: MemoryEvent) -> ShadowWriteResult:
        try:
            _assert_exact_staging_value(
                "honcho_workspace",
                self.honcho_workspace,
                SHADOW_STAGING_HONCHO_WORKSPACE,
            )
            _assert_exact_staging_value(
                "honcho_session_prefix",
                self.honcho_session_prefix,
                SHADOW_STAGING_HONCHO_SESSION_PREFIX,
            )
            self._require_allowed_client(self.honcho_client, "honcho")
            payload = build_honcho_shadow_payload(
                event,
                workspace=self.honcho_workspace,
                session_prefix=self.honcho_session_prefix,
            )
            response = self.honcho_client.add_shadow_message(**payload)
            return ShadowWriteResult("honcho", event.event_id, True, response=response)
        except ShadowWriteKillSwitchError:
            raise
        except Exception as exc:
            if not self.fail_open:
                raise
            return ShadowWriteResult("honcho", event.event_id, False, error=str(exc))
