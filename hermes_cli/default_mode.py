"""Helpers for OMO-style default-mode auto activation.

This module intentionally keeps the v4.3.0 OMO feature mapped to Hermes-native
surfaces: a prompt overlay for ultrawork posture and the existing ``/goal``
GoalManager for Ralph-loop continuation.  It must stay opt-in and only be
called by top-level session adapters (CLI/gateway/API), not by AIAgent itself
or delegated children.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

ULTRAWORK_PROMPT_MARKER = "<ultrawork-mode>"

ULTRAWORK_PROMPT_BLOCK = """<ultrawork-mode>
Default ultrawork mode is active for this top-level session.

Operate in a high-agency execution posture:
- bias toward concrete progress over passive advice;
- decompose multi-step work, use tools, and verify results before claiming completion;
- keep working until the user's request is actually complete or a real blocker requires user input;
- preserve normal safety/approval gates for destructive or production-impacting actions.
</ultrawork-mode>"""


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def default_mode_config(config: Optional[Mapping[str, Any]]) -> dict[str, bool]:
    """Return normalized default_mode booleans from a loaded config mapping."""

    raw = _as_mapping(_as_mapping(config).get("default_mode"))
    return {
        "ultrawork": _as_bool(raw.get("ultrawork", False)),
        "ralph_loop": _as_bool(raw.get("ralph_loop", False)),
    }


def is_default_ultrawork_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    return default_mode_config(config)["ultrawork"]


def is_default_ralph_loop_enabled(config: Optional[Mapping[str, Any]]) -> bool:
    return default_mode_config(config)["ralph_loop"]


def append_ultrawork_prompt(existing: Optional[str]) -> str:
    """Append the ultrawork overlay once, preserving any existing prompt text."""

    current = (existing or "").strip()
    if ULTRAWORK_PROMPT_MARKER in current:
        return current
    if not current:
        return ULTRAWORK_PROMPT_BLOCK
    return f"{current}\n\n{ULTRAWORK_PROMPT_BLOCK}"


def activate_ultrawork_on_agent(agent: Any) -> bool:
    """Apply the ultrawork prompt overlay to an already-created top-level agent.

    Returns True when the overlay was added and False when it was already
    present or the target object is unusable.  Callers are responsible for
    enforcing top-level/main-session-only semantics before invoking this.
    """

    if agent is None:
        return False
    current = getattr(agent, "ephemeral_system_prompt", None) or ""
    updated = append_ultrawork_prompt(current)
    if updated == current:
        return False
    try:
        setattr(agent, "ephemeral_system_prompt", updated)
        setattr(agent, "_default_mode_ultrawork", True)
    except Exception:
        return False
    return True


__all__ = [
    "ULTRAWORK_PROMPT_BLOCK",
    "ULTRAWORK_PROMPT_MARKER",
    "activate_ultrawork_on_agent",
    "append_ultrawork_prompt",
    "default_mode_config",
    "is_default_ralph_loop_enabled",
    "is_default_ultrawork_enabled",
]
