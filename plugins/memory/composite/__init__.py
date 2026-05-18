"""Composite memory provider shell.

Presents as one external provider to MemoryManager while defensively wrapping
zero or more child providers supplied by config/tests. Safe by default:
- no child providers are instantiated unless explicitly enabled
- child tools are hidden unless tools_enabled=true
- child prompt injection is disabled unless injection_enabled=true
- child failures are fail-open/non-fatal

This module intentionally stays stdlib-only and backend-free.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.memory.composite.shadow import (
    SHADOW_STAGING_HINDSIGHT_BANK_ID,
    SHADOW_STAGING_HONCHO_SESSION_PREFIX,
    SHADOW_STAGING_HONCHO_WORKSPACE,
)
from tools.registry import tool_error

logger = logging.getLogger(__name__)


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _child_enabled_for(value: Any, default: bool = True) -> bool:
    """Interpret per-child read/write gates from YAML-friendly values."""
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"0", "false", "no", "n", "off", "disabled", "none"}:
            return False
    return _as_bool(value, default)


def _child_read_enabled(child_cfg: Dict[str, Any]) -> bool:
    """Return whether a child may participate in read/recall flows."""
    return _child_enabled_for(child_cfg.get("read"), True)


def _normalize_child_configs(children: Any) -> List[Dict[str, Any]]:
    """Accept the Wave 3 dict-keyed config shape while preserving test-injected list specs.

    The planned live shape is:
        children:
          rasputin: {enabled: true, ...}
          hindsight: {enabled: true, ...}
          honcho: {enabled: true, ...}

    Unit tests may still pass a list so they can inject fake provider instances/classes
    without pretending those objects are YAML-serializable live config.
    """
    if isinstance(children, dict):
        normalized: List[Dict[str, Any]] = []
        for name, child_cfg in children.items():
            if not isinstance(child_cfg, dict):
                continue
            merged = dict(child_cfg)
            merged.setdefault("name", str(name))
            normalized.append(merged)
        return normalized
    if isinstance(children, list):
        return [dict(child_cfg) for child_cfg in children if isinstance(child_cfg, dict)]
    return []


class CompositeMemoryProvider(MemoryProvider):
    """Disabled-by-default shell for composing child memory providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = dict(config or {})
        self._session_id = ""
        self._children: List[Dict[str, Any]] = []
        self._tool_map: Dict[str, Dict[str, Any]] = {}
        self._tools_enabled = False
        self._injection_enabled = False

    @property
    def name(self) -> str:
        return "composite"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._children = []
        self._tool_map = {}

        runtime_config = kwargs.get("composite_config")
        if isinstance(runtime_config, dict):
            self._config = dict(runtime_config)

        self._tools_enabled = _as_bool(self._config.get("tools_enabled"), False)
        self._injection_enabled = _as_bool(self._config.get("injection_enabled"), False)

        for child_cfg in _normalize_child_configs(self._config.get("children")):
            if not _as_bool(child_cfg.get("enabled"), False):
                continue

            child_name = str(child_cfg.get("name") or "child").strip() or "child"
            provider = self._build_child_provider(child_cfg)
            if provider is None:
                continue

            try:
                provider.initialize(session_id=session_id, **kwargs)
            except Exception as exc:
                logger.warning(
                    "Composite child '%s' initialize failed (non-fatal): %s",
                    child_name,
                    exc,
                )
                continue

            record = {"name": child_name, "provider": provider, "config": dict(child_cfg)}
            self._children.append(record)

        if self._tools_enabled:
            self._rebuild_tool_map()

    def _build_child_provider(self, child_cfg: Dict[str, Any]) -> Optional[MemoryProvider]:
        provider = child_cfg.get("provider")
        if provider is not None:
            if isinstance(provider, MemoryProvider):
                return provider
            logger.warning("Composite child '%s' provider is not a MemoryProvider instance", child_cfg.get("name", "child"))
            return None

        provider_class = child_cfg.get("provider_class")
        if provider_class is None:
            provider_class = self._provider_class_for_child(str(child_cfg.get("name") or ""))
        if provider_class is None:
            return None

        try:
            provider = provider_class()
        except Exception as exc:
            logger.warning(
                "Composite child '%s' construction failed (non-fatal): %s",
                child_cfg.get("name", "child"),
                exc,
            )
            return None

        if not isinstance(provider, MemoryProvider):
            logger.warning("Composite child '%s' constructed non-MemoryProvider object", child_cfg.get("name", "child"))
            return None
        return provider

    def _provider_class_for_child(self, child_name: str) -> Any | None:
        """Resolve bundled child provider classes from YAML-safe child names.

        Live config cannot contain Python objects, so the production config shape
        uses ``children.<name>.enabled``. Keep this allow-list explicit: the
        composite provider should not import arbitrary modules from config.
        """
        name = child_name.strip().lower().replace("-", "_")
        if name == "hindsight":
            from plugins.memory.hindsight import HindsightMemoryProvider

            return HindsightMemoryProvider
        if name == "honcho":
            from plugins.memory.honcho import HonchoMemoryProvider

            return HonchoMemoryProvider
        logger.warning("Composite child '%s' has no bundled provider factory", child_name)
        return None

    def _rebuild_tool_map(self) -> None:
        self._tool_map = {}
        for child in self._children:
            provider = child["provider"]
            child_name = child["name"]
            try:
                schemas = provider.get_tool_schemas() or []
            except Exception as exc:
                logger.warning(
                    "Composite child '%s' get_tool_schemas failed (non-fatal): %s",
                    child_name,
                    exc,
                )
                continue

            for schema in schemas:
                if not isinstance(schema, dict):
                    continue
                raw_name = str(schema.get("name") or "").strip()
                if not raw_name:
                    continue
                composite_name = self._namespace_tool_name(child_name, raw_name)
                namespaced_schema = copy.deepcopy(schema)
                namespaced_schema["name"] = composite_name
                description = str(namespaced_schema.get("description") or "").strip()
                namespaced_schema["description"] = (
                    f"[Composite:{child_name}] {description}" if description else f"[Composite:{child_name}]"
                )
                self._tool_map[composite_name] = {
                    "provider": provider,
                    "child_name": child_name,
                    "raw_tool_name": raw_name,
                    "schema": namespaced_schema,
                }

    def _namespace_tool_name(self, child_name: str, tool_name: str) -> str:
        return f"composite_{child_name}_{tool_name}"

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        if not self._tools_enabled:
            return []
        return [entry["schema"] for entry in self._tool_map.values()]

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._injection_enabled:
            return ""

        parts = []
        for child in self._children:
            if not _child_read_enabled(child.get("config", {})):
                continue
            try:
                result = child["provider"].prefetch(query, session_id=session_id)
            except Exception as exc:
                logger.debug(
                    "Composite child '%s' prefetch failed (non-fatal): %s",
                    child["name"],
                    exc,
                )
                continue
            if result and str(result).strip():
                parts.append(str(result))
        return "\n\n".join(parts)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._injection_enabled:
            return

        for child in self._children:
            if not _child_read_enabled(child.get("config", {})):
                continue
            try:
                child["provider"].queue_prefetch(query, session_id=session_id)
            except Exception as exc:
                logger.debug(
                    "Composite child '%s' queue_prefetch failed (non-fatal): %s",
                    child["name"],
                    exc,
                )

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        for child in self._children:
            if not _child_enabled_for(child.get("config", {}).get("write"), True):
                continue
            try:
                child["provider"].sync_turn(
                    user_content,
                    assistant_content,
                    session_id=session_id,
                )
            except Exception as exc:
                logger.warning(
                    "Composite child '%s' sync_turn failed (non-fatal): %s",
                    child["name"],
                    exc,
                )

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._tools_enabled:
            return tool_error("Composite memory tools are disabled")

        entry = self._tool_map.get(tool_name)
        if not entry:
            return tool_error(f"Unknown composite memory tool: {tool_name}")

        try:
            return entry["provider"].handle_tool_call(entry["raw_tool_name"], args, **kwargs)
        except Exception as exc:
            logger.warning(
                "Composite child '%s' tool '%s' failed (non-fatal): %s",
                entry["child_name"],
                entry["raw_tool_name"],
                exc,
            )
            return tool_error(f"Composite child tool failed: {exc}")

    def shutdown(self) -> None:
        for child in reversed(self._children):
            try:
                child["provider"].shutdown()
            except Exception as exc:
                logger.debug(
                    "Composite child '%s' shutdown failed (non-fatal): %s",
                    child["name"],
                    exc,
                )

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "enabled",
                "description": "Enable composite provider shell when memory.provider=composite (default: false)",
                "default": False,
            },
            {
                "key": "mode",
                "description": "Composite operating mode placeholder for later waves (default: shadow)",
                "default": "shadow",
            },
            {
                "key": "tools_enabled",
                "description": "Expose child provider tools (default: false)",
                "default": False,
            },
            {
                "key": "injection_enabled",
                "description": "Inject child provider prompt context (default: false)",
                "default": False,
            },
            {
                "key": "canonical_first",
                "description": "Prefer canonical memory before derived providers in later waves (default: true)",
                "default": True,
            },
            {
                "key": "fail_open",
                "description": "Keep normal turns non-fatal on child provider errors (default: true)",
                "default": True,
            },
            {
                "key": "max_prefetch_chars",
                "description": "Maximum merged prefetch characters when injection is enabled (default: 2000)",
                "default": 2000,
            },
            {
                "key": "shadow_write_enabled",
                "description": "Enable fake-client shadow write harness only; not wired to live turns (default: false)",
                "default": False,
            },
            {
                "key": "fake_clients_required",
                "description": "Require injected clients to be explicitly fake-marked before any shadow write (default: true)",
                "default": True,
            },
            {
                "key": "allow_non_fake_staging_clients",
                "description": "Allow non-fake staging clients only after explicit one-off canary approval (default: false)",
                "default": False,
            },
            {
                "key": "hindsight_staging_bank_id",
                "description": "Exact Hindsight staging bank allowed for the gated canary",
                "default": SHADOW_STAGING_HINDSIGHT_BANK_ID,
            },
            {
                "key": "honcho_staging_workspace",
                "description": "Exact Honcho staging workspace allowed for the gated canary",
                "default": SHADOW_STAGING_HONCHO_WORKSPACE,
            },
            {
                "key": "honcho_staging_session_prefix",
                "description": "Exact Honcho staging session namespace prefix allowed for the gated canary",
                "default": SHADOW_STAGING_HONCHO_SESSION_PREFIX,
            },
            {
                "key": "children",
                "description": "Dict-keyed child provider specs, e.g. rasputin/hindsight/honcho (default: {})",
                "default": {},
            },
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        # Intentionally no live config write behavior for this disabled shell.
        _ = (values, hermes_home)


def register(ctx) -> None:
    ctx.register_memory_provider(CompositeMemoryProvider())
