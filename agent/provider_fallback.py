"""Provider fallback helpers.

Small, pure helpers for Hermes' runtime provider fallback path.  These
capture the OMO v4.3.0-inspired behavior we want without importing OMO's
TypeScript implementation: disabled-provider pruning, runtime-fallback
configuration defaults, and structured provenance records.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable, Mapping


DEFAULT_RUNTIME_FALLBACK_CONFIG: dict[str, Any] = {
    "enabled": True,
    "retry_on_errors": [401, 402, 403, 404, 429, 500, 502, 503, 529],
    "max_fallback_attempts": None,
    "cooldown_seconds": 60,
    "timeout_seconds": None,
    "notify_on_fallback": True,
}


def normalize_provider_name(value: Any) -> str:
    """Return a canonical provider key for comparisons."""
    return str(value or "").strip().lower()


def normalize_model_name(value: Any) -> str:
    return str(value or "").strip()


def normalize_disabled_providers(values: Any) -> set[str]:
    """Normalize disabled-provider config values.

    Supports YAML-friendly shapes:
    - disabled_providers: [openai, anthropic]
    - providers.disabled: [openai]
    - providers.openai.disabled: true
    """
    disabled: set[str] = set()
    if isinstance(values, str):
        values = [values]
    if isinstance(values, Iterable) and not isinstance(values, (bytes, bytearray, Mapping)):
        for value in values:
            name = normalize_provider_name(value)
            if name:
                disabled.add(name)
    return disabled


def disabled_providers_from_config(config: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(config, Mapping):
        return set()
    disabled = normalize_disabled_providers(config.get("disabled_providers"))
    providers = config.get("providers")
    if isinstance(providers, Mapping):
        disabled |= normalize_disabled_providers(providers.get("disabled"))
        for name, provider_cfg in providers.items():
            if name == "disabled":
                continue
            if isinstance(provider_cfg, Mapping) and bool(provider_cfg.get("disabled")):
                canonical = normalize_provider_name(name)
                if canonical:
                    disabled.add(canonical)
    return disabled


def runtime_fallback_config_from_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_RUNTIME_FALLBACK_CONFIG)
    if isinstance(config, Mapping):
        raw = config.get("runtime_fallback")
        if isinstance(raw, Mapping):
            merged.update(dict(raw))
    merged["enabled"] = bool(merged.get("enabled", True))
    try:
        if merged.get("max_fallback_attempts") is not None:
            merged["max_fallback_attempts"] = max(0, int(merged["max_fallback_attempts"]))
    except (TypeError, ValueError):
        merged["max_fallback_attempts"] = None
    try:
        merged["cooldown_seconds"] = max(0, int(merged.get("cooldown_seconds", 60)))
    except (TypeError, ValueError):
        merged["cooldown_seconds"] = 60
    if not isinstance(merged.get("retry_on_errors"), list):
        merged["retry_on_errors"] = list(DEFAULT_RUNTIME_FALLBACK_CONFIG["retry_on_errors"])
    return merged


def normalize_fallback_chain(fallback_model: Any, *, disabled_providers: set[str] | None = None) -> list[dict[str, Any]]:
    """Normalize legacy dict/list fallback config and prune disabled entries."""
    disabled = disabled_providers or set()
    if isinstance(fallback_model, list):
        entries = fallback_model
    elif isinstance(fallback_model, Mapping):
        entries = [fallback_model]
    else:
        entries = []

    chain: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        provider = normalize_provider_name(entry.get("provider"))
        model = normalize_model_name(entry.get("model"))
        if not provider or not model:
            continue
        if provider in disabled:
            continue
        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model
        chain.append(normalized)
    return chain


def should_try_runtime_fallback(classified_error: Any, runtime_config: Mapping[str, Any] | None) -> bool:
    """Return whether a classified API error is eligible for runtime fallback."""
    cfg = runtime_fallback_config_from_config({"runtime_fallback": runtime_config or {}})
    if not cfg.get("enabled", True):
        return False
    if not bool(getattr(classified_error, "should_fallback", False)):
        return False
    status = getattr(classified_error, "status_code", None)
    retry_on = cfg.get("retry_on_errors") or []
    if status is None:
        # No status: honor classifier's should_fallback signal.
        return True
    try:
        return int(status) in {int(v) for v in retry_on}
    except (TypeError, ValueError):
        return True


@dataclass(frozen=True)
class FallbackProvenance:
    provenance: str
    reason: str | None
    status_code: int | None
    from_provider: str
    from_model: str
    from_base_url: str
    to_provider: str
    to_model: str
    to_base_url: str
    chain_index: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_fallback_provenance(
    *,
    reason: Any,
    status_code: int | None,
    from_provider: Any,
    from_model: Any,
    from_base_url: Any,
    to_provider: Any,
    to_model: Any,
    to_base_url: Any,
    chain_index: int,
) -> dict[str, Any]:
    reason_value = getattr(reason, "value", None) or (str(reason) if reason is not None else None)
    return FallbackProvenance(
        provenance="provider-fallback",
        reason=reason_value,
        status_code=status_code,
        from_provider=normalize_provider_name(from_provider),
        from_model=normalize_model_name(from_model),
        from_base_url=str(from_base_url or ""),
        to_provider=normalize_provider_name(to_provider),
        to_model=normalize_model_name(to_model),
        to_base_url=str(to_base_url or ""),
        chain_index=chain_index,
    ).as_dict()
