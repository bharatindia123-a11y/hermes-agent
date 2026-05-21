"""Tests for provider fallback normalization/config/provenance helpers."""

from types import SimpleNamespace

from agent.provider_fallback import (
    build_fallback_provenance,
    disabled_providers_from_config,
    normalize_fallback_chain,
    runtime_fallback_config_from_config,
    should_try_runtime_fallback,
)


def test_disabled_providers_from_multiple_config_shapes():
    cfg = {
        "disabled_providers": ["OpenAI"],
        "providers": {
            "disabled": ["anthropic"],
            "zai": {"disabled": True},
            "nous": {"disabled": False},
        },
    }

    assert disabled_providers_from_config(cfg) == {"openai", "anthropic", "zai"}


def test_normalize_fallback_chain_prunes_invalid_and_disabled_entries():
    chain = normalize_fallback_chain(
        [
            {"provider": "OpenAI", "model": "gpt-5.5"},
            {"provider": "zai", "model": "glm-4.7"},
            {"provider": "", "model": "missing-provider"},
            {"provider": "anthropic"},
            "bad-entry",
        ],
        disabled_providers={"openai"},
    )

    assert chain == [{"provider": "zai", "model": "glm-4.7"}]


def test_runtime_fallback_config_defaults_and_overrides():
    cfg = runtime_fallback_config_from_config(
        {"runtime_fallback": {"enabled": False, "max_fallback_attempts": "2", "cooldown_seconds": "5"}}
    )

    assert cfg["enabled"] is False
    assert cfg["max_fallback_attempts"] == 2
    assert cfg["cooldown_seconds"] == 5
    assert 429 in cfg["retry_on_errors"]


def test_should_try_runtime_fallback_honors_enabled_status_and_classifier_signal():
    classified = SimpleNamespace(should_fallback=True, status_code=429)
    assert should_try_runtime_fallback(classified, {"enabled": True, "retry_on_errors": [429]}) is True
    assert should_try_runtime_fallback(classified, {"enabled": False, "retry_on_errors": [429]}) is False
    assert should_try_runtime_fallback(classified, {"enabled": True, "retry_on_errors": [500]}) is False
    assert should_try_runtime_fallback(SimpleNamespace(should_fallback=False, status_code=429), {"enabled": True}) is False


def test_build_fallback_provenance_records_status_and_source_target():
    reason = SimpleNamespace(value="rate_limit")
    event = build_fallback_provenance(
        reason=reason,
        status_code=429,
        from_provider="OpenRouter",
        from_model="z-ai/glm-4.7",
        from_base_url="https://openrouter.ai/api/v1",
        to_provider="OpenAI",
        to_model="gpt-5.5",
        to_base_url="https://api.openai.com/v1",
        chain_index=1,
    )

    assert event["provenance"] == "provider-fallback"
    assert event["reason"] == "rate_limit"
    assert event["status_code"] == 429
    assert event["from_provider"] == "openrouter"
    assert event["to_provider"] == "openai"
    assert event["chain_index"] == 1
