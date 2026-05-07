from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.delegate_tool import delegate_task


def _parent():
    return SimpleNamespace(
        session_id="parent-session",
        model="test-model",
        provider="test-provider",
        base_url="",
        api_key="",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        enabled_toolsets=["file"],
        valid_tool_names={"read_file", "patch"},
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        reasoning_config=None,
        max_tokens=None,
        prefill_messages=None,
        platform="cli",
        _delegate_depth=0,
        _active_children=[],
        _active_children_lock=None,
        _session_db=None,
        tool_progress_callback=None,
        thinking_callback=None,
        _print_fn=None,
        _memory_manager=None,
    )


def test_foreground_delegate_preserves_explicit_empty_enabled_tools(monkeypatch):
    captured = {}

    def fake_build_child_agent(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr("tools.delegate_tool._load_config", lambda: {"profiles": {"denyall": {"enabled_tools": []}}})
    monkeypatch.setattr("tools.delegate_tool._load_full_config", lambda: {})
    monkeypatch.setattr(
        "tools.delegate_tool._resolve_delegation_credentials",
        lambda cfg, parent: {"model": "m", "provider": "p", "base_url": "", "api_key": "", "api_mode": "chat_completions"},
    )
    monkeypatch.setattr("tools.delegate_tool._build_child_agent", fake_build_child_agent)

    with pytest.raises(RuntimeError, match="stop-after-capture"):
        delegate_task(goal="inspect", delegation_profile="denyall", parent_agent=_parent())

    assert captured["enabled_tools"] == []


def test_literal_category_maps_to_route_category_without_explicit_route():
    from tools.delegate_tool import _resolve_task_inputs

    resolved = _resolve_task_inputs({"goal": "inspect", "category": "visual-engineering"}, full_config={}, delegation_config={})

    assert resolved["category"] == "visual-engineering"
    assert resolved["route_category"] == "visual"
