from __future__ import annotations

from types import SimpleNamespace

import pytest

from run_agent import AIAgent
from agent.task_contracts import build_named_workflow_artifact
from tools.delegate_tool import _build_child_agent, _resolve_task_inputs


def _contract(required_tools: list[str] | None = None) -> dict:
    return {
        "task": "review candidate runtime activation",
        "expected_outcome": "report",
        "required_skills": [],
        "required_tools": list(required_tools or []),
        "must_do": [],
        "must_not_do": [],
        "context": {"repo": "candidate"},
    }


def _agent_stub(valid_tool_names: set[str] | None = None):
    agent = object.__new__(AIAgent)
    agent.valid_tool_names = set(valid_tool_names or set())
    agent.ephemeral_system_prompt = ""
    agent._delegate_named_workflow = None
    return agent


def _parent_stub():
    return SimpleNamespace(
        model="test",
        base_url="",
        provider="",
        api_key="",
        api_mode="chat_completions",
        acp_command=None,
        acp_args=[],
        platform="cli",
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort=None,
        _session_db=None,
        _delegate_depth=0,
        tool_progress_callback=None,
        thinking_callback=None,
        prefill_messages=None,
        max_tokens=None,
        _print_fn=None,
        _active_children=[],
        _active_children_lock=None,
        enabled_toolsets=None,
        valid_tool_names={"read_file", "patch"},
        reasoning_config=None,
    )


def _install_fake_agent(monkeypatch):
    class FakeAIAgent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.tools = [
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "patch"}},
            ]
            self.valid_tool_names = {"read_file", "patch"}
            self.ephemeral_system_prompt = kwargs.get("ephemeral_system_prompt")
            self._delegate_depth = 0
            self._delegate_named_workflow = None

        def _delegate_named_workflow_prompt_block(self, named_workflow):
            return AIAgent._delegate_named_workflow_prompt_block(named_workflow)

        def _inject_delegate_named_workflow_prompt_block(self, named_workflow):
            return AIAgent._inject_delegate_named_workflow_prompt_block(self, named_workflow)

        def activate_delegate_runtime(self, delegate_resolution):
            AIAgent.activate_delegate_runtime(self, delegate_resolution)

    import run_agent

    monkeypatch.setattr(run_agent, "AIAgent", FakeAIAgent)
    return FakeAIAgent


def test_activate_delegate_runtime_stores_runtime_and_contract_state():
    agent = _agent_stub({"read_file", "search_files"})
    resolution = {
        "runtime_mode": "execution_supervisor",
        "task_contract": _contract(["read_file"]),
    }

    agent.activate_delegate_runtime(resolution)

    assert agent._delegate_resolution == resolution
    assert agent._delegate_runtime_mode == "execution_supervisor"
    assert agent._delegate_task_contract == resolution["task_contract"]
    assert agent._delegate_required_tools == {"read_file"}



def test_activate_delegate_runtime_derives_contract_and_prompt_from_named_workflow():
    agent = _agent_stub({"read_file", "search_files"})
    artifact = build_named_workflow_artifact(
        objective="Plan Bucket E",
        specialist="planner",
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
        task_contract=None,
    )
    resolution = {
        "runtime_mode": "default",
        "named_workflow": artifact,
    }

    agent.activate_delegate_runtime(resolution)

    assert agent._delegate_named_workflow == artifact
    assert agent._delegate_task_contract == artifact["execution_task_contract"]
    assert agent._delegate_resolution["task_contract"] == artifact["execution_task_contract"]
    assert "Named workflow activated: planner" in agent.ephemeral_system_prompt
    assert "<named-workflow>" in agent.ephemeral_system_prompt
    assert "execution_task_contract" in agent.ephemeral_system_prompt


def test_activate_delegate_runtime_rejects_unavailable_required_tools():
    agent = _agent_stub({"read_file"})

    with pytest.raises(ValueError, match="required_tools unavailable"):
        agent.activate_delegate_runtime(
            {
                "runtime_mode": "execution_supervisor",
                "task_contract": _contract(["patch"]),
            }
        )


def test_build_child_agent_activates_runtime_after_tool_restrictions(monkeypatch):
    _install_fake_agent(monkeypatch)
    resolution = {
        "named_agent": "atlas",
        "archetype": "implementer",
        "specialist": "builder",
        "runtime_mode": "execution_supervisor",
        "task_contract": _contract(["read_file"]),
    }

    child = _build_child_agent(
        task_index=0,
        goal="review",
        context=None,
        toolsets=None,
        model="test",
        max_iterations=10,
        task_count=1,
        parent_agent=_parent_stub(),
        enabled_tools=["read_file"],
        delegate_resolution=resolution,
        wave1_overlay_prompt="## OMO / Named-Agent Runtime Contract\n{}",
    )

    assert child._delegate_resolution == resolution
    assert child._delegate_runtime_mode == "execution_supervisor"
    assert child._delegate_task_contract == resolution["task_contract"]
    assert child._delegate_required_tools == {"read_file"}
    assert child.valid_tool_names == {"read_file"}
    assert "OMO / Named-Agent Runtime Contract" in child.ephemeral_system_prompt


def test_build_child_agent_rejects_contract_after_enabled_tool_filter(monkeypatch):
    _install_fake_agent(monkeypatch)

    with pytest.raises(ValueError, match="required_tools unavailable"):
        _build_child_agent(
            task_index=0,
            goal="edit",
            context=None,
            toolsets=None,
            model="test",
            max_iterations=10,
            task_count=1,
            parent_agent=_parent_stub(),
            enabled_tools=["read_file"],
            delegate_resolution={
                "named_agent": "atlas",
                "archetype": "implementer",
                "specialist": "builder",
                "runtime_mode": "execution_supervisor",
                "task_contract": _contract(["patch"]),
            },
            wave1_overlay_prompt="## OMO / Named-Agent Runtime Contract\n{}",
        )


def test_build_child_agent_prompt_includes_named_workflow_activation_block(monkeypatch):
    _install_fake_agent(monkeypatch)
    resolution = _resolve_task_inputs(
        {"goal": "Plan Bucket E", "specialist": "planner"},
        full_config={},
        delegation_config={},
    )

    child = _build_child_agent(
        task_index=0,
        goal="Plan Bucket E",
        context=None,
        toolsets=None,
        model="test",
        max_iterations=10,
        task_count=1,
        parent_agent=_parent_stub(),
        enabled_tools=["read_file", "search_files"],
        delegate_resolution=resolution,
        wave1_overlay_prompt=resolution["overlay_prompt"],
    )

    assert "WAVE 1 DELEGATION INPUTS" in child.ephemeral_system_prompt
    assert "OMO / Named-Agent Runtime Contract" in child.ephemeral_system_prompt
    assert "Named workflow activated: planner" in child.ephemeral_system_prompt
    assert "<named-workflow>" in child.ephemeral_system_prompt
    assert "workflow_name" in child.ephemeral_system_prompt
    assert "planner" in child.ephemeral_system_prompt
    assert "execution_task_contract" in child.ephemeral_system_prompt
    assert child._delegate_named_workflow["workflow_name"] == "planner"
