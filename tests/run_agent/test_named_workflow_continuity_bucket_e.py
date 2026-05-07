from __future__ import annotations

import json

from agent.task_contracts import build_named_workflow_artifact
from run_agent import AIAgent


def _contract(required_tools: list[str] | None = None) -> dict:
    return {
        "task": "Plan Bucket E continuity",
        "expected_outcome": "Execution-ready handoff produced from planner workflow activation.",
        "required_skills": ["general_reasoning", "task_execution"],
        "required_tools": list(required_tools or ["read_file", "search_files"]),
        "must_do": ["decompose the work into ordered steps"],
        "must_not_do": ["do not collapse the workflow into prose-only instructions"],
        "context": {"bucket": "E"},
    }


class FakeSessionDB:
    def __init__(self, parent_messages):
        self.parent_messages = list(parent_messages)
        self.loaded_sessions = []
        self.loaded_messages = []

    def get_session(self, session_id):
        self.loaded_sessions.append(session_id)
        if session_id == "child-session":
            return {"id": "child-session", "parent_session_id": "parent-session"}
        return None

    def get_messages(self, session_id):
        self.loaded_messages.append(session_id)
        if session_id == "parent-session":
            return list(self.parent_messages)
        return []


def _agent_stub(parent_messages, valid_tool_names: set[str] | None = None):
    agent = object.__new__(AIAgent)
    agent.session_id = "child-session"
    agent._session_db = FakeSessionDB(parent_messages)
    agent._parent_session_id = None
    agent._delegate_depth = 0
    agent._delegate_resolution = {}
    agent._delegate_runtime_mode = None
    agent._delegate_named_workflow = None
    agent._delegate_task_contract = None
    agent._delegate_required_tools = set()
    agent._delegate_lineage_rehydrated = False
    agent.ephemeral_system_prompt = ""
    agent.valid_tool_names = set(valid_tool_names or {"read_file", "search_files"})
    return agent


def _delegate_parent_messages(payload):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-delegate",
                    "type": "function",
                    "function": {"name": "delegate_task", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-delegate",
            "tool_name": "delegate_task",
            "content": json.dumps(payload),
        },
    ]


def test_rehydrate_delegate_runtime_recovers_task_contract_and_named_workflow_from_delegate_result():
    contract = _contract()
    artifact = build_named_workflow_artifact(
        objective=contract["task"],
        specialist="planner",
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
        task_contract=contract,
    )
    resolution = {
        "specialist": "planner",
        "archetype": "generalist",
        "route_category": "deep",
        "runtime_mode": "default",
        "delegation_profile": "general",
        "task_contract": contract,
        "named_workflow": artifact,
    }
    payload = {
        "results": [
            {
                "task_index": 0,
                "status": "completed",
                "summary": "planned",
                "child_session_id": "child-session",
                "delegate_depth": 2,
                "delegate_resolution": resolution,
            }
        ]
    }
    agent = _agent_stub(_delegate_parent_messages(payload), {"read_file", "search_files"})

    assert agent._rehydrate_delegate_runtime_from_parent_result() is True

    assert agent._delegate_depth == 2
    assert agent._delegate_task_contract == contract
    assert agent._delegate_resolution["named_workflow"]["workflow_name"] == "planner"
    assert agent._delegate_named_workflow["workflow_name"] == "planner"
    assert agent._delegate_named_workflow["execution_task_contract"] == contract
    assert agent._delegate_required_tools == {"read_file", "search_files"}
    assert "Named workflow activated: planner" in agent.ephemeral_system_prompt
    assert "<named-workflow>" in agent.ephemeral_system_prompt


def test_rehydrate_named_workflow_ignores_non_delegate_or_top_level_prose():
    artifact = build_named_workflow_artifact(
        objective="Plan without delegation",
        specialist="planner",
        archetype="generalist",
        route_category="deep",
        runtime_mode="default",
        delegation_profile="general",
        task_contract=None,
    )
    parent_messages = [
        {"role": "user", "content": f"Please use planner workflow: {json.dumps(artifact)}"},
        {"role": "assistant", "content": "I will plan this without delegation."},
        {
            "role": "tool",
            "tool_call_id": "call-other",
            "tool_name": "read_file",
            "content": json.dumps({"named_workflow": artifact}),
        },
    ]
    agent = _agent_stub(parent_messages)

    assert agent._rehydrate_delegate_runtime_from_parent_result() is False
    assert agent._delegate_resolution == {}
    assert agent._delegate_task_contract is None
    assert agent._delegate_named_workflow is None
    assert agent.ephemeral_system_prompt == ""
