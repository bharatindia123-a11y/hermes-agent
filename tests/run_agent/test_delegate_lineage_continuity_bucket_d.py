from __future__ import annotations

import inspect
import json

from run_agent import AIAgent
from tools.delegate_tool import _delegate_child_lineage_fields


def _contract(required_tools: list[str] | None = None) -> dict:
    return {
        "task": "continue delegated child work",
        "expected_outcome": "child keeps delegated runtime identity",
        "required_skills": [],
        "required_tools": list(required_tools or []),
        "must_do": [],
        "must_not_do": [],
        "context": {"bucket": "D"},
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
    agent.ephemeral_system_prompt = ""
    agent._delegate_lineage_rehydrated = False
    agent.valid_tool_names = set(valid_tool_names or {"read_file"})
    return agent


def test_delegate_result_lineage_fields_are_public_and_json_safe():
    class Child:
        session_id = "child-session"
        _delegate_depth = 2
        _delegate_resolution = {
            "named_agent": "atlas",
            "runtime_mode": "execution_supervisor",
            "task_contract": _contract(["read_file"]),
            "non_json": object(),
        }

    fields = _delegate_child_lineage_fields(Child())

    assert fields["child_session_id"] == "child-session"
    assert fields["delegate_depth"] == 2
    assert fields["delegate_resolution"]["named_agent"] == "atlas"
    assert fields["delegate_resolution"]["runtime_mode"] == "execution_supervisor"
    # The field must be serializable because it is embedded in delegate_task's
    # JSON result and later replayed from the parent transcript.
    json.dumps(fields)


def test_rehydrate_delegate_runtime_from_parent_delegate_tool_result():
    resolution = {
        "named_agent": "atlas",
        "archetype": "implementer",
        "specialist": "builder",
        "runtime_mode": "execution_supervisor",
        "task_contract": _contract(["read_file"]),
    }
    payload = {
        "results": [
            {
                "task_index": 0,
                "status": "completed",
                "summary": "done",
                "child_session_id": "child-session",
                "delegate_depth": 2,
                "delegate_resolution": resolution,
            }
        ],
        "total_duration_seconds": 1.0,
    }
    parent_messages = [
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
            # raw_decode should tolerate bounded trailing text from stored tool
            # result annotations without requiring persisted-output replay.
            "content": json.dumps(payload) + "\n\n[trailing annotation]",
        },
    ]
    agent = _agent_stub(parent_messages, {"read_file", "search_files"})

    assert agent._rehydrate_delegate_runtime_from_parent_result() is True

    assert agent._parent_session_id == "parent-session"
    assert agent._delegate_depth == 2
    assert agent._delegate_resolution == resolution
    assert agent._delegate_runtime_mode == "execution_supervisor"
    assert agent._delegate_task_contract == resolution["task_contract"]
    assert agent._delegate_required_tools == {"read_file"}
    assert agent._session_db.loaded_sessions == ["child-session"]
    assert agent._session_db.loaded_messages == ["parent-session"]


def test_rehydrate_delegate_runtime_ignores_non_matching_child_results():
    payload = {
        "results": [
            {
                "task_index": 0,
                "child_session_id": "other-child",
                "delegate_depth": 3,
                "delegate_resolution": {"runtime_mode": "execution_supervisor"},
            }
        ]
    }
    agent = _agent_stub(
        [
            {"role": "assistant", "tool_calls": [{"id": "call-delegate", "function": {"name": "delegate_task"}}]},
            {"role": "tool", "tool_call_id": "call-delegate", "content": json.dumps(payload)},
        ]
    )

    assert agent._rehydrate_delegate_runtime_from_parent_result() is False
    assert agent._delegate_resolution == {}
    assert agent._delegate_depth == 0


def test_run_conversation_hooks_delegate_lineage_after_primary_restore():
    from agent import conversation_loop

    source = inspect.getsource(conversation_loop.run_conversation)

    restore_idx = source.index("agent._restore_primary_runtime()")
    rehydrate_idx = source.index("agent._rehydrate_delegate_runtime_from_parent_result()")

    assert restore_idx < rehydrate_idx
