"""Tests for the background_agent runtime tool."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import patch

from tools.background_agent_tool import (
    BACKGROUND_AGENT_SCHEMA,
    _reset_background_agent_registry,
    background_agent_tool,
)


class Parent(SimpleNamespace):
    session_id: str = "parent-session"


def setup_function():
    _reset_background_agent_registry()


def _wait_for(agent_id: str, parent: Parent, status: str = "completed", timeout: float = 2.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = json.loads(background_agent_tool(action="status", agent_id=agent_id, parent_agent=parent))
        if last["job"]["status"] == status:
            return last
        time.sleep(0.01)
    return last


def test_schema_actions_are_stable():
    action = BACKGROUND_AGENT_SCHEMA["parameters"]["properties"]["action"]
    assert action["enum"] == ["create", "list", "status", "output", "cancel"]


def test_create_requires_parent_agent_and_goal():
    no_parent = json.loads(background_agent_tool(action="create", goal="do x"))
    assert no_parent["success"] is False
    assert "parent agent" in no_parent["error"]

    no_goal = json.loads(background_agent_tool(action="create", parent_agent=Parent(session_id="s1")))
    assert no_goal["success"] is False
    assert "goal" in no_goal["error"]


def test_create_status_output_and_list_completed_job():
    parent = Parent(session_id="s1")

    def fake_delegate(**kwargs):
        assert kwargs["goal"] == "do background work"
        assert kwargs["parent_agent"] is parent
        return json.dumps({"final_response": "done", "api_calls": 1, "child_session_id": "child-1"})

    with patch("tools.background_agent_tool._run_delegate_task_for_job", side_effect=lambda job, parent_agent: json.loads(fake_delegate(**job.request, parent_agent=parent_agent))):
        created = json.loads(background_agent_tool(action="create", goal="do background work", parent_agent=parent))
        assert created["success"] is True
        assert created["agent_id"].startswith("bg_")
        agent_id = created["agent_id"]
        status = _wait_for(agent_id, parent)

    assert status["job"]["status"] == "completed"
    output = json.loads(background_agent_tool(action="output", agent_id=agent_id, parent_agent=parent))
    assert output["success"] is True
    assert output["job"]["child_session_id"] == "child-1"
    assert output["job"]["api_calls"] == 1
    assert output["job"]["events"][-1]["event"] == "completed"

    listed = json.loads(background_agent_tool(action="list", parent_agent=parent))
    assert [j["agent_id"] for j in listed["jobs"]] == [agent_id]


def test_list_is_parent_session_scoped():
    p1 = Parent(session_id="s1")
    p2 = Parent(session_id="s2")
    with patch("tools.background_agent_tool._run_delegate_task_for_job", return_value={"final_response": "done"}):
        j1 = json.loads(background_agent_tool(action="create", goal="one", parent_agent=p1))["agent_id"]
        j2 = json.loads(background_agent_tool(action="create", goal="two", parent_agent=p2))["agent_id"]
        _wait_for(j1, p1)
        _wait_for(j2, p2)

    listed = json.loads(background_agent_tool(action="list", parent_agent=p1))
    assert [j["agent_id"] for j in listed["jobs"]] == [j1]


def test_cancel_marks_queued_or_running_job_for_cancellation():
    parent = Parent(session_id="s1")

    def slow_delegate(job, parent_agent):
        time.sleep(0.2)
        return {"final_response": "late"}

    with patch("tools.background_agent_tool._run_delegate_task_for_job", side_effect=slow_delegate):
        created = json.loads(background_agent_tool(action="create", goal="slow", parent_agent=parent))
        agent_id = created["agent_id"]
        cancelled = json.loads(background_agent_tool(action="cancel", agent_id=agent_id, parent_agent=parent))

    assert cancelled["success"] is True
    assert cancelled["job"]["status"] in {"cancelling", "cancelled"}


def test_concurrency_cap_is_enforced(monkeypatch):
    parent = Parent(session_id="s1")
    monkeypatch.setattr("tools.background_agent_tool._load_bg_config", lambda: {"max_concurrent": 1, "max_retained_jobs": 100})

    def slow_delegate(job, parent_agent):
        time.sleep(0.2)
        return {"final_response": "done"}

    with patch("tools.background_agent_tool._run_delegate_task_for_job", side_effect=slow_delegate):
        first = json.loads(background_agent_tool(action="create", goal="one", parent_agent=parent))
        second = json.loads(background_agent_tool(action="create", goal="two", parent_agent=parent))

    assert first["success"] is True
    assert second["success"] is False
    assert "concurrency cap" in second["error"]


def test_recursive_background_toolsets_are_stripped():
    parent = Parent(session_id="s1")
    captured = {}

    def fake_delegate(job, parent_agent):
        captured.update(job.request)
        return {"final_response": "done"}

    with patch("tools.background_agent_tool._run_delegate_task_for_job", side_effect=fake_delegate):
        created = json.loads(background_agent_tool(
            action="create",
            goal="toolset check",
            toolsets=["terminal", "background_agents", "team_mode", "file"],
            parent_agent=parent,
        ))
        _wait_for(created["agent_id"], parent)

    assert captured["toolsets"] == ["terminal", "file"]
