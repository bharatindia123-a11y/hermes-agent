"""Tests for the background_agent runtime tool."""

from __future__ import annotations

import json
import time
import threading
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


class QueuedCancelFuture:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True
        return True


class QueuedCancelExecutor:
    def __init__(self):
        self.future = QueuedCancelFuture()

    def submit(self, *args, **kwargs):
        return self.future


def test_cancel_queued_job_terminalizes_when_future_cancel_succeeds(monkeypatch):
    parent = Parent(session_id="s1")
    fake_executor = QueuedCancelExecutor()
    monkeypatch.setattr("tools.background_agent_tool._EXECUTOR", fake_executor)

    created = json.loads(background_agent_tool(action="create", goal="queued", parent_agent=parent))
    cancelled = json.loads(background_agent_tool(action="cancel", agent_id=created["agent_id"], parent_agent=parent))

    assert cancelled["success"] is True
    assert cancelled["job"]["status"] == "cancelled"
    assert cancelled["job"]["finished_at"] is not None
    assert fake_executor.future.cancelled is True

    output = json.loads(background_agent_tool(action="output", agent_id=created["agent_id"], parent_agent=parent))
    assert [event["event"] for event in output["job"]["events"]][-2:] == ["cancel_requested", "cancelled"]


def test_queued_cancel_applies_durable_retention(monkeypatch, tmp_path):
    from tools import background_agent_tool as bg

    parent = Parent(session_id="s1")
    fake_executor = QueuedCancelExecutor()
    monkeypatch.setattr("tools.background_agent_tool._EXECUTOR", fake_executor)
    monkeypatch.setattr("tools.background_agent_tool._store_dir", lambda: tmp_path / "background_agents")
    monkeypatch.setattr("tools.background_agent_tool._load_bg_config", lambda: {"max_concurrent": 1, "max_retained_jobs": 1})
    store = tmp_path / "background_agents"
    store.mkdir(parents=True)
    now = time.time()
    (store / "bg_old.json").write_text(json.dumps({
        "agent_id": "bg_old",
        "goal": "old",
        "context": "",
        "parent_session_id": "s1",
        "parent_task_id": "",
        "status": "completed",
        "created_at": now - 10,
        "started_at": now - 9,
        "finished_at": now - 8,
        "final_response": "old",
        "error": "",
        "api_calls": 0,
        "child_session_id": "",
        "cancel_requested": False,
        "request": {"goal": "old"},
    }))
    (store / "bg_old.events.jsonl").write_text(json.dumps({"ts": now - 8, "event": "completed"}) + "\n")
    with bg._JOBS_LOCK:
        bg._JOBS.clear()
        bg._STORE_LOADED = False

    created = json.loads(background_agent_tool(action="create", goal="queued", parent_agent=parent))
    cancelled = json.loads(background_agent_tool(action="cancel", agent_id=created["agent_id"], parent_agent=parent))

    assert cancelled["job"]["status"] == "cancelled"
    assert not (store / "bg_old.json").exists()
    assert not (store / "bg_old.events.jsonl").exists()


def test_running_cancel_sets_private_cancel_event_and_suppresses_late_result():
    parent = Parent(session_id="s1")
    captured = {}
    started = threading.Event()
    release = threading.Event()

    def fake_delegate(job, parent_agent):
        captured["event"] = job.cancel_event
        assert "_cancel_event" in job.request
        assert job.request["_cancel_event"] is job.cancel_event
        started.set()
        release.wait(timeout=2)
        return {"final_response": "late", "api_calls": 1, "child_session_id": "child-late"}

    with patch("tools.background_agent_tool._run_delegate_task_for_job", side_effect=fake_delegate):
        created = json.loads(background_agent_tool(action="create", goal="slow", parent_agent=parent))
        assert started.wait(timeout=2)
        cancelled = json.loads(background_agent_tool(action="cancel", agent_id=created["agent_id"], parent_agent=parent))
        release.set()
        final = _wait_for(created["agent_id"], parent, status="cancelled")

    assert cancelled["success"] is True
    assert captured["event"].is_set() is True
    assert "cancel_event" not in final["job"]
    output = json.loads(background_agent_tool(action="output", agent_id=created["agent_id"], parent_agent=parent))
    assert output["job"]["status"] == "cancelled"
    assert output["job"]["final_response"] == ""
    assert "result" not in output["job"]["events"][-1]


def test_delegate_task_receives_private_cancel_event_not_public_output():
    from tools.background_agent_tool import BackgroundAgentJob, _run_delegate_task_for_job

    parent = Parent(session_id="s1")
    job = BackgroundAgentJob(agent_id="bg_private", goal="private", request={"goal": "private"})
    captured = {}

    def fake_delegate_task(**kwargs):
        captured.update(kwargs)
        return json.dumps({"final_response": "ok"})

    with patch("tools.delegate_tool.delegate_task", side_effect=fake_delegate_task):
        result = _run_delegate_task_for_job(job, parent)

    assert result["final_response"] == "ok"
    assert captured["parent_agent"] is parent
    assert captured["_cancel_event"] is job.cancel_event
    assert "_cancel_event" not in job.public_dict(include_output=True)


def test_private_cancel_event_interrupts_delegate_child_execution(monkeypatch):
    from tools import delegate_tool

    cancel_event = threading.Event()
    started = threading.Event()
    interrupted = {}

    class Child:
        session_id = "child-cancel"
        _delegate_depth = 1
        _delegate_resolution = {}
        _delegate_role = "leaf"
        _delegate_saved_tool_names = []
        tool_progress_callback = None
        _credential_pool = None

        def run_conversation(self, user_message, task_id=None):
            started.set()
            time.sleep(1)
            return {"final_response": "late", "completed": True, "api_calls": 1, "messages": []}

        def interrupt(self, message=None):
            interrupted["message"] = message

        def close(self):
            interrupted["closed"] = True

    monkeypatch.setattr("tools.delegate_tool._get_child_timeout", lambda: 5)
    child = Child()
    thread = threading.Thread(
        target=lambda: interrupted.setdefault(
            "result",
            delegate_tool._run_single_child(0, "cancel me", child, Parent(session_id="s1"), cancel_event=cancel_event),
        )
    )
    thread.start()
    assert started.wait(timeout=2)
    cancel_event.set()
    deadline = time.time() + 1
    while "message" not in interrupted and time.time() < deadline:
        time.sleep(0.01)
    assert thread.is_alive() is True
    assert interrupted["message"] == "Background agent cancellation requested"
    assert interrupted.get("result") is None

    thread.join(timeout=2)

    assert thread.is_alive() is False
    assert interrupted["result"]["status"] == "interrupted"
    assert interrupted["result"]["exit_reason"] == "interrupted"


def test_malformed_persisted_metadata_is_skipped(monkeypatch, tmp_path):
    from tools import background_agent_tool as bg

    monkeypatch.setattr("tools.background_agent_tool._store_dir", lambda: tmp_path / "background_agents")
    store = tmp_path / "background_agents"
    store.mkdir(parents=True)
    (store / "bg_bad.json").write_text(json.dumps({
        "agent_id": "bg_bad",
        "goal": "bad",
        "status": "completed",
        "created_at": "not-a-float",
        "started_at": "not-a-float",
        "finished_at": "also-not-a-float",
        "api_calls": "not-an-int",
    }))
    with bg._JOBS_LOCK:
        bg._JOBS.clear()
        bg._STORE_LOADED = False

    listed = json.loads(background_agent_tool(action="list", parent_agent=Parent(session_id="s1")))
    assert listed["success"] is True
    assert listed["jobs"] == []


def test_durable_retention_prunes_metadata_and_events(monkeypatch, tmp_path):
    from tools import background_agent_tool as bg

    monkeypatch.setattr("tools.background_agent_tool._store_dir", lambda: tmp_path / "background_agents")
    monkeypatch.setattr("tools.background_agent_tool._load_bg_config", lambda: {"max_concurrent": 1, "max_retained_jobs": 1})
    store = tmp_path / "background_agents"
    store.mkdir(parents=True)
    now = time.time()
    for idx in range(2):
        agent_id = f"bg_done_{idx}"
        (store / f"{agent_id}.json").write_text(json.dumps({
            "agent_id": agent_id,
            "goal": str(idx),
            "context": "",
            "parent_session_id": "s1",
            "parent_task_id": "",
            "status": "completed",
            "created_at": now + idx,
            "started_at": now + idx,
            "finished_at": now + idx,
            "final_response": "done",
            "error": "",
            "api_calls": 0,
            "child_session_id": "",
            "cancel_requested": False,
            "request": {"goal": str(idx)},
        }))
        (store / f"{agent_id}.events.jsonl").write_text(json.dumps({"ts": now + idx, "event": "completed"}) + "\n")
    with bg._JOBS_LOCK:
        bg._JOBS.clear()
        bg._STORE_LOADED = False

    listed = json.loads(background_agent_tool(action="list", parent_agent=Parent(session_id="s1")))
    assert [job["agent_id"] for job in listed["jobs"]] == ["bg_done_1"]
    assert not (store / "bg_done_0.json").exists()
    assert not (store / "bg_done_0.events.jsonl").exists()


def test_completed_job_metadata_and_events_survive_memory_reload(monkeypatch, tmp_path):
    parent = Parent(session_id="s1")
    monkeypatch.setattr("tools.background_agent_tool._store_dir", lambda: tmp_path / "background_agents")

    with patch("tools.background_agent_tool._run_delegate_task_for_job", return_value={"final_response": "done", "api_calls": 2, "child_session_id": "child-1"}):
        created = json.loads(background_agent_tool(action="create", goal="durable", parent_agent=parent))
        agent_id = created["agent_id"]
        _wait_for(agent_id, parent)

    from tools import background_agent_tool as bg
    with bg._JOBS_LOCK:
        bg._JOBS.clear()
        bg._STORE_LOADED = False

    reloaded = json.loads(background_agent_tool(action="output", agent_id=agent_id, parent_agent=parent))
    assert reloaded["success"] is True
    assert reloaded["job"]["status"] == "completed"
    assert reloaded["job"]["api_calls"] == 2
    assert reloaded["job"]["child_session_id"] == "child-1"
    assert reloaded["job"]["events"][-1]["event"] == "completed"


def test_non_terminal_persisted_jobs_recover_as_lost_not_resumed(monkeypatch, tmp_path):
    from tools import background_agent_tool as bg

    monkeypatch.setattr("tools.background_agent_tool._store_dir", lambda: tmp_path / "background_agents")
    store = tmp_path / "background_agents"
    store.mkdir(parents=True)
    for status in ["queued", "running", "cancelling"]:
        agent_id = f"bg_{status}"
        (store / f"{agent_id}.json").write_text(json.dumps({
            "agent_id": agent_id,
            "goal": status,
            "context": "",
            "parent_session_id": "s1",
            "parent_task_id": "",
            "status": status,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "final_response": "",
            "error": "",
            "api_calls": 0,
            "child_session_id": "",
            "cancel_requested": status == "cancelling",
            "request": {"goal": status},
        }))
        (store / f"{agent_id}.events.jsonl").write_text(json.dumps({"ts": time.time(), "event": "created"}) + "\n")

    with bg._JOBS_LOCK:
        bg._JOBS.clear()
        bg._STORE_LOADED = False

    listed = json.loads(background_agent_tool(action="list", parent_agent=Parent(session_id="s1")))
    assert {job["status"] for job in listed["jobs"]} == {"lost"}
    assert len(listed["jobs"]) == 3
    for job in listed["jobs"]:
        assert job["finished_at"] is not None
        output = json.loads(background_agent_tool(action="output", agent_id=job["agent_id"], parent_agent=Parent(session_id="s1")))
        assert output["job"]["events"][-1]["event"] == "lost"

    monkeypatch.setattr("tools.background_agent_tool._load_bg_config", lambda: {"max_concurrent": 1, "max_retained_jobs": 100})
    with patch("tools.background_agent_tool._run_delegate_task_for_job", return_value={"final_response": "new"}):
        created = json.loads(background_agent_tool(action="create", goal="new job", parent_agent=Parent(session_id="s1")))
    assert created["success"] is True
