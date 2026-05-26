"""Background-agent runtime tool.

A small Hermes-native background job registry for agent work.  This is the
safe prerequisite for Team Mode: it provides bg_ IDs, status/output/cancel
semantics, bounded in-process retention, and an opt-in tool surface without
auto-enabling Team Mode.
"""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home
from tools.registry import registry


TERMINAL_STATUSES = {"completed", "failed", "cancelled", "lost"}
_DEFAULT_MAX_CONCURRENT = 3
_DEFAULT_MAX_RETAINED = 100
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_DEFAULT_MAX_CONCURRENT,
    thread_name_prefix="hermes-bg-agent",
)
_JOBS_LOCK = threading.RLock()
_JOBS: Dict[str, "BackgroundAgentJob"] = {}
_STORE_LOADED = False


@dataclass
class BackgroundAgentJob:
    agent_id: str
    goal: str
    context: str = ""
    parent_session_id: str = ""
    parent_task_id: str = ""
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_events: List[Dict[str, Any]] = field(default_factory=list)
    final_response: str = ""
    error: str = ""
    api_calls: int = 0
    child_session_id: str = ""
    cancel_requested: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    future: Any = field(default=None, repr=False, compare=False)
    request: Dict[str, Any] = field(default_factory=dict, repr=False)

    def public_dict(self, *, include_output: bool = False, offset: int = 0, limit: int = 200) -> Dict[str, Any]:
        # Manual serialization avoids dataclasses.asdict() deep-copying the
        # Future object, which contains unpickleable thread locks.
        data = {
            "agent_id": self.agent_id,
            "goal": self.goal,
            "context": self.context,
            "parent_session_id": self.parent_session_id,
            "parent_task_id": self.parent_task_id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "final_response": self.final_response,
            "error": self.error,
            "api_calls": self.api_calls,
            "child_session_id": self.child_session_id,
            "cancel_requested": self.cancel_requested,
        }
        if include_output:
            safe_offset = max(0, int(offset or 0))
            safe_limit = max(1, min(1000, int(limit or 200)))
            data["events"] = self.output_events[safe_offset:safe_offset + safe_limit]
            data["event_count"] = len(self.output_events)
        return data


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def _job_id() -> str:
    return f"bg_{uuid.uuid4().hex[:12]}"


def _load_bg_config() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    raw = cfg.get("background_agents", {}) if isinstance(cfg, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    try:
        max_concurrent = int(raw.get("max_concurrent", _DEFAULT_MAX_CONCURRENT))
    except (TypeError, ValueError):
        max_concurrent = _DEFAULT_MAX_CONCURRENT
    try:
        max_retained = int(raw.get("max_retained_jobs", _DEFAULT_MAX_RETAINED))
    except (TypeError, ValueError):
        max_retained = _DEFAULT_MAX_RETAINED
    return {
        "max_concurrent": max(1, min(16, max_concurrent)),
        "max_retained_jobs": max(1, min(1000, max_retained)),
    }


def _parent_session_id(parent_agent: Any = None) -> str:
    return str(getattr(parent_agent, "session_id", "") or "")


def _append_event(job: BackgroundAgentJob, event: str, **fields: Any) -> None:
    entry = {"ts": time.time(), "event": event, **fields}
    job.output_events.append(entry)
    _persist_event(job, entry)


def _store_dir() -> Path:
    return Path(get_hermes_home()) / "background_agents"


def _job_meta_path(agent_id: str) -> Path:
    return _store_dir() / f"{agent_id}.json"


def _job_events_path(agent_id: str) -> Path:
    return _store_dir() / f"{agent_id}.events.jsonl"


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except TypeError:
        return str(value)


def _persist_job_metadata(job: BackgroundAgentJob) -> None:
    try:
        directory = _store_dir()
        directory.mkdir(parents=True, exist_ok=True)
        request = {k: _json_safe(v) for k, v in (job.request or {}).items() if not str(k).startswith("_")}
        data = {
            "agent_id": job.agent_id,
            "goal": job.goal,
            "context": job.context,
            "parent_session_id": job.parent_session_id,
            "parent_task_id": job.parent_task_id,
            "status": job.status,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "final_response": job.final_response,
            "error": job.error,
            "api_calls": job.api_calls,
            "child_session_id": job.child_session_id,
            "cancel_requested": job.cancel_requested,
            "request": request,
        }
        tmp = _job_meta_path(job.agent_id).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(_job_meta_path(job.agent_id))
    except Exception:
        # Background-agent persistence must not break the live tool path.
        pass


def _persist_event(job: BackgroundAgentJob, event: Dict[str, Any]) -> None:
    try:
        directory = _store_dir()
        directory.mkdir(parents=True, exist_ok=True)
        with _job_events_path(job.agent_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def _load_events(agent_id: str) -> List[Dict[str, Any]]:
    path = _job_events_path(agent_id)
    events: List[Dict[str, Any]] = []
    try:
        if not path.exists():
            return events
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    except Exception:
        return events
    return events


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _load_job_from_meta(path: Path) -> Optional[BackgroundAgentJob]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        agent_id = str(data.get("agent_id") or path.stem)
        job = BackgroundAgentJob(
            agent_id=agent_id,
            goal=str(data.get("goal") or ""),
            context=str(data.get("context") or ""),
            parent_session_id=str(data.get("parent_session_id") or ""),
            parent_task_id=str(data.get("parent_task_id") or ""),
            status=str(data.get("status") or "queued"),
            created_at=float(data.get("created_at") or time.time()),
            started_at=_coerce_optional_float(data.get("started_at")),
            finished_at=_coerce_optional_float(data.get("finished_at")),
            final_response=str(data.get("final_response") or ""),
            error=str(data.get("error") or ""),
            api_calls=int(data.get("api_calls") or 0),
            child_session_id=str(data.get("child_session_id") or ""),
            cancel_requested=bool(data.get("cancel_requested") or False),
            request=data.get("request") if isinstance(data.get("request"), dict) else {},
        )
    except Exception:
        return None
    if job.cancel_requested:
        job.cancel_event.set()
    job.output_events = _load_events(agent_id)
    return job


def _recover_orphaned_job_as_lost(job: BackgroundAgentJob) -> None:
    if job.status in TERMINAL_STATUSES:
        return
    job.status = "lost"
    job.finished_at = time.time()
    if not job.error:
        job.error = "Background agent orphaned during process restart; marked lost and not resumed."
    _append_event(job, "lost", message=job.error)
    _persist_job_metadata(job)


def _ensure_store_loaded() -> None:
    global _STORE_LOADED
    with _JOBS_LOCK:
        if _STORE_LOADED:
            return
        try:
            directory = _store_dir()
            if directory.exists():
                for path in sorted(directory.glob("bg_*.json")):
                    job = _load_job_from_meta(path)
                    if job is None:
                        continue
                    _recover_orphaned_job_as_lost(job)
                    _JOBS.setdefault(job.agent_id, job)
        finally:
            _STORE_LOADED = True
        _prune_retained_jobs()


def _matching_jobs(parent_agent: Any = None) -> List[BackgroundAgentJob]:
    _ensure_store_loaded()
    sid = _parent_session_id(parent_agent)
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    if sid:
        jobs = [j for j in jobs if j.parent_session_id == sid]
    return sorted(jobs, key=lambda j: j.created_at)


def _prune_retained_jobs() -> None:
    cfg = _load_bg_config()
    max_retained = cfg["max_retained_jobs"]
    with _JOBS_LOCK:
        if len(_JOBS) <= max_retained:
            return
        terminal = sorted(
            [j for j in _JOBS.values() if j.status in TERMINAL_STATUSES],
            key=lambda j: j.finished_at or j.created_at,
        )
        while len(_JOBS) > max_retained and terminal:
            victim = terminal.pop(0)
            _JOBS.pop(victim.agent_id, None)
            try:
                _job_meta_path(victim.agent_id).unlink(missing_ok=True)
                _job_events_path(victim.agent_id).unlink(missing_ok=True)
            except Exception:
                pass


def _running_count(parent_agent: Any = None) -> int:
    return sum(1 for j in _matching_jobs(parent_agent) if j.status in {"queued", "running", "cancelling"})


def _sanitize_child_toolsets(toolsets: Any) -> Optional[List[str]]:
    if not isinstance(toolsets, list):
        return None
    blocked = {"background_agents", "background_agent", "team_mode", "team"}
    cleaned: List[str] = []
    for item in toolsets:
        name = str(item or "").strip()
        if not name or name in blocked:
            continue
        if name not in cleaned:
            cleaned.append(name)
    return cleaned or None


def _run_delegate_task_for_job(job: BackgroundAgentJob, parent_agent: Any) -> Dict[str, Any]:
    from tools.delegate_tool import delegate_task

    req = dict(job.request)
    req["_cancel_event"] = job.cancel_event
    return json.loads(delegate_task(parent_agent=parent_agent, **req))


def _worker(job_id: str, parent_agent: Any) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        if job.cancel_requested:
            job.status = "cancelled"
            job.finished_at = time.time()
            _append_event(job, "cancelled", message="Cancelled before start")
            _persist_job_metadata(job)
            _prune_retained_jobs()
            return
        job.status = "running"
        job.started_at = time.time()
        _append_event(job, "started")
        _persist_job_metadata(job)
    try:
        result = _run_delegate_task_for_job(job, parent_agent)
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            if job.cancel_requested:
                job.status = "cancelled"
                job.finished_at = time.time()
                _append_event(job, "cancelled", message="Cancelled during delegate execution")
                _persist_job_metadata(job)
                return
            job.api_calls = int(result.get("api_calls") or 0)
            job.child_session_id = str(result.get("child_session_id") or "")
            job.final_response = json.dumps(result, ensure_ascii=False, default=str)
            job.status = "completed"
            job.finished_at = time.time()
            _append_event(job, job.status, result=result)
            _persist_job_metadata(job)
    except Exception as exc:
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if not job:
                return
            job.error = str(exc)
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.finished_at = time.time()
            _append_event(job, job.status, error=str(exc))
            _persist_job_metadata(job)
    finally:
        _prune_retained_jobs()


def background_agent_tool(
    *,
    action: str,
    agent_id: str | None = None,
    goal: str | None = None,
    context: str | None = None,
    toolsets: list[str] | None = None,
    agent: str | None = None,
    subagent_type: str | None = None,
    category: str | None = None,
    archetype: str | None = None,
    specialist: str | None = None,
    route_category: str | None = None,
    delegation_profile: str | None = None,
    runtime_mode: str | None = None,
    skills: list[str] | None = None,
    task_contract: dict | None = None,
    named_workflow: dict | None = None,
    offset: int = 0,
    limit: int = 200,
    parent_agent: Any = None,
    task_id: str | None = None,
) -> str:
    """Create/list/status/output/cancel background agent jobs."""
    action = str(action or "").strip().lower()
    if action not in {"create", "list", "status", "output", "cancel"}:
        return _json({"success": False, "error": "Unknown background_agent action"})
    _ensure_store_loaded()

    if action == "list":
        return _json({
            "success": True,
            "jobs": [j.public_dict() for j in _matching_jobs(parent_agent)],
        })

    if action in {"status", "output", "cancel"}:
        if not agent_id:
            return _json({"success": False, "error": f"background_agent action='{action}' requires agent_id"})
        with _JOBS_LOCK:
            job = _JOBS.get(agent_id)
            if not job:
                return _json({"success": False, "error": f"Unknown background agent id: {agent_id}"})
            if action == "cancel":
                if job.status in TERMINAL_STATUSES:
                    return _json({"success": True, "job": job.public_dict(include_output=True), "message": "Job already finished"})
                job.cancel_requested = True
                job.cancel_event.set()
                job.status = "cancelling"
                _append_event(job, "cancel_requested")
                fut = job.future
                cancelled_before_start = False
                if fut is not None:
                    try:
                        cancelled_before_start = bool(fut.cancel())
                    except Exception:
                        cancelled_before_start = False
                if cancelled_before_start:
                    job.status = "cancelled"
                    job.finished_at = time.time()
                    _append_event(job, "cancelled", message="Cancelled before start")
                _persist_job_metadata(job)
                if cancelled_before_start:
                    _prune_retained_jobs()
                return _json({"success": True, "job": job.public_dict(), "message": "Cancellation requested"})
            return _json({
                "success": True,
                "job": job.public_dict(include_output=(action == "output"), offset=offset, limit=limit),
            })

    # create
    if parent_agent is None:
        return _json({"success": False, "error": "background_agent create requires parent agent context"})
    goal = str(goal or "").strip()
    if not goal:
        return _json({"success": False, "error": "background_agent create requires a non-empty goal"})

    cfg = _load_bg_config()
    if _running_count(parent_agent) >= cfg["max_concurrent"]:
        return _json({"success": False, "error": f"background_agent concurrency cap reached ({cfg['max_concurrent']})"})

    req = {
        "goal": goal,
        "context": context or "",
        "toolsets": _sanitize_child_toolsets(toolsets),
        "agent": agent,
        "subagent_type": subagent_type,
        "category": category,
        "archetype": archetype,
        "specialist": specialist,
        "route_category": route_category,
        "delegation_profile": delegation_profile,
        "runtime_mode": runtime_mode,
        "skills": skills,
        "task_contract": task_contract,
        "named_workflow": named_workflow,
    }
    req = {k: v for k, v in req.items() if v not in (None, [], {})}
    job = BackgroundAgentJob(
        agent_id=_job_id(),
        goal=goal,
        context=context or "",
        parent_session_id=_parent_session_id(parent_agent),
        parent_task_id=task_id or "",
        request=req,
    )
    _append_event(job, "created")
    job.request["_cancel_event"] = job.cancel_event
    _persist_job_metadata(job)
    with _JOBS_LOCK:
        _JOBS[job.agent_id] = job
        job.future = _EXECUTOR.submit(_worker, job.agent_id, parent_agent)
    return _json({"success": True, "agent_id": job.agent_id, "job": job.public_dict()})


def _handle_background_agent(args: Dict[str, Any], **kw: Any) -> str:
    parent_agent = kw.get("parent_agent")
    if parent_agent is None:
        return _json({"success": False, "error": "background_agent must be dispatched by the agent loop"})
    return background_agent_tool(parent_agent=parent_agent, task_id=kw.get("task_id"), **args)


def _reset_background_agent_registry() -> None:
    """Test helper: clear in-process jobs without deleting durable job artifacts."""
    global _STORE_LOADED
    with _JOBS_LOCK:
        _JOBS.clear()
        # Unit tests start from an empty in-memory registry by default. Tests
        # that exercise durable recovery explicitly set this back to False.
        _STORE_LOADED = True


BACKGROUND_AGENT_SCHEMA = {
    "name": "background_agent",
    "description": (
        "Create and manage background agent tasks. Use action='create' for independent "
        "subtasks that should continue while you do other work; use 'list', 'status', "
        "'output', and 'cancel' to manage bg_ jobs. Off default core toolsets; intended "
        "for explicit opt-in and Team Mode."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["create", "list", "status", "output", "cancel"]},
            "agent_id": {"type": "string", "description": "bg_ id returned by create; required for status/output/cancel."},
            "goal": {"type": "string", "description": "Goal for action='create'."},
            "context": {"type": "string", "description": "Relevant context for the background agent."},
            "toolsets": {"type": "array", "items": {"type": "string"}},
            "agent": {"type": "string"},
            "subagent_type": {"type": "string"},
            "category": {"type": "string"},
            "archetype": {"type": "string"},
            "specialist": {"type": "string"},
            "route_category": {"type": "string"},
            "delegation_profile": {"type": "string"},
            "runtime_mode": {"type": "string"},
            "skills": {"type": "array", "items": {"type": "string"}},
            "task_contract": {"type": "object"},
            "named_workflow": {"type": "object"},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1},
        },
        "required": ["action"],
    },
}


registry.register(
    name="background_agent",
    toolset="background_agents",
    schema=BACKGROUND_AGENT_SCHEMA,
    handler=_handle_background_agent,
    check_fn=lambda: True,
    emoji="🧵",
)
