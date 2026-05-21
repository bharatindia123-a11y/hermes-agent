"""Disabled-by-default Team Mode MVP tools.

This is intentionally small: Team Mode is an orchestration UX over the
background_agent runtime, not a separate runner.  The feature gate lives in
config (`team_mode.enabled: false` by default) and the tools are not part of
core toolsets.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tools.registry import registry
from tools.background_agent_tool import background_agent_tool

TEAM_TOOL_NAMES = ["team_spawn", "team_list", "team_status", "team_output", "team_cancel"]


def _json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, default=str)


def team_mode_enabled() -> bool:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return False
    team_cfg = cfg.get("team_mode", {}) if isinstance(cfg, dict) else {}
    if not isinstance(team_cfg, dict):
        return False
    return bool(team_cfg.get("enabled", False))


def _require_parent(parent_agent: Any = None) -> str | None:
    if parent_agent is None:
        return _json({"success": False, "error": "Team Mode tools must be dispatched by the agent loop"})
    return None


def team_spawn_tool(
    *,
    goal: str,
    context: str | None = None,
    role: str | None = None,
    toolsets: List[str] | None = None,
    skills: List[str] | None = None,
    agent: str | None = None,
    specialist: str | None = None,
    runtime_mode: str | None = None,
    parent_agent: Any = None,
    task_id: str | None = None,
) -> str:
    """Spawn a team member as a background agent."""
    error = _require_parent(parent_agent)
    if error:
        return error
    team_context = context or ""
    if role:
        team_context = f"Team role: {role}\n\n{team_context}".strip()
    result = json.loads(background_agent_tool(
        action="create",
        goal=goal,
        context=team_context,
        toolsets=toolsets,
        skills=skills,
        agent=agent,
        specialist=specialist,
        runtime_mode=runtime_mode,
        parent_agent=parent_agent,
        task_id=task_id,
    ))
    if result.get("success"):
        result["team_member_id"] = result.get("agent_id")
        result["mode"] = "team_mode"
    return _json(result)


def team_list_tool(*, parent_agent: Any = None, task_id: str | None = None) -> str:
    error = _require_parent(parent_agent)
    if error:
        return error
    result = json.loads(background_agent_tool(action="list", parent_agent=parent_agent, task_id=task_id))
    if result.get("success"):
        result["team_members"] = result.pop("jobs", [])
    return _json(result)


def team_status_tool(*, team_member_id: str, parent_agent: Any = None, task_id: str | None = None) -> str:
    error = _require_parent(parent_agent)
    if error:
        return error
    return background_agent_tool(action="status", agent_id=team_member_id, parent_agent=parent_agent, task_id=task_id)


def team_output_tool(
    *,
    team_member_id: str,
    offset: int = 0,
    limit: int = 200,
    parent_agent: Any = None,
    task_id: str | None = None,
) -> str:
    error = _require_parent(parent_agent)
    if error:
        return error
    return background_agent_tool(
        action="output",
        agent_id=team_member_id,
        offset=offset,
        limit=limit,
        parent_agent=parent_agent,
        task_id=task_id,
    )


def team_cancel_tool(*, team_member_id: str, parent_agent: Any = None, task_id: str | None = None) -> str:
    error = _require_parent(parent_agent)
    if error:
        return error
    return background_agent_tool(action="cancel", agent_id=team_member_id, parent_agent=parent_agent, task_id=task_id)


def _handle_team_spawn(args: Dict[str, Any], **kw: Any) -> str:
    return team_spawn_tool(parent_agent=kw.get("parent_agent"), task_id=kw.get("task_id"), **args)


def _handle_team_list(args: Dict[str, Any], **kw: Any) -> str:
    return team_list_tool(parent_agent=kw.get("parent_agent"), task_id=kw.get("task_id"))


def _handle_team_status(args: Dict[str, Any], **kw: Any) -> str:
    return team_status_tool(parent_agent=kw.get("parent_agent"), task_id=kw.get("task_id"), **args)


def _handle_team_output(args: Dict[str, Any], **kw: Any) -> str:
    return team_output_tool(parent_agent=kw.get("parent_agent"), task_id=kw.get("task_id"), **args)


def _handle_team_cancel(args: Dict[str, Any], **kw: Any) -> str:
    return team_cancel_tool(parent_agent=kw.get("parent_agent"), task_id=kw.get("task_id"), **args)


_TEAM_MEMBER_ID_PROP = {"type": "string", "description": "Team member bg_ id returned by team_spawn."}

registry.register(
    name="team_spawn",
    toolset="team_mode",
    schema={
        "name": "team_spawn",
        "description": "Spawn a Team Mode member as a background agent. Disabled unless team_mode.enabled is true.",
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {"type": "string"},
                "context": {"type": "string"},
                "role": {"type": "string"},
                "toolsets": {"type": "array", "items": {"type": "string"}},
                "skills": {"type": "array", "items": {"type": "string"}},
                "agent": {"type": "string"},
                "specialist": {"type": "string"},
                "runtime_mode": {"type": "string"},
            },
            "required": ["goal"],
        },
    },
    handler=_handle_team_spawn,
    check_fn=team_mode_enabled,
    emoji="👥",
)

registry.register(
    name="team_list",
    toolset="team_mode",
    schema={"name": "team_list", "description": "List Team Mode background members.", "parameters": {"type": "object", "properties": {}}},
    handler=_handle_team_list,
    check_fn=team_mode_enabled,
    emoji="👥",
)

for _name, _handler, _desc in [
    ("team_status", _handle_team_status, "Get Team Mode member status."),
    ("team_output", _handle_team_output, "Read Team Mode member output/events."),
    ("team_cancel", _handle_team_cancel, "Cancel a Team Mode member."),
]:
    props = {"team_member_id": _TEAM_MEMBER_ID_PROP}
    if _name == "team_output":
        props.update({"offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1}})
    registry.register(
        name=_name,
        toolset="team_mode",
        schema={"name": _name, "description": _desc, "parameters": {"type": "object", "properties": props, "required": ["team_member_id"]}},
        handler=_handler,
        check_fn=team_mode_enabled,
        emoji="👥",
    )
