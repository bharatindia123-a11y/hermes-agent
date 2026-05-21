"""Tests for disabled-by-default Team Mode MVP tools."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tools.background_agent_tool import _reset_background_agent_registry
from tools.team_mode_tool import TEAM_TOOL_NAMES, team_mode_enabled, team_spawn_tool, team_list_tool


class Parent(SimpleNamespace):
    session_id: str = "team-parent"


def setup_function():
    _reset_background_agent_registry()


def test_team_mode_enabled_defaults_false_when_config_missing(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {})
    assert team_mode_enabled() is False


def test_team_mode_enabled_respects_config(monkeypatch):
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"team_mode": {"enabled": True}})
    assert team_mode_enabled() is True


def test_team_spawn_requires_agent_loop_parent():
    result = json.loads(team_spawn_tool(goal="go"))
    assert result["success"] is False
    assert "agent loop" in result["error"]


def test_team_spawn_wraps_background_agent_and_returns_member_id():
    parent = Parent(session_id="team-s1")
    with patch("tools.team_mode_tool.background_agent_tool") as mock_bg:
        mock_bg.return_value = json.dumps({"success": True, "agent_id": "bg_123", "job": {"status": "queued"}})
        result = json.loads(team_spawn_tool(
            goal="build feature",
            context="ctx",
            role="frontend",
            toolsets=["file"],
            skills=["skill-a"],
            parent_agent=parent,
            task_id="task-1",
        ))

    assert result["success"] is True
    assert result["team_member_id"] == "bg_123"
    kwargs = mock_bg.call_args.kwargs
    assert kwargs["action"] == "create"
    assert kwargs["goal"] == "build feature"
    assert kwargs["context"].startswith("Team role: frontend")
    assert kwargs["parent_agent"] is parent
    assert kwargs["task_id"] == "task-1"


def test_team_list_maps_background_jobs_to_team_members():
    parent = Parent(session_id="team-s1")
    with patch("tools.team_mode_tool.background_agent_tool", return_value=json.dumps({"success": True, "jobs": [{"agent_id": "bg_1"}]})):
        result = json.loads(team_list_tool(parent_agent=parent))
    assert result == {"success": True, "team_members": [{"agent_id": "bg_1"}]}


def test_team_tool_names_are_stable():
    assert TEAM_TOOL_NAMES == ["team_spawn", "team_list", "team_status", "team_output", "team_cancel"]
