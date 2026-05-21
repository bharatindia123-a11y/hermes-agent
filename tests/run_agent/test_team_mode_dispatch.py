"""AIAgent dispatch integration for Team Mode MVP."""

from unittest.mock import patch

from run_agent import AIAgent


def test_dispatch_team_spawn_forwards_parent_and_fields():
    agent = object.__new__(AIAgent)
    args = {
        "goal": "coordinate work",
        "context": "ctx",
        "role": "reviewer",
        "toolsets": ["file"],
        "skills": ["review"],
        "agent": "critic",
        "specialist": "reviewer",
        "runtime_mode": "default",
    }
    with patch("tools.team_mode_tool.team_spawn_tool", return_value="OK") as mock_spawn:
        result = agent._dispatch_team_mode("team_spawn", args, effective_task_id="task-1")

    assert result == "OK"
    kwargs = mock_spawn.call_args.kwargs
    assert kwargs["parent_agent"] is agent
    assert kwargs["task_id"] == "task-1"
    assert kwargs["goal"] == "coordinate work"
    assert kwargs["role"] == "reviewer"


def test_dispatch_team_status_forwards_member_id():
    agent = object.__new__(AIAgent)
    with patch("tools.team_mode_tool.team_status_tool", return_value="OK") as mock_status:
        result = agent._dispatch_team_mode("team_status", {"team_member_id": "bg_1"}, effective_task_id="task-2")
    assert result == "OK"
    assert mock_status.call_args.kwargs["team_member_id"] == "bg_1"
    assert mock_status.call_args.kwargs["parent_agent"] is agent
