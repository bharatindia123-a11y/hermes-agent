"""AIAgent dispatch integration for background_agent."""

from unittest.mock import patch

from run_agent import AIAgent


def test_dispatch_background_agent_forwards_parent_and_fields():
    agent = object.__new__(AIAgent)
    args = {
        "action": "create",
        "goal": "do work",
        "context": "ctx",
        "toolsets": ["file"],
        "agent": "builder",
        "specialist": "planner",
        "runtime_mode": "default",
        "skills": ["x"],
        "offset": 2,
        "limit": 3,
    }
    with patch("tools.background_agent_tool.background_agent_tool", return_value="OK") as mock_bg:
        result = agent._dispatch_background_agent(args, effective_task_id="task-1")

    assert result == "OK"
    assert mock_bg.call_args.kwargs["parent_agent"] is agent
    assert mock_bg.call_args.kwargs["task_id"] == "task-1"
    assert mock_bg.call_args.kwargs["goal"] == "do work"
    assert mock_bg.call_args.kwargs["toolsets"] == ["file"]
