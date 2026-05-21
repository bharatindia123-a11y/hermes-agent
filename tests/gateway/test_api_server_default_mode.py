import pytest

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_cli.goals import clear_goal, load_goal


class _FakeAgent:
    def __init__(self, ephemeral_system_prompt=None, session_id=None, **_kwargs):
        self.ephemeral_system_prompt = ephemeral_system_prompt
        self.session_id = session_id
        self.calls = []
        self.conversation_history = []
        self.session_prompt_tokens = 0
        self.session_completion_tokens = 0
        self.session_total_tokens = 0

    def run_conversation(self, user_message, conversation_history=None, task_id=None):
        self.calls.append(user_message)
        self.conversation_history = list(conversation_history or []) + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"reply {len(self.calls)}"},
        ]
        self.session_prompt_tokens += 1
        self.session_completion_tokens += 1
        self.session_total_tokens += 2
        return {
            "final_response": f"reply {len(self.calls)}",
            "messages": list(self.conversation_history),
        }


@pytest.mark.asyncio
async def test_api_server_default_mode_runs_inline_goal_loop(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"default_mode": {"ultrawork": True, "ralph_loop": True}, "goals": {"max_turns": 5}},
    )
    verdicts = iter([("continue", "not yet", False), ("done", "finished", False)])
    monkeypatch.setattr("hermes_cli.goals.judge_goal", lambda *_args, **_kwargs: next(verdicts))

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agents = []

    def _create_agent(**kwargs):
        agent = _FakeAgent(**kwargs)
        agents.append(agent)
        return agent

    monkeypatch.setattr(adapter, "_create_agent", _create_agent)
    clear_goal("api-default-mode-session")

    result, usage = await adapter._run_agent(
        user_message="finish this task",
        conversation_history=[],
        session_id="api-default-mode-session",
    )

    assert result["final_response"] == "reply 2"
    assert usage == {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}
    assert len(agents) == 1
    assert agents[0].calls[0] == "finish this task"
    assert "Continue working toward this goal" in agents[0].calls[1]
    assert "<ultrawork-mode>" in agents[0].ephemeral_system_prompt
    assert load_goal("api-default-mode-session").status == "done"


@pytest.mark.asyncio
async def test_api_server_default_mode_disabled_stays_single_turn(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"default_mode": {}})

    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    agents = []
    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: agents.append(_FakeAgent(**kwargs)) or agents[-1])

    result, usage = await adapter._run_agent(
        user_message="one turn",
        conversation_history=[],
        session_id="api-default-mode-off-session",
    )

    assert result["final_response"] == "reply 1"
    assert usage == {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    assert agents[0].calls == ["one turn"]
    assert agents[0].ephemeral_system_prompt is None
    assert load_goal("api-default-mode-off-session") is None
