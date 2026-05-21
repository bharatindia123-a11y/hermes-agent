from types import SimpleNamespace

from hermes_cli.default_mode import (
    ULTRAWORK_PROMPT_BLOCK,
    ULTRAWORK_PROMPT_MARKER,
    activate_ultrawork_on_agent,
    append_ultrawork_prompt,
    default_mode_config,
)


def test_default_mode_config_defaults_disabled():
    assert default_mode_config({}) == {"ultrawork": False, "ralph_loop": False}
    assert default_mode_config({"default_mode": None}) == {"ultrawork": False, "ralph_loop": False}


def test_default_mode_config_coerces_bool_like_values():
    cfg = {"default_mode": {"ultrawork": "true", "ralph_loop": 1}}
    assert default_mode_config(cfg) == {"ultrawork": True, "ralph_loop": True}


def test_append_ultrawork_prompt_adds_marker_once():
    prompt = append_ultrawork_prompt("Existing prompt")
    assert prompt.startswith("Existing prompt")
    assert ULTRAWORK_PROMPT_MARKER in prompt
    assert prompt.count(ULTRAWORK_PROMPT_MARKER) == 1
    assert append_ultrawork_prompt(prompt) == prompt


def test_activate_ultrawork_on_agent_updates_ephemeral_prompt_once():
    agent = SimpleNamespace(ephemeral_system_prompt="Base")
    assert activate_ultrawork_on_agent(agent) is True
    assert agent._default_mode_ultrawork is True
    assert agent.ephemeral_system_prompt.startswith("Base")
    assert ULTRAWORK_PROMPT_BLOCK in agent.ephemeral_system_prompt
    assert activate_ultrawork_on_agent(agent) is False
    assert agent.ephemeral_system_prompt.count(ULTRAWORK_PROMPT_MARKER) == 1
