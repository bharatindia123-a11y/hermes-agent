from unittest.mock import patch

from agent.memory_provider import MemoryProvider
from run_agent import AIAgent


class CapturingCompositeProvider(MemoryProvider):
    def __init__(self):
        self.initialize_calls = []

    @property
    def name(self):
        return "composite"

    def is_available(self):
        return True

    def initialize(self, session_id: str, **kwargs):
        self.initialize_calls.append({"session_id": session_id, **kwargs})

    def get_tool_schemas(self):
        return []

    def prefetch(self, query: str, *, session_id: str = ""):
        raise AssertionError("composite prefetch should not be needed during init")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = ""):
        pass


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_bucket_a_run_agent_loads_composite_and_passes_composite_config(tmp_path, monkeypatch):
    composite_config = {
        "enabled": True,
        "mode": "shadow",
        "tools_enabled": False,
        "injection_enabled": False,
        "children": {
            "hindsight": {"enabled": True, "read": "shadow", "write": False},
            "honcho": {"enabled": True, "read": "shadow", "write": True},
        },
    }
    loaded_provider = CapturingCompositeProvider()

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value={
            "memory": {
                "provider": "composite",
                "composite": composite_config,
            },
            "compression": {"enabled": False},
            "model": {"context_length": 128000},
        }),
        patch("plugins.memory.load_memory_provider", return_value=loaded_provider) as load_mem,
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            model="test-model",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=False,
        )

    load_mem.assert_called_once_with("composite")
    assert agent._memory_manager is not None
    assert [p.name for p in agent._memory_manager.providers] == ["composite"]

    assert len(loaded_provider.initialize_calls) == 1
    init_call = loaded_provider.initialize_calls[0]
    assert init_call["session_id"] == agent.session_id
    assert init_call["composite_config"] == composite_config

    tool_names = {tool["function"]["name"] for tool in agent.tools if tool.get("type") == "function"}
    assert "composite_hindsight_hindsight_recall" not in tool_names
    assert "composite_hindsight_hindsight_retain" not in tool_names
    assert "composite_honcho_honcho_search" not in tool_names
    assert "composite_honcho_honcho_conclude" not in tool_names
