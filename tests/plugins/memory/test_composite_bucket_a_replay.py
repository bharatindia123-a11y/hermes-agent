import json

from agent.memory_provider import MemoryProvider
from plugins.memory import load_memory_provider
from plugins.memory.composite import CompositeMemoryProvider


class FakeChildProvider(MemoryProvider):
    def __init__(self, name):
        self._name = name
        self.initialize_calls = []
        self.sync_calls = []
        self.prefetch_calls = []
        self.tool_schema_calls = 0

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id: str, **kwargs):
        self.initialize_calls.append({"session_id": session_id, **kwargs})

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self.prefetch_calls.append({"query": query, "session_id": session_id})
        return f"{self._name} SHOULD NOT INJECT"

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        self.sync_calls.append(
            {
                "user_content": user_content,
                "assistant_content": assistant_content,
                "session_id": session_id,
            }
        )

    def get_tool_schemas(self):
        self.tool_schema_calls += 1
        return [
            {
                "name": f"{self._name}_tool",
                "description": f"{self._name} tool",
                "parameters": {"type": "object", "properties": {}},
            }
        ]

    def handle_tool_call(self, tool_name, args, **kwargs):
        return json.dumps({"tool_name": tool_name, "args": args})


class FakeChildFactory:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.provider


def test_bucket_a_provider_composite_loads_from_plugin_loader():
    provider = load_memory_provider("composite")

    assert provider is not None
    assert isinstance(provider, CompositeMemoryProvider)
    assert provider.name == "composite"
    assert provider.is_available() is True


def test_bucket_a_composite_config_write_gates_and_safety_defaults():
    hindsight = FakeChildProvider("hindsight")
    honcho = FakeChildProvider("honcho")

    provider = CompositeMemoryProvider()
    provider.initialize(
        "bucket-a-session",
        platform="cli",
        composite_config={
            "enabled": True,
            "mode": "shadow",
            "tools_enabled": False,
            "injection_enabled": False,
            "children": {
                "hindsight": {
                    "enabled": True,
                    "read": "shadow",
                    "write": False,
                    "provider_class": FakeChildFactory(hindsight),
                },
                "honcho": {
                    "enabled": True,
                    "read": "shadow",
                    "write": True,
                    "provider_class": FakeChildFactory(honcho),
                },
            },
        },
    )

    assert [child["name"] for child in provider._children] == ["hindsight", "honcho"]
    assert provider.get_tool_schemas() == []
    assert provider._tools_enabled is False
    assert provider.prefetch("who am I?", session_id="bucket-a-session") == ""
    assert provider._injection_enabled is False
    assert hindsight.prefetch_calls == []
    assert honcho.prefetch_calls == []

    provider.sync_turn("user msg", "assistant msg", session_id="bucket-a-session")
    assert hindsight.sync_calls == []
    assert honcho.sync_calls == [
        {
            "user_content": "user msg",
            "assistant_content": "assistant msg",
            "session_id": "bucket-a-session",
        }
    ]
