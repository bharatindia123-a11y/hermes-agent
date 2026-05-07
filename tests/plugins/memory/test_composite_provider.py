import json

from agent.memory_provider import MemoryProvider
from plugins.memory.composite import CompositeMemoryProvider
from plugins.memory.composite.shadow import (
    SHADOW_STAGING_HINDSIGHT_BANK_ID,
    SHADOW_STAGING_HONCHO_SESSION_PREFIX,
    SHADOW_STAGING_HONCHO_WORKSPACE,
)


class FakeChildProvider(MemoryProvider):
    def __init__(
        self,
        name,
        *,
        tools=None,
        context="",
        init_error=None,
        sync_error=None,
        tool_response=None,
    ):
        self._name = name
        self._tools = list(tools or [])
        self._context = context
        self._init_error = init_error
        self._sync_error = sync_error
        self._tool_response = tool_response or {"ok": True}
        self.initialize_calls = []
        self.sync_calls = []
        self.prefetch_calls = []
        self.tool_calls = []

    @property
    def name(self):
        return self._name

    def is_available(self):
        return True

    def initialize(self, session_id: str, **kwargs):
        self.initialize_calls.append({"session_id": session_id, **kwargs})
        if self._init_error:
            raise self._init_error

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self.prefetch_calls.append({"query": query, "session_id": session_id})
        return self._context

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        self.sync_calls.append(
            {
                "user_content": user_content,
                "assistant_content": assistant_content,
                "session_id": session_id,
            }
        )
        if self._sync_error:
            raise self._sync_error

    def get_tool_schemas(self):
        return list(self._tools)

    def handle_tool_call(self, tool_name, args, **kwargs):
        self.tool_calls.append({"tool_name": tool_name, "args": args, **kwargs})
        return json.dumps({"child": self._name, "tool_name": tool_name, **self._tool_response})


class FakeChildFactory:
    def __init__(self, provider):
        self.provider = provider
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.provider


SHARED_TOOL = {
    "name": "shared_lookup",
    "description": "Shared tool name for collision tests.",
    "parameters": {"type": "object", "properties": {}},
}


def test_composite_with_all_children_off_initializes_empty():
    alpha_factory = FakeChildFactory(FakeChildProvider("alpha"))
    beta_factory = FakeChildFactory(FakeChildProvider("beta"))
    provider = CompositeMemoryProvider(
        config={
            "children": [
                {"name": "alpha", "enabled": False, "provider_class": alpha_factory},
                {"name": "beta", "enabled": False, "provider_class": beta_factory},
            ]
        }
    )

    provider.initialize("session-1", platform="cli")

    assert provider.get_tool_schemas() == []
    assert provider.prefetch("hello", session_id="session-1") == ""
    assert provider._children == []
    assert alpha_factory.calls == 0
    assert beta_factory.calls == 0


def test_child_initialize_failure_does_not_fail_provider_initialize():
    failing = FakeChildProvider("failing", init_error=RuntimeError("boom"))
    healthy = FakeChildProvider("healthy")
    provider = CompositeMemoryProvider(
        config={
            "children": [
                {"name": "failing", "enabled": True, "provider_class": FakeChildFactory(failing)},
                {"name": "healthy", "enabled": True, "provider_class": FakeChildFactory(healthy)},
            ]
        }
    )

    provider.initialize("session-2", platform="cli")

    assert [child["name"] for child in provider._children] == ["healthy"]
    assert len(failing.initialize_calls) == 1
    assert len(healthy.initialize_calls) == 1


def test_child_sync_failure_does_not_fail_sync_turn():
    failing = FakeChildProvider("failing", sync_error=RuntimeError("sync boom"))
    healthy = FakeChildProvider("healthy")
    provider = CompositeMemoryProvider(
        config={
            "children": [
                {"name": "failing", "enabled": True, "provider_class": FakeChildFactory(failing)},
                {"name": "healthy", "enabled": True, "provider_class": FakeChildFactory(healthy)},
            ]
        }
    )
    provider.initialize("session-3", platform="cli")

    provider.sync_turn("user", "assistant", session_id="session-3")

    assert len(failing.sync_calls) == 1
    assert len(healthy.sync_calls) == 1


def test_tool_collision_is_namespaced():
    alpha = FakeChildProvider("alpha", tools=[SHARED_TOOL])
    beta = FakeChildProvider("beta", tools=[SHARED_TOOL])
    provider = CompositeMemoryProvider(
        config={
            "tools_enabled": True,
            "children": [
                {"name": "alpha", "enabled": True, "provider_class": FakeChildFactory(alpha)},
                {"name": "beta", "enabled": True, "provider_class": FakeChildFactory(beta)},
            ],
        }
    )
    provider.initialize("session-4", platform="cli")

    schemas = provider.get_tool_schemas()
    names = sorted(schema["name"] for schema in schemas)

    assert names == [
        "composite_alpha_shared_lookup",
        "composite_beta_shared_lookup",
    ]
    assert alpha.tool_calls == []
    assert beta.tool_calls == []


def test_tools_enabled_false_exposes_no_child_tools():
    alpha = FakeChildProvider("alpha", tools=[SHARED_TOOL])
    provider = CompositeMemoryProvider(
        config={
            "tools_enabled": False,
            "children": [
                {"name": "alpha", "enabled": True, "provider_class": FakeChildFactory(alpha)},
            ],
        }
    )
    provider.initialize("session-5", platform="cli")

    assert provider.get_tool_schemas() == []


def test_namespaced_tool_call_routes_to_child_provider():
    alpha = FakeChildProvider("alpha", tools=[SHARED_TOOL], tool_response={"saved": True})
    provider = CompositeMemoryProvider(
        config={
            "tools_enabled": True,
            "children": [
                {"name": "alpha", "enabled": True, "provider_class": FakeChildFactory(alpha)},
            ],
        }
    )
    provider.initialize("session-5b", platform="cli")

    result = json.loads(
        provider.handle_tool_call("composite_alpha_shared_lookup", {"query": "hello"})
    )

    assert result == {"child": "alpha", "tool_name": "shared_lookup", "saved": True}
    assert alpha.tool_calls == [{"tool_name": "shared_lookup", "args": {"query": "hello"}}]


def test_injection_enabled_false_returns_no_prompt_context():
    alpha = FakeChildProvider("alpha", context="alpha context")
    beta = FakeChildProvider("beta", context="beta context")
    provider = CompositeMemoryProvider(
        config={
            "injection_enabled": False,
            "children": [
                {"name": "alpha", "enabled": True, "provider_class": FakeChildFactory(alpha)},
                {"name": "beta", "enabled": True, "provider_class": FakeChildFactory(beta)},
            ],
        }
    )
    provider.initialize("session-6", platform="cli")

    result = provider.prefetch("hello", session_id="session-6")

    assert result == ""
    assert alpha.prefetch_calls == []
    assert beta.prefetch_calls == []


def test_dict_keyed_wave3_config_shape_is_supported():
    hindsight = FakeChildProvider("hindsight")
    honcho = FakeChildProvider("honcho", context="inferred context should stay hidden")
    provider = CompositeMemoryProvider(
        config={
            "enabled": True,
            "mode": "shadow",
            "injection_enabled": False,
            "tools_enabled": False,
            "canonical_first": True,
            "fail_open": True,
            "max_prefetch_chars": 2000,
            "children": {
                "rasputin": {"enabled": False, "read": "primary", "write": False},
                "hindsight": {"enabled": True, "read": "shadow", "write": "shadow", "provider_class": FakeChildFactory(hindsight)},
                "honcho": {
                    "enabled": True,
                    "read": "shadow",
                    "write": "curated",
                    "inject_inferred": False,
                    "provider_class": FakeChildFactory(honcho),
                },
            },
        }
    )

    provider.initialize("session-dict", platform="cli")

    assert [child["name"] for child in provider._children] == ["hindsight", "honcho"]
    assert provider.get_tool_schemas() == []
    assert provider.prefetch("hello", session_id="session-dict") == ""
    assert len(hindsight.initialize_calls) == 1
    assert len(honcho.initialize_calls) == 1


def test_dict_keyed_live_config_constructs_bundled_children_without_provider_class(monkeypatch):
    """YAML-safe live config should create real bundled children, not an empty shell."""
    hindsight = FakeChildProvider("hindsight")
    honcho = FakeChildProvider("honcho")

    import plugins.memory.hindsight as hindsight_module
    import plugins.memory.honcho as honcho_module

    monkeypatch.setattr(
        hindsight_module,
        "HindsightMemoryProvider",
        FakeChildFactory(hindsight),
    )
    monkeypatch.setattr(
        honcho_module,
        "HonchoMemoryProvider",
        FakeChildFactory(honcho),
    )

    provider = CompositeMemoryProvider(
        config={
            "enabled": True,
            "mode": "shadow",
            "injection_enabled": False,
            "tools_enabled": False,
            "children": {
                "rasputin": {"enabled": False, "read": "primary", "write": False},
                "hindsight": {"enabled": True, "read": "shadow", "write": "shadow"},
                "honcho": {"enabled": True, "read": "shadow", "write": "curated"},
            },
        }
    )

    provider.initialize("session-live-config", platform="cli")

    assert [child["name"] for child in provider._children] == ["hindsight", "honcho"]
    assert [child["provider"].name for child in provider._children] == ["hindsight", "honcho"]
    assert len(hindsight.initialize_calls) == 1
    assert hindsight.initialize_calls[0]["session_id"] == "session-live-config"
    assert len(honcho.initialize_calls) == 1
    assert honcho.initialize_calls[0]["session_id"] == "session-live-config"


def test_sync_turn_skips_children_with_write_disabled():
    hindsight = FakeChildProvider("hindsight")
    honcho = FakeChildProvider("honcho")
    provider = CompositeMemoryProvider(
        config={
            "children": {
                "hindsight": {"enabled": True, "write": False, "provider_class": FakeChildFactory(hindsight)},
                "honcho": {"enabled": True, "write": True, "provider_class": FakeChildFactory(honcho)},
            }
        }
    )

    provider.initialize("session-write-gate", platform="cli")
    provider.sync_turn("user", "assistant", session_id="session-write-gate")

    assert hindsight.sync_calls == []
    assert len(honcho.sync_calls) == 1
    assert honcho.sync_calls[0]["session_id"] == "session-write-gate"


def test_config_schema_documents_wave3_dict_shape_and_safety_defaults():
    schema = {item["key"]: item for item in CompositeMemoryProvider().get_config_schema()}

    assert schema["children"]["default"] == {}
    assert schema["tools_enabled"]["default"] is False
    assert schema["injection_enabled"]["default"] is False
    assert schema["fail_open"]["default"] is True
    assert schema["canonical_first"]["default"] is True
    assert schema["mode"]["default"] == "shadow"
    assert schema["max_prefetch_chars"]["default"] == 2000
    assert schema["shadow_write_enabled"]["default"] is False
    assert schema["fake_clients_required"]["default"] is True
    assert schema["allow_non_fake_staging_clients"]["default"] is False
    assert schema["hindsight_staging_bank_id"]["default"] == SHADOW_STAGING_HINDSIGHT_BANK_ID
    assert schema["honcho_staging_workspace"]["default"] == SHADOW_STAGING_HONCHO_WORKSPACE
    assert schema["honcho_staging_session_prefix"]["default"] == SHADOW_STAGING_HONCHO_SESSION_PREFIX
