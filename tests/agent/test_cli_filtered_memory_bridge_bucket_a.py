from __future__ import annotations

import inspect

from agent.cli_filtered_memory_bridge import CliFilteredMemoryBridge


class StubPolicy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class StubPrewarmCache:
    def __init__(self, path, ttl_seconds=300):
        self.path = path
        self.ttl_seconds = ttl_seconds

    def load(self, query, *, namespace, mode="default"):
        return ["cached-hindsight-candidate"], {"cache_status": "hit", "cache_age_seconds": 1.0}


class StubHarness:
    FilterPolicy = StubPolicy
    HindsightPrewarmCache = StubPrewarmCache

    def __init__(self):
        self.calls = []

    def fetch_honcho(self, query, *, namespace, limit=4):
        self.calls.append(("honcho", query, namespace, limit))
        return ["honcho-candidate"], 7.0, None

    def fetch_hindsight(self, query, *, namespace, max_tokens=1000):
        self.calls.append(("hindsight", query, namespace, max_tokens))
        raise AssertionError("live Hindsight fetch must stay disabled in Bucket A")

    def build_context(self, query, candidates, *, policy):
        self.calls.append(("build", query, list(candidates), policy))
        return {
            "gate_status": "pass",
            "rendered_context": "source=honcho\nsource=hindsight",
            "injected_blocks": [{"backend": "honcho"}, {"backend": "hindsight"}],
            "dropped_hits": [],
            "raw_adversarial_text_present": False,
        }


def test_bucket_a_cli_bridge_uses_hindsight_cache_without_live_fetch():
    harness = StubHarness()
    bridge = CliFilteredMemoryBridge(
        harness=harness,
        enabled=True,
        namespace="staging",
        hindsight_fetch_enabled=False,
        hindsight_cache_enabled=True,
        hindsight_cache_path="/tmp/fake-cache.json",
    )

    result = bridge.prefetch("cached canonical query")

    assert "source=honcho" in result.context
    assert "source=hindsight" in result.context
    assert [call[0] for call in harness.calls] == ["honcho", "build"]
    assert result.metadata["hindsight_fetch_enabled"] is False
    assert result.metadata["hindsight_cache_enabled"] is True
    assert result.metadata["hindsight_cache_status"] == "hit"


def test_bucket_a_run_agent_guards_broad_prefetch_but_allows_honcho_sync():
    import run_agent
    from agent import conversation_loop

    src = inspect.getsource(conversation_loop.run_conversation)
    guard = "not (agent._cli_filtered_memory_bridge and agent._cli_filtered_memory_bridge.is_enabled())"
    assert guard in src
    assert src.index(guard) < src.index("agent._memory_manager.prefetch_all(_query)")
    assert "agent._cli_filtered_memory_bridge.prefetch(_query)" in src

    sync_src = inspect.getsource(run_agent.AIAgent._sync_external_memory_for_turn)
    sync_guard = "not (self._cli_filtered_memory_bridge and self._cli_filtered_memory_bridge.is_enabled())"
    assert "self._memory_manager.sync_all" in sync_src
    assert sync_guard in sync_src
    assert sync_src.index("self._memory_manager.sync_all") < sync_src.index(sync_guard)
    assert sync_src.index(sync_guard) < sync_src.index("self._memory_manager.queue_prefetch_all")
