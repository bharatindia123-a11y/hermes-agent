"""CLI-only filtered memory read bridge for memory-vnext.

This module deliberately does not activate Hermes' broad memory injection/tool
surface. It wraps the standalone memory-vnext filtered read harness and returns
only per-turn ephemeral context suitable for the existing run_conversation()
user-message memory-context fence.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HARNESS_PATH = Path('/root/.hermes/profiles/thindi/workspace/memory-vnext/scripts/filtered_read_harness.py')
DEFAULT_NAMESPACE = 'hermes-thindi-memory-vnext-staging'


@dataclass(frozen=True)
class CliFilteredMemoryResult:
    context: str = ''
    metadata: dict[str, Any] = field(default_factory=dict)


class CliFilteredMemoryBridge:
    """Small CLI-only bridge around the filtered memory-vnext harness.

    The bridge is intentionally conservative:
    - no context is returned unless explicitly enabled;
    - skip_memory disables it;
    - raw adversarial text reported by the harness blocks injection;
    - errors degrade to empty context;
    - the caller is expected to additionally gate on platform == "cli".
    """

    def __init__(
        self,
        *,
        harness: Any | None = None,
        harness_path: str | Path = DEFAULT_HARNESS_PATH,
        enabled: bool = False,
        skip_memory: bool = False,
        namespace: str = DEFAULT_NAMESPACE,
        latency_budget_ms: int = 2000,
        max_blocks: int = 2,
        honcho_limit: int = 3,
        hindsight_max_tokens: int = 1000,
        hindsight_fetch_enabled: bool = False,
        hindsight_cache_enabled: bool = False,
        hindsight_cache_path: str | Path | None = None,
        hindsight_cache_ttl_seconds: int = 300,
    ) -> None:
        self.enabled = bool(enabled)
        self.skip_memory = bool(skip_memory)
        self.namespace = namespace or DEFAULT_NAMESPACE
        self.latency_budget_ms = int(latency_budget_ms or 2000)
        self.max_blocks = int(max_blocks or 2)
        self.honcho_limit = int(honcho_limit or 3)
        self.hindsight_max_tokens = int(hindsight_max_tokens or 1000)
        self.hindsight_fetch_enabled = bool(hindsight_fetch_enabled)
        self.hindsight_cache_enabled = bool(hindsight_cache_enabled)
        self.hindsight_cache_path = Path(hindsight_cache_path) if hindsight_cache_path else None
        self.hindsight_cache_ttl_seconds = int(hindsight_cache_ttl_seconds or 300)
        self._harness_path = Path(harness_path)
        self._harness = harness
        self._query_cache = None

    def is_enabled(self) -> bool:
        return self.enabled and not self.skip_memory

    def _load_harness(self) -> Any:
        if self._harness is not None:
            return self._harness
        path = self._harness_path
        if not path.exists():
            raise FileNotFoundError(f'filtered memory harness not found: {path}')
        module_name = 'hermes_memory_vnext_filtered_read_harness'
        cached = sys.modules.get(module_name)
        if cached is not None:
            self._harness = cached
            return cached
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot import filtered memory harness from {path}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self._harness = module
        return module

    def prefetch(self, query: str) -> CliFilteredMemoryResult:
        if not self.is_enabled():
            return CliFilteredMemoryResult(metadata={'enabled': False})
        if not isinstance(query, str) or not query.strip():
            return CliFilteredMemoryResult(metadata={'enabled': True, 'blocked_reason': 'empty_query'})
        try:
            harness = self._load_harness()
            if self._query_cache is None and hasattr(harness, 'QueryCache'):
                self._query_cache = harness.QueryCache(ttl_seconds=300)

            honcho_candidates, honcho_latency, honcho_error = harness.fetch_honcho(
                query,
                namespace=self.namespace,
                limit=self.honcho_limit,
            )

            def _fetch_hindsight() -> Any:
                return harness.fetch_hindsight(
                    query,
                    namespace=self.namespace,
                    max_tokens=self.hindsight_max_tokens,
                )

            hindsight_cache_status = 'disabled'
            if self.hindsight_fetch_enabled:
                if self._query_cache is not None:
                    hindsight_candidates, hindsight_latency, hindsight_error = self._query_cache.get_or_set(
                        f'hindsight:{self.namespace}:{query}',
                        _fetch_hindsight,
                    )
                else:
                    hindsight_candidates, hindsight_latency, hindsight_error = _fetch_hindsight()
                hindsight_cache_status = 'live_fetch_enabled'
            elif self.hindsight_cache_enabled and self.hindsight_cache_path and hasattr(harness, 'HindsightPrewarmCache'):
                cache = harness.HindsightPrewarmCache(self.hindsight_cache_path, ttl_seconds=self.hindsight_cache_ttl_seconds)
                hindsight_candidates, cache_meta = cache.load(query, namespace=self.namespace, mode='default')
                hindsight_latency = 0.0
                hindsight_error = None if cache_meta.get('cache_status') == 'hit' else f"hindsight_cache_{cache_meta.get('cache_status')}"
                hindsight_cache_status = str(cache_meta.get('cache_status') or 'unknown')
            else:
                hindsight_candidates, hindsight_latency, hindsight_error = [], 0.0, 'hindsight_fetch_disabled_not_prewarmed'

            candidates = list(honcho_candidates or []) + list(hindsight_candidates or [])
            policy = harness.FilterPolicy(
                namespace=self.namespace,
                mode='default',
                max_blocks=self.max_blocks,
                latency_budget_ms=self.latency_budget_ms,
                allow_historical_evidence=False,
            )
            context_result = harness.build_context(query, candidates, policy=policy)
            rendered = str(context_result.get('rendered_context') or '')
            metadata = {
                'enabled': True,
                'namespace': self.namespace,
                'gate_status': context_result.get('gate_status'),
                'latency_budget_exceeded': bool(context_result.get('latency_budget_exceeded')),
                'raw_adversarial_text_present': bool(context_result.get('raw_adversarial_text_present')),
                'injected_blocks': len(context_result.get('injected_blocks') or []),
                'dropped_hits': len(context_result.get('dropped_hits') or []),
                'backend_errors': {
                    'honcho': honcho_error,
                    'hindsight': hindsight_error,
                },
                'backend_latency_ms': {
                    'honcho': honcho_latency,
                    'hindsight': hindsight_latency,
                },
                'hindsight_fetch_enabled': self.hindsight_fetch_enabled,
                'hindsight_cache_enabled': self.hindsight_cache_enabled,
                'hindsight_cache_status': hindsight_cache_status,
            }
            if metadata['raw_adversarial_text_present']:
                metadata['blocked_reason'] = 'raw_adversarial_text_present'
                return CliFilteredMemoryResult(metadata=metadata)
            if not rendered.strip():
                return CliFilteredMemoryResult(metadata=metadata)
            return CliFilteredMemoryResult(context=rendered, metadata=metadata)
        except Exception as exc:
            logger.debug('CLI filtered memory bridge failed (non-fatal): %s', exc)
            return CliFilteredMemoryResult(metadata={'enabled': True, 'error': str(exc)})
