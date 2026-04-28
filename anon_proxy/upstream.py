"""Upstream provider registry and configuration.

Maps provider prefixes to their base URLs and adapter types.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from anon_proxy.masker import Masker


@dataclass(frozen=True)
class UpstreamConfig:
    """Configuration for an upstream API provider."""

    name: str
    """Provider identifier (used in URL path)."""

    base_url: str
    """Base URL for the upstream API."""

    path_prefix: str = ""
    """Optional path prefix to insert after provider name (e.g., 'api/anthropic')."""

    adapter: str = "anthropic"
    """Adapter to use: 'anthropic' or 'openai'."""

    sse: bool = True
    """Whether the upstream uses Server-Sent Events for streaming."""


# Built-in upstream configurations
BUILT_IN_UPSTREAMS: dict[str, UpstreamConfig] = {
    "anthropic": UpstreamConfig(
        name="anthropic",
        base_url="https://api.anthropic.com",
        adapter="anthropic",
        sse=True,
    ),
    "openai": UpstreamConfig(
        name="openai",
        base_url="https://api.openai.com",
        path_prefix="v1",
        adapter="openai",
        sse=True,
    ),
    "zai": UpstreamConfig(
        name="zai",
        base_url="https://api.z.ai",
        path_prefix="api/anthropic",
        adapter="anthropic",
        sse=True,
    ),
}


def get_upstream_config(provider: str, extra_upstreams: dict[str, UpstreamConfig] | None = None) -> UpstreamConfig:
    """Get upstream configuration for a provider.

    Args:
        provider: Provider name (e.g., 'anthropic', 'openai', 'zai')
        extra_upstreams: Additional upstreams configured via CLI/env

    Returns:
        UpstreamConfig for the provider

    Raises:
        ValueError: If provider is not found
    """
    all_upstreams = {**BUILT_IN_UPSTREAMS, **(extra_upstreams or {})}
    config = all_upstreams.get(provider)
    if config is None:
        raise ValueError(f"Unknown upstream provider: {provider}. Available: {sorted(all_upstreams)}")
    return config
