"""Adapter construction helpers."""

from __future__ import annotations

from adele_runner.config import AppConfig
from adele_runner.runtime.types import AdapterKind


def build_request_response_adapter(
    adapter_kind: AdapterKind,
    config: AppConfig,
    *,
    rate_limiter: object | None = None,
    configured_rate_limits=None,  # type: ignore[no-untyped-def]
):
    """Build a request-response adapter for the given provider."""
    if adapter_kind == "foundry":
        from adele_runner.adapters.foundry import FoundryAdapter

        return FoundryAdapter(
            config,
            rate_limiter=rate_limiter,
            configured_rate_limits=configured_rate_limits,
        )
    if adapter_kind == "google_genai":
        from adele_runner.adapters.google_genai import GoogleGenAIAdapter

        return GoogleGenAIAdapter(config, rate_limiter=rate_limiter)
    raise ValueError(f"Adapter '{adapter_kind}' does not support request-response execution.")


def build_batch_adapter(adapter_kind: AdapterKind, config: AppConfig):
    """Build a batch adapter for the given provider."""
    if adapter_kind == "azure_openai":
        from adele_runner.adapters.azure_openai import AzureOpenAIAdapter

        return AzureOpenAIAdapter(config)
    raise ValueError(f"Adapter '{adapter_kind}' does not support batch execution.")
