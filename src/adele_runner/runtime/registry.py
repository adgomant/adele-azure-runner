"""Provider registry and capability resolution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from adele_runner.config import AppConfig
from adele_runner.runtime.types import (
    CapabilityResolution,
    ExecutionKind,
    ExecutionMode,
    ProviderKind,
    ResolvedModeBinding,
    ResolvedProviderTarget,
)

RequestFactory = Callable[..., Any]
BatchFactory = Callable[..., Any]
CapabilityResolver = Callable[[AppConfig, ResolvedProviderTarget, ExecutionMode], CapabilityResolution]

_AZURE_BATCH_DEPLOYMENT_TYPES = {"global_batch", "data_zone_batch"}


@dataclass(frozen=True, slots=True)
class ProviderDescriptor:
    """Describes one provider and how to bind it to a mode adapter."""

    provider_kind: ProviderKind
    request_factory: RequestFactory | None
    batch_factory: BatchFactory | None
    resolve_capability: CapabilityResolver


def _resolve_mode_auto(target: ResolvedProviderTarget, capability: set[ExecutionKind]) -> ExecutionKind:
    target_modes = target.metadata.get("supported_modes")
    if target_modes == ["batch"]:
        return "batch"
    return "request_response" if "request_response" in capability else "batch"


def _capability_azure_openai(
    config: AppConfig,
    target: ResolvedProviderTarget,
    mode: ExecutionMode,
) -> CapabilityResolution:
    if mode == "auto":
        return CapabilityResolution(True, resolved_mode=_resolve_mode_auto(target, {"request_response", "batch"}))
    if mode == "request_response":
        return CapabilityResolution(True, resolved_mode="request_response")

    target_cfg = config.get_target_config(target.model)
    if target_cfg is None:
        return CapabilityResolution(
            False,
            reason=(
                f"Target '{target.model}' needs targets.{target.model}.supported_modes, "
                "batch_capable=true, or a batch deployment_type hint for azure_openai batch mode."
            ),
        )
    if target_cfg.supported_modes and "batch" in target_cfg.supported_modes:
        return CapabilityResolution(True, resolved_mode="batch")
    if target_cfg.batch_capable:
        return CapabilityResolution(True, resolved_mode="batch")
    if target_cfg.deployment_type and target_cfg.deployment_type.lower() in _AZURE_BATCH_DEPLOYMENT_TYPES:
        return CapabilityResolution(True, resolved_mode="batch")
    return CapabilityResolution(
        False,
        reason=f"Target '{target.model}' is not marked as batch-capable for provider 'azure_openai'.",
    )


def _capability_azure_ai_inference(
    config: AppConfig,
    target: ResolvedProviderTarget,
    mode: ExecutionMode,
) -> CapabilityResolution:
    if mode == "batch":
        return CapabilityResolution(
            False,
            reason="Provider 'azure_ai_inference' does not support batch mode.",
        )
    return CapabilityResolution(True, resolved_mode="request_response")


def _capability_google_genai(
    config: AppConfig,
    target: ResolvedProviderTarget,
    mode: ExecutionMode,
) -> CapabilityResolution:
    backend = config.providers.google_genai.backend
    if mode == "auto":
        return CapabilityResolution(True, resolved_mode="request_response")
    if mode == "request_response":
        return CapabilityResolution(True, resolved_mode="request_response")
    if backend not in {"gemini_api", "vertex_ai"}:
        return CapabilityResolution(
            False,
            reason=f"Backend '{backend}' does not support google_genai batch mode.",
        )
    return CapabilityResolution(True, resolved_mode="batch")


def _capability_anthropic(
    config: AppConfig,
    target: ResolvedProviderTarget,
    mode: ExecutionMode,
) -> CapabilityResolution:
    if mode == "auto":
        return CapabilityResolution(True, resolved_mode="request_response")
    return CapabilityResolution(True, resolved_mode=mode)


def _build_azure_openai_request(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.azure_openai.request_response import (
        AzureOpenAIRequestResponseAdapter,
    )

    return AzureOpenAIRequestResponseAdapter(config, **kwargs)


def _build_azure_openai_batch(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.azure_openai.batch import AzureOpenAIBatchAdapter

    return AzureOpenAIBatchAdapter(config, **kwargs)


def _build_azure_ai_inference_request(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.azure_ai_inference.request_response import (
        AzureAIInferenceRequestResponseAdapter,
    )

    return AzureAIInferenceRequestResponseAdapter(config, **kwargs)


def _build_google_request(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.google_genai.request_response import (
        GoogleGenAIRequestResponseAdapter,
    )

    return GoogleGenAIRequestResponseAdapter(config, **kwargs)


def _build_google_batch(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.google_genai.batch import GoogleGenAIBatchAdapter

    return GoogleGenAIBatchAdapter(config, **kwargs)


def _build_anthropic_request(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.anthropic.request_response import (
        AnthropicRequestResponseAdapter,
    )

    return AnthropicRequestResponseAdapter(config, **kwargs)


def _build_anthropic_batch(config: AppConfig, **kwargs):  # noqa: ANN003
    from adele_runner.adapters.providers.anthropic.batch import AnthropicBatchAdapter

    return AnthropicBatchAdapter(config, **kwargs)


PROVIDER_REGISTRY: dict[ProviderKind, ProviderDescriptor] = {
    "azure_openai": ProviderDescriptor(
        provider_kind="azure_openai",
        request_factory=_build_azure_openai_request,
        batch_factory=_build_azure_openai_batch,
        resolve_capability=_capability_azure_openai,
    ),
    "azure_ai_inference": ProviderDescriptor(
        provider_kind="azure_ai_inference",
        request_factory=_build_azure_ai_inference_request,
        batch_factory=None,
        resolve_capability=_capability_azure_ai_inference,
    ),
    "google_genai": ProviderDescriptor(
        provider_kind="google_genai",
        request_factory=_build_google_request,
        batch_factory=_build_google_batch,
        resolve_capability=_capability_google_genai,
    ),
    "anthropic": ProviderDescriptor(
        provider_kind="anthropic",
        request_factory=_build_anthropic_request,
        batch_factory=_build_anthropic_batch,
        resolve_capability=_capability_anthropic,
    ),
}


def get_provider_descriptor(provider: ProviderKind) -> ProviderDescriptor:
    return PROVIDER_REGISTRY[provider]


def bind_mode(
    config: AppConfig,
    provider_target: ResolvedProviderTarget,
    requested_mode: ExecutionMode,
) -> ResolvedModeBinding:
    descriptor = get_provider_descriptor(provider_target.provider_kind)
    capability = descriptor.resolve_capability(config, provider_target, requested_mode)
    if not capability.supported or capability.resolved_mode is None:
        raise ValueError(capability.reason or "Unsupported provider/mode combination.")

    if capability.resolved_mode == "request_response":
        if descriptor.request_factory is None:
            raise ValueError(
                f"Provider '{provider_target.provider_kind}' does not implement request_response mode."
            )
        return ResolvedModeBinding(
            provider_target=provider_target,
            execution_kind="request_response",
            create_adapter=descriptor.request_factory,
            capability_reason=capability.reason,
        )

    if descriptor.batch_factory is None:
        raise ValueError(f"Provider '{provider_target.provider_kind}' does not implement batch mode.")
    return ResolvedModeBinding(
        provider_target=provider_target,
        execution_kind="batch",
        create_adapter=descriptor.batch_factory,
        capability_reason=capability.reason,
    )
