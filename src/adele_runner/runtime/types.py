"""Internal transport and runtime contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from adele_runner.config import RateLimitsConfig

ProviderKind = Literal["azure_openai", "azure_ai_inference", "google_genai", "anthropic"]
ExecutionKind = Literal["request_response", "batch"]
ExecutionMode = Literal["request_response", "batch", "auto"]
MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A normalized chat message for provider adapters."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class ChatRequest:
    """A normalized request for request-response or batch execution."""

    request_id: str
    model: str
    messages: tuple[ChatMessage, ...]
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """A normalized response returned by provider adapters."""

    request_id: str
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_s: float | None = None
    raw_output: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Capabilities exposed by a provider mode adapter."""

    request_response: bool = False
    batch: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Execution settings resolved for a single pipeline lane."""

    max_in_flight: int
    request_timeout_s: float
    max_retries: int
    backoff_base_s: float
    backoff_max_s: float
    max_poll_time_s: float
    batch_completion_window: str
    effective_rpm: int | None = None


@dataclass(frozen=True, slots=True)
class CapabilityResolution:
    """Result of validating whether a target supports a mode."""

    supported: bool
    reason: str | None = None
    resolved_mode: ExecutionKind | None = None


@dataclass(frozen=True, slots=True)
class ResolvedProviderTarget:
    """Provider-specific target resolved from public config."""

    provider_kind: ProviderKind
    model: str
    rate_limits: RateLimitsConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedModeBinding:
    """Mode binding resolved from provider descriptor + capability checks."""

    provider_target: ResolvedProviderTarget
    execution_kind: ExecutionKind
    create_adapter: Callable[..., Any]
    capability_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedInferenceTarget:
    """Internal inference target resolved from the public config surface."""

    provider_target: ResolvedProviderTarget
    requested_mode: ExecutionMode
    prompt_mode: ExecutionKind
    temperature: float
    max_tokens: int
    top_p: float

    @property
    def model(self) -> str:
        return self.provider_target.model

    @property
    def provider_kind(self) -> ProviderKind:
        return self.provider_target.provider_kind

    @property
    def rate_limits(self) -> RateLimitsConfig | None:
        return self.provider_target.rate_limits


@dataclass(frozen=True, slots=True)
class ResolvedJudgeTarget:
    """Internal judge target resolved from the public config surface."""

    judge_name: str
    provider_target: ResolvedProviderTarget
    requested_mode: ExecutionMode
    prompt_mode: ExecutionKind
    prompt_template: str
    max_tokens: int

    @property
    def model(self) -> str:
        return self.provider_target.model

    @property
    def provider_kind(self) -> ProviderKind:
        return self.provider_target.provider_kind

    @property
    def rate_limits(self) -> RateLimitsConfig | None:
        return self.provider_target.rate_limits
