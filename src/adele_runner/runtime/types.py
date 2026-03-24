"""Internal transport and resolution contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from adele_runner.config import RateLimitsConfig

AdapterKind = Literal["foundry", "google_genai", "azure_openai"]
ExecutionKind = Literal["request_response", "batch"]
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
    """Capabilities exposed by a provider adapter."""

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
class ResolvedInferenceTarget:
    """Internal inference target resolved from the public config surface."""

    adapter_kind: AdapterKind
    execution_kind: ExecutionKind
    model: str
    temperature: float
    max_tokens: int
    top_p: float
    rate_limits: RateLimitsConfig | None = None


@dataclass(frozen=True, slots=True)
class ResolvedJudgeTarget:
    """Internal judge target resolved from the public config surface."""

    judge_name: str
    adapter_kind: AdapterKind
    execution_kind: ExecutionKind
    model: str
    prompt_template: str
    max_tokens: int
    rate_limits: RateLimitsConfig | None = None

