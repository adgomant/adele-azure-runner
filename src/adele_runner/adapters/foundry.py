"""Compatibility shim for Azure AI Inference request-response adapter."""

from adele_runner.adapters.providers.azure_ai_inference.request_response import (
    AzureAIInferenceRequestResponseAdapter as FoundryAdapter,
)

__all__ = ["FoundryAdapter"]
