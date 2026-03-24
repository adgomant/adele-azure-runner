"""Compatibility shims for Azure OpenAI adapters."""

from adele_runner.adapters.providers.azure_openai.batch import AzureOpenAIBatchAdapter
from adele_runner.adapters.providers.azure_openai.request_response import (
    AzureOpenAIRequestResponseAdapter,
)

AzureOpenAIAdapter = AzureOpenAIBatchAdapter

__all__ = [
    "AzureOpenAIAdapter",
    "AzureOpenAIBatchAdapter",
    "AzureOpenAIRequestResponseAdapter",
]
