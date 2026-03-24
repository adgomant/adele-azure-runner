"""Provider adapter shims."""

from adele_runner.adapters.azure_openai import (
    AzureOpenAIAdapter,
    AzureOpenAIBatchAdapter,
    AzureOpenAIRequestResponseAdapter,
)
from adele_runner.adapters.foundry import FoundryAdapter
from adele_runner.adapters.google_genai import GoogleGenAIAdapter

__all__ = [
    "AzureOpenAIAdapter",
    "AzureOpenAIBatchAdapter",
    "AzureOpenAIRequestResponseAdapter",
    "FoundryAdapter",
    "GoogleGenAIAdapter",
]
