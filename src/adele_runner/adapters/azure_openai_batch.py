"""Compatibility shim for the shared Azure OpenAI batch adapter."""

from adele_runner.adapters.azure_openai import AzureOpenAIAdapter

AzureOpenAIBatchAdapter = AzureOpenAIAdapter

__all__ = ["AzureOpenAIBatchAdapter", "AzureOpenAIAdapter"]
