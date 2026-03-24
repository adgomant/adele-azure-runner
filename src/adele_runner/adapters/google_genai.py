"""Compatibility shim for Google GenAI request-response adapter."""

from adele_runner.adapters.providers.google_genai.request_response import (
    GoogleGenAIRequestResponseAdapter as GoogleGenAIAdapter,
)

__all__ = ["GoogleGenAIAdapter"]
