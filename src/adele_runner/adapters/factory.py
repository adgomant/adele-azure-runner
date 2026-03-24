"""Compatibility shim for older adapter-construction imports."""

from adele_runner.runtime.registry import bind_mode, get_provider_descriptor

__all__ = ["bind_mode", "get_provider_descriptor"]
