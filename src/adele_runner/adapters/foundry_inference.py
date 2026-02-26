from __future__ import annotations

import os
from typing import Any

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

from adele_runner.config import FoundryConfig


class FoundryChatAdapter:
    def __init__(self, config: FoundryConfig) -> None:
        self._config = config
        credential = self._build_credential(config)
        self._client = ChatCompletionsClient(endpoint=config.endpoint, credential=credential, api_version=config.api_version)

    def _build_credential(self, config: FoundryConfig) -> AzureKeyCredential | DefaultAzureCredential:
        if config.api_key_env:
            value = os.getenv(config.api_key_env)
            if value:
                return AzureKeyCredential(value)
        return DefaultAzureCredential()

    def complete(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> dict[str, Any]:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(UserMessage(content=prompt))

        response = self._client.complete(
            messages=messages,
            model=self._config.model,
            temperature=kwargs.get("temperature", self._config.temperature),
            top_p=kwargs.get("top_p", self._config.top_p),
            max_tokens=kwargs.get("max_tokens", self._config.max_tokens),
        )
        text = response.choices[0].message.content if response.choices else ""
        return {"text": text, "raw": response.as_dict()}
