# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""The EmbeddingsLLM class."""

import ollama
from typing_extensions import Unpack

from graphrag.llm.base import BaseLLM
from graphrag.llm.others.factories import is_valid_llm_type, use_embeddings
from graphrag.llm.types import (
    EmbeddingInput,
    EmbeddingOutput,
    LLMInput,
)
from .openai_configuration import OpenAIConfiguration
from .types import OpenAIClientTypes


class OpenAIEmbeddingsLLM(BaseLLM[EmbeddingInput, EmbeddingOutput]):
    """A text-embedding generator LLM."""

    _client: OpenAIClientTypes
    _configuration: OpenAIConfiguration

    def __init__(self, client: OpenAIClientTypes, configuration: OpenAIConfiguration):
        self.client = client
        self.configuration = configuration

    async def _execute_llm(
        self, input: EmbeddingInput, **kwargs: Unpack[LLMInput]
    ) -> EmbeddingOutput | None:
        model = self.configuration.lookup('model', '')
        llm_type, *models = model.split('.')
        if is_valid_llm_type(llm_type):
            embeddings = use_embeddings(llm_type, model='.'.join(models))
            return await embeddings.aembed_documents(input)
        if llm_type == 'ollama':
            return [ollama.embeddings(model='.'.join(models), prompt=v)['embedding'] for v in input]

        args = {
            "model": self.configuration.model,
            **(kwargs.get("model_parameters") or {}),
        }
        embedding = await self.client.embeddings.create(
            input=input,
            **args,
        )
        return [d.embedding for d in embedding.data]
