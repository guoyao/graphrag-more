#!/usr/bin/env python3
# coding=utf-8

"""
Baidu Qianfan: https://qianfan.cloud.baidu.com/
"""

import os
from typing import Union

from langchain_community.chat_models import QianfanChatEndpoint
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.llms import QianfanLLMEndpoint
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import BaseMessage

# https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu
INSTRUCT_MODEL = 'ERNIE-Speed-8K'
CHAT_MODEL = INSTRUCT_MODEL
EMBEDDINGS_MODEL = 'bge-large-zh'

common_options = {
    'qianfan_ak': os.getenv('QIANFAN_AK'),
    'qianfan_sk': os.getenv('QIANFAN_SK')
}

_llm, _chat_llm, _embeddings = None, None, None


def create_llm(**kwargs) -> BaseLanguageModel[Union[str, BaseMessage]]:
    """create `QianfanLLM`, can be used to replace `OpenAI`"""
    global _llm

    if len(kwargs) == 0:
        if _llm is None:
            _llm = QianfanLLMEndpoint(model=INSTRUCT_MODEL, **common_options)
        return _llm

    options = {'model': INSTRUCT_MODEL, **common_options, **kwargs}
    return QianfanLLMEndpoint(**options)


def create_chat_llm(**kwargs) -> BaseChatModel:
    """create `QianfanChat`, can be used to replace `ChatOpenAI`"""
    global _chat_llm

    if len(kwargs) == 0:
        if _chat_llm is None:
            _chat_llm = QianfanChatEndpoint(model=CHAT_MODEL, **common_options)
        return _chat_llm

    options = {'model': CHAT_MODEL, **common_options, **kwargs}
    return QianfanChatEndpoint(**options)


def create_embeddings(**kwargs) -> Embeddings:
    """create `QianfanEmbeddings`, can be used to replace `OpenAIEmbeddings`"""
    global _embeddings

    if len(kwargs) == 0:
        if _embeddings is None:
            _embeddings = QianfanEmbeddingsEndpoint(model=EMBEDDINGS_MODEL, **common_options)
        return _embeddings

    options = {'model': EMBEDDINGS_MODEL, **common_options, **kwargs}
    return QianfanEmbeddingsEndpoint(**options)


creators = (create_llm, create_chat_llm, create_embeddings)
