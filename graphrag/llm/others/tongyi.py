#!/usr/bin/env python3
# coding=utf-8

"""
Alibaba Tongyi: https://tongyi.aliyun.com/
"""

import os
from typing import Union

from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import BaseMessage

# https://help.aliyun.com/zh/model-studio/user-guide/tongyi-qianwen
INSTRUCT_MODEL = 'qwen-turbo'
CHAT_MODEL = INSTRUCT_MODEL
EMBEDDINGS_MODEL = 'text-embedding-v2'

common_options = {
    'dashscope_api_key': os.getenv('TONGYI_API_KEY')
}

_llm, _chat_llm, _embeddings = None, None, None


def create_llm(**kwargs) -> BaseLanguageModel[Union[str, BaseMessage]]:
    """create `Tongyi`, can be used to replace `OpenAI`"""
    global _llm

    if len(kwargs) == 0:
        if _llm is None:
            _llm = Tongyi(model=INSTRUCT_MODEL, **common_options)
        return _llm

    options = {'model': INSTRUCT_MODEL, **common_options, **kwargs}
    return Tongyi(**options)


def create_chat_llm(**kwargs) -> BaseChatModel:
    """create `ChatTongyi`, can be used to replace `ChatOpenAI`"""
    global _chat_llm

    if len(kwargs) == 0:
        if _chat_llm is None:
            _chat_llm = ChatTongyi(model=CHAT_MODEL, **common_options)
        return _chat_llm

    options = {'model': CHAT_MODEL, **common_options, **kwargs}
    return ChatTongyi(**options)


def create_embeddings(**kwargs) -> Embeddings:
    """create `DashScopeEmbeddings`, can be used to replace `OpenAIEmbeddings`"""
    global _embeddings

    if len(kwargs) == 0:
        if _embeddings is None:
            _embeddings = DashScopeEmbeddings(model=EMBEDDINGS_MODEL, **common_options)
        return _embeddings

    options = {'model': EMBEDDINGS_MODEL, **common_options, **kwargs}
    return DashScopeEmbeddings(**options)


creators = (create_llm, create_chat_llm, create_embeddings)
