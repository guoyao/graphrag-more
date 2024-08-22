#!/usr/bin/env python3
# coding=utf-8

from enum import Enum, unique
from functools import wraps
from typing import Dict, Callable, Optional
from typing import Tuple, Union

from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages.base import BaseMessage

from graphrag.llm.others import qianfan, tongyi, ollama

_llm_creators: Dict[str, Callable[..., BaseLanguageModel[Union[str, BaseMessage]]]] = {}
_chat_llm_creators: Dict[str, Callable[..., BaseChatModel]] = {}
_embeddings_creators: Dict[str, Callable[..., Embeddings]] = {}

_global_type: Optional[str] = None
_global_embeddings_type: Optional[str] = None


def once(func):
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'result' in cache:
            return cache['result']
        try:
            cache['result'] = func(*args, **kwargs)
            return cache['result']
        except Exception as e:
            cache['result'] = None
            raise e

    return wrapper


@unique
class LLMType(Enum):
    QIANFAN = 'qianfan'  # Baidu Qianfan
    TONGYI = 'tongyi'  # Alibaba Tongyi
    OLLAMA = 'ollama'  # Ollama


def is_valid_llm_type(llm_type: str) -> bool:
    return llm_type in [v.value for v in LLMType]


def llm_type_to_str(llm_type: Union[str, LLMType]) -> str:
    return llm_type.value if isinstance(llm_type, LLMType) else llm_type


@once
def set_global_type(llm_type: Union[str, LLMType]):
    global _global_type
    _global_type = llm_type_to_str(llm_type)


@once
def set_global_embeddings_type(embeddings_type: Union[str, LLMType]):
    global _global_embeddings_type
    _global_embeddings_type = llm_type_to_str(embeddings_type)


def register_llm_creator(llm_type: Union[str, LLMType],
                         creator: Callable[..., BaseLanguageModel[Union[str, BaseMessage]]]):
    llm_type = llm_type_to_str(llm_type)
    if llm_type in _llm_creators:
        raise RuntimeError(f'The specified llm `{llm_type}` has been registered.')
    _llm_creators[llm_type] = creator


def register_chat_llm_creator(llm_type: Union[str, LLMType],
                              creator: Callable[..., BaseChatModel]):
    llm_type = llm_type_to_str(llm_type)
    if llm_type in _chat_llm_creators:
        raise RuntimeError(f'The specified chat llm `{llm_type}` has been registered.')
    _chat_llm_creators[llm_type] = creator


def register_embeddings_creator(llm_type: Union[str, LLMType],
                                creator: Callable[..., Embeddings]):
    llm_type = llm_type_to_str(llm_type)
    if llm_type in _embeddings_creators:
        raise RuntimeError(f'The specified embeddings `{llm_type}` has been registered.')
    _embeddings_creators[llm_type] = creator


def register_creators(llm_type: Union[str, LLMType],
                      llm_creator: Callable[..., BaseLanguageModel[Union[str, BaseMessage]]],
                      chat_llm_creator: Callable[..., BaseChatModel],
                      embeddings_creator: Callable[..., Embeddings]):
    register_llm_creator(llm_type, llm_creator)
    register_chat_llm_creator(llm_type, chat_llm_creator)
    register_embeddings_creator(llm_type, embeddings_creator)


def use_llm(llm_type: Optional[Union[str, LLMType]] = None, **kwargs) -> \
        BaseLanguageModel[Union[str, BaseMessage]]:
    _register_all()

    llm_type = llm_type or _global_type
    if llm_type is None:
        raise RuntimeError(f'You must provide llm type or set global type')

    llm_type = llm_type_to_str(llm_type)
    if llm_type not in _llm_creators:
        raise RuntimeError(f'The specified llm `{llm_type}` does not exist'
                           f'{"." if llm_type is None else ", it must be registered before using."}')

    return _llm_creators[llm_type](**kwargs)


def use_chat_llm(llm_type: Optional[Union[str, LLMType]] = None, **kwargs) -> BaseChatModel:
    _register_all()

    llm_type = llm_type or _global_type
    if llm_type is None:
        raise RuntimeError(f'You must provide chat llm type or set global type')

    llm_type = llm_type_to_str(llm_type)
    if llm_type not in _chat_llm_creators:
        raise RuntimeError(f'The specified chat llm `{llm_type}` does not exist'
                           f'{"." if llm_type is None else ", it must be registered before using."}')

    return _chat_llm_creators[llm_type](**kwargs)


def use_embeddings(llm_type: Optional[Union[str, LLMType]] = None, **kwargs) -> Embeddings:
    _register_all()

    llm_type = llm_type or _global_embeddings_type or _global_type
    if llm_type is None:
        raise RuntimeError(
            f'You must provide embeddings type or set global embeddings type or set global type')

    llm_type = llm_type_to_str(llm_type)
    if llm_type not in _embeddings_creators:
        raise RuntimeError(f'The specified embeddings `{llm_type}` does not exist'
                           f'{"." if llm_type is None else ", it must be registered before using."}')

    return _embeddings_creators[llm_type](**kwargs)


def use_llm_all(llm_type: Optional[Union[str, LLMType]] = None) -> \
        Tuple[BaseLanguageModel[Union[str, BaseMessage]], BaseChatModel, Embeddings]:
    return use_llm(llm_type), use_chat_llm(llm_type), use_embeddings(llm_type)


@once
def _register_all():
    creators = [
        (LLMType.QIANFAN, qianfan.creators),
        (LLMType.TONGYI, tongyi.creators),
        (LLMType.OLLAMA, ollama.creators)
    ]
    for v in creators:
        register_creators(v[0], *v[1])
