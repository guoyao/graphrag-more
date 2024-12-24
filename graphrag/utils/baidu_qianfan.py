# Licensed under the MIT License

"""Utilities for working with Baidu Qianfan."""

from typing import List, Optional, Union

from openai.types import CreateEmbeddingResponse

_embedding_client = None


def get_embedding_client():
    global _embedding_client
    if _embedding_client is not None:
        return _embedding_client

    try:
        import qianfan
    except ImportError:
        raise Exception('No qianfan SDK installed, '
                        'install qianfan SDK if you want use Baidu Qianfan LLMs')

    _embedding_client = qianfan.Embedding()
    return _embedding_client


def embed_documents(input: Union[str, List[str]],
                    model: Optional[str] = 'Embedding-V1',
                    **kwargs) -> CreateEmbeddingResponse:
    client = get_embedding_client()
    texts = input if isinstance(input, List) else [input]
    response = client.do(texts=texts, model=model, **kwargs)
    response.body['object'] = 'list'
    response.body.setdefault('model', model)
    return CreateEmbeddingResponse(**response.body)


async def aembed_documents(input: Union[str, List[str]],
                           model: Optional[str] = 'Embedding-V1',
                           **kwargs) -> CreateEmbeddingResponse:
    client = get_embedding_client()
    texts = input if isinstance(input, List) else [input]
    response = await client.ado(texts=texts, model=model, **kwargs)
    response.body['object'] = 'list'
    response.body.setdefault('model', model)
    return CreateEmbeddingResponse(**response.body)
