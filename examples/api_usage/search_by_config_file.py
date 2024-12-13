#!/usr/bin/env python3
# coding=utf-8

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Optional, Any

import pandas as pd

from graphrag.api import query as api
from graphrag.config.load_config import load_config
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.config.resolve_path import resolve_paths
from graphrag.index.create_pipeline_config import create_pipeline_config
from graphrag.storage.factory import create_storage
from graphrag.utils.storage import _load_table_from_storage

ENTITY_NODES_TABLE = 'create_final_nodes'
ENTITY_EMBEDDING_TABLE = 'create_final_entities'
COMMUNITIES_TABLE = 'create_final_communities'
COMMUNITY_REPORT_TABLE = 'create_final_community_reports'
TEXT_UNIT_TABLE = 'create_final_text_units'
RELATIONSHIP_TABLE = 'create_final_relationships'
COVARIATE_TABLE = 'create_final_covariates'

local_search_parquet_list = [
    f'{ENTITY_NODES_TABLE}.parquet',
    f'{ENTITY_EMBEDDING_TABLE}.parquet',
    f'{COMMUNITY_REPORT_TABLE}.parquet',
    f'{TEXT_UNIT_TABLE}.parquet',
    f'{RELATIONSHIP_TABLE}.parquet'
]

global_search_parquet_list = [
    f'{ENTITY_NODES_TABLE}.parquet',
    f'{ENTITY_EMBEDDING_TABLE}.parquet',
    f'{COMMUNITIES_TABLE}.parquet',
    f'{COMMUNITY_REPORT_TABLE}.parquet'
]

optional_parquet_list = [f'{COVARIATE_TABLE}.parquet']


async def local_search(
        query: str,
        root_dir: str,
        config_filepath: Optional[str] = None,
        community_level: Optional[int] = 2,
        response_type: Optional[str] = 'Multiple Paragraphs'
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame]
]:
    """Perform a local search and return the response and context data.

    Parameters
    ----------
    - query (str): The user query to search for
    - root_dir (str): The data project root
    - config_filepath (str): The configuration yaml file to use when running the query
    - community_level (int): The community level to search at
    - response_type (str): The response type to return

    Returns
    -------
    - response
    - context data
    """
    root = Path(root_dir).resolve()
    config = load_config(root, config_filepath)
    resolve_paths(config)

    dataframe_dict = await resolve_parquet_files(
        config=config,
        parquet_list=local_search_parquet_list,
        optional_list=optional_parquet_list
    )
    final_nodes: pd.DataFrame = dataframe_dict[ENTITY_NODES_TABLE]
    final_entities: pd.DataFrame = dataframe_dict[ENTITY_EMBEDDING_TABLE]
    final_community_reports: pd.DataFrame = dataframe_dict[COMMUNITY_REPORT_TABLE]
    final_text_units: pd.DataFrame = dataframe_dict[TEXT_UNIT_TABLE]
    final_relationships: pd.DataFrame = dataframe_dict[RELATIONSHIP_TABLE]
    final_covariates: pd.DataFrame | None = dataframe_dict[COVARIATE_TABLE]

    response, context_data = await api.local_search(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        covariates=final_covariates,
        community_level=community_level,
        response_type=response_type,
        query=query
    )

    return response, context_data


async def local_search_streaming(
        query: str,
        root_dir: str,
        config_filepath: Optional[str] = None,
        community_level: Optional[int] = 2,
        response_type: Optional[str] = 'Multiple Paragraphs'
) -> AsyncGenerator:
    """Perform a local search and return the context data and response via a generator.

    Parameters
    ----------
    - query (str): The user query to search for
    - root_dir (str): The data project root
    - config_filepath (str): The configuration yaml file to use when running the query
    - community_level (int): The community level to search at
    - response_type (str): The response type to return

    Returns
    -------
    - the context data and response via a generator
    """
    root = Path(root_dir).resolve()
    config = load_config(root, config_filepath)
    resolve_paths(config)

    dataframe_dict = await resolve_parquet_files(
        config=config,
        parquet_list=local_search_parquet_list,
        optional_list=optional_parquet_list
    )
    final_nodes: pd.DataFrame = dataframe_dict[ENTITY_NODES_TABLE]
    final_entities: pd.DataFrame = dataframe_dict[ENTITY_EMBEDDING_TABLE]
    final_community_reports: pd.DataFrame = dataframe_dict[COMMUNITY_REPORT_TABLE]
    final_text_units: pd.DataFrame = dataframe_dict[TEXT_UNIT_TABLE]
    final_relationships: pd.DataFrame = dataframe_dict[RELATIONSHIP_TABLE]
    final_covariates: pd.DataFrame | None = dataframe_dict[COVARIATE_TABLE]

    async for stream_chunk in api.local_search_streaming(
            config=config,
            nodes=final_nodes,
            entities=final_entities,
            community_reports=final_community_reports,
            text_units=final_text_units,
            relationships=final_relationships,
            covariates=final_covariates,
            community_level=community_level,
            response_type=response_type,
            query=query
    ):
        yield stream_chunk


async def global_search(
        query: str,
        root_dir: str,
        config_filepath: Optional[str] = None,
        community_level: Optional[int] = 2,
        dynamic_community_selection: bool = False,
        response_type: Optional[str] = 'Multiple Paragraphs'
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame]
]:
    """Perform a global search and return the response and context data.

    Parameters
    ----------
    - query (str): The user query to search for
    - root_dir (str): The data project root
    - config_filepath (str): The configuration yaml file to use when running the query
    - community_level (int): The community level to search at
    - response_type (str): The response type to return

    Returns
    -------
    - response
    - context data
    """
    root = Path(root_dir).resolve()
    config = load_config(root, config_filepath)
    resolve_paths(config)

    dataframe_dict = await resolve_parquet_files(
        config=config,
        parquet_list=global_search_parquet_list,
        optional_list=[]
    )
    final_nodes: pd.DataFrame = dataframe_dict[ENTITY_NODES_TABLE]
    final_entities: pd.DataFrame = dataframe_dict[ENTITY_EMBEDDING_TABLE]
    final_communities: pd.DataFrame = dataframe_dict[COMMUNITIES_TABLE]
    final_community_reports: pd.DataFrame = dataframe_dict[COMMUNITY_REPORT_TABLE]

    response, context_data = await api.global_search(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        community_level=community_level,
        dynamic_community_selection=dynamic_community_selection,
        response_type=response_type,
        query=query
    )

    return response, context_data


async def global_search_streaming(
        query: str,
        root_dir: str,
        config_filepath: Optional[str] = None,
        community_level: Optional[int] = 2,
        dynamic_community_selection: bool = False,
        response_type: Optional[str] = 'Multiple Paragraphs'
) -> AsyncGenerator:
    """Perform a global search and return the context data and response via a generator.

    Parameters
    ----------
    - query (str): The user query to search for
    - root_dir (str): The data project root
    - config_filepath (str): The configuration yaml file to use when running the query
    - community_level (int): The community level to search at
    - response_type (str): The response type to return

    Returns
    -------
    - the context data and response via a generator
    """
    root = Path(root_dir).resolve()
    config = load_config(root, config_filepath)
    resolve_paths(config)

    dataframe_dict = await resolve_parquet_files(
        config=config,
        parquet_list=global_search_parquet_list,
        optional_list=optional_parquet_list
    )
    final_nodes: pd.DataFrame = dataframe_dict[ENTITY_NODES_TABLE]
    final_entities: pd.DataFrame = dataframe_dict[ENTITY_EMBEDDING_TABLE]
    final_communities: pd.DataFrame = dataframe_dict[COMMUNITIES_TABLE]
    final_community_reports: pd.DataFrame = dataframe_dict[COMMUNITY_REPORT_TABLE]

    async for stream_chunk in api.global_search_streaming(
            config=config,
            nodes=final_nodes,
            entities=final_entities,
            communities=final_communities,
            community_reports=final_community_reports,
            community_level=community_level,
            dynamic_community_selection=dynamic_community_selection,
            response_type=response_type,
            query=query
    ):
        yield stream_chunk


async def drift_search(
        query: str,
        root_dir: str,
        config_filepath: Optional[str] = None,
        community_level: Optional[int] = 2
) -> tuple[
    str | dict[str, Any] | list[dict[str, Any]],
    str | list[pd.DataFrame] | dict[str, pd.DataFrame]
]:
    """Perform a local search and return the response and context data.

    Parameters
    ----------
    - query (str): The user query to search for
    - root_dir (str): The data project root
    - config_filepath (str): The configuration yaml file to use when running the query
    - community_level (int): The community level to search at
    - response_type (str): The response type to return

    Returns
    -------
    - response
    - context data
    """
    root = Path(root_dir).resolve()
    config = load_config(root, config_filepath)
    resolve_paths(config)

    dataframe_dict = await resolve_parquet_files(
        config=config,
        parquet_list=local_search_parquet_list,
        optional_list=optional_parquet_list
    )
    final_nodes: pd.DataFrame = dataframe_dict[ENTITY_NODES_TABLE]
    final_entities: pd.DataFrame = dataframe_dict[ENTITY_EMBEDDING_TABLE]
    final_community_reports: pd.DataFrame = dataframe_dict[COMMUNITY_REPORT_TABLE]
    final_text_units: pd.DataFrame = dataframe_dict[TEXT_UNIT_TABLE]
    final_relationships: pd.DataFrame = dataframe_dict[RELATIONSHIP_TABLE]

    response, context_data = await api.drift_search(
        config=config,
        nodes=final_nodes,
        entities=final_entities,
        community_reports=final_community_reports,
        text_units=final_text_units,
        relationships=final_relationships,
        community_level=community_level,
        query=query
    )

    return response, context_data


async def resolve_parquet_files(
        config: GraphRagConfig,
        parquet_list: list[str],
        optional_list: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read parquet files to a dataframe dict."""
    dataframe_dict = {}
    pipeline_config = create_pipeline_config(config)
    storage_obj = create_storage(pipeline_config.storage)
    for parquet_file in parquet_list:
        df_key = parquet_file.split('.')[0]
        df_value = await _load_table_from_storage(name=parquet_file, storage=storage_obj)
        dataframe_dict[df_key] = df_value

    # for optional parquet files, set the dict entry to None instead of erroring out if it does not exist
    if optional_list:
        for optional_file in optional_list:
            file_exists = await storage_obj.has(optional_file)
            df_key = optional_file.split('.')[0]
            if file_exists:
                df_value = await _load_table_from_storage(name=optional_file, storage=storage_obj)
                dataframe_dict[df_key] = df_value
            else:
                dataframe_dict[df_key] = None

    return dataframe_dict


async def local_search_demo():
    query = 'Who is Scrooge, and what are his main relationships?'
    response, context_data = await local_search(query, DEMO_ROOT_DIR)
    print(context_data)
    print(response)


async def local_search_streaming_demo():
    query = 'Who is Scrooge, and what are his main relationships?'
    response, context_data = '', {}
    async for chunk in local_search_streaming(query, DEMO_ROOT_DIR):
        if isinstance(chunk, str):
            response += chunk
        else:
            context_data = chunk
    print(context_data)
    print(response)


async def global_search_demo():
    query = 'What are the top themes in this story?'
    response, context_data = await global_search(query, DEMO_ROOT_DIR)
    print(context_data)
    print(response)


async def global_search_streaming_demo():
    query = 'What are the top themes in this story?'
    response, context_data = '', {}
    async for chunk in global_search_streaming(query, DEMO_ROOT_DIR):
        if isinstance(chunk, str):
            response += chunk
        else:
            context_data = chunk
    print(context_data)
    print(response)


async def drift_search_demo():
    query = 'Who is agent Mercer?'
    response, context_data = await drift_search(query, DEMO_ROOT_DIR)
    print(context_data)
    print(response)


if __name__ == '__main__':
    DEMO_ROOT_DIR = './ragtest'

    # local search
    asyncio.run(local_search_demo())
    # asyncio.run(local_search_streaming_demo())

    # global search
    # asyncio.run(global_search_demo())
    # asyncio.run(global_search_streaming_demo())

    # drift search
    # asyncio.run(drift_search_demo())
