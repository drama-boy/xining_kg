# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import lru_cache

from py2neo import Graph

from config import logger
from utils.util import get_neo4j_config, search_knowledge


NEO4J_CONFIG = get_neo4j_config()
DEFAULT_NEO4J_URL = NEO4J_CONFIG["url"]
DEFAULT_NEO4J_USER = NEO4J_CONFIG["user"]
DEFAULT_NEO4J_PASSWORD = NEO4J_CONFIG["password"]


def retrieve_wrapper(query: str):
    """旧版检索包装器。"""
    logger.info("开始搜索数据库，输入: {}", query)
    graph = Graph(DEFAULT_NEO4J_URL, auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD))
    result = search_knowledge(graph, query)

    if not result or not result.get("entity_info"):
        logger.error("搜索数据库失败: {}", query)
        return None

    logger.info("任务完成，查询目标={}, 耗时={}秒", result["target_entity"], result.get("processing_time_s"))
    return {
        "target_entity": result["target_entity"],
        "entity_info": result.get("entity_info", ""),
        "relationships": result.get("relationships", ""),
        "processing_time_s": result.get("processing_time_s", ""),
    }


@lru_cache(maxsize=1)
def get_graph() -> Graph:
    """缓存Neo4j连接。"""
    return Graph(DEFAULT_NEO4J_URL, auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD))


@lru_cache(maxsize=256)
def _cached_search(query: str):
    """缓存高频检索结果。"""
    return search_knowledge(get_graph(), query)


def clear_retrieve_cache() -> None:
    """清空检索相关缓存。"""
    _cached_search.cache_clear()
    get_graph.cache_clear()


def retrieve_wrapper(query: str):
    """检索接口统一入口。"""
    query = (query or "").strip()
    logger.info("开始搜索数据库，输入: {}", query)
    result = _cached_search(query)

    if not result or not result.get("entity_info"):
        logger.error("搜索数据库失败: {}", query)
        return None

    logger.info("任务完成，查询目标={}, 耗时={}秒", result["target_entity"], result.get("processing_time_s"))
    return {
        "target_entity": result["target_entity"],
        "entity_info": result.get("entity_info", ""),
        "relationships": result.get("relationships", ""),
        "processing_time_s": result.get("processing_time_s", ""),
    }


if __name__ == "__main__":
    target = "T-90S"
    graph = Graph(DEFAULT_NEO4J_URL, auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD))
    logger.info("{}", search_knowledge(graph, target))
