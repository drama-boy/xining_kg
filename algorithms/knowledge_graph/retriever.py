# -*- coding: utf-8 -*-
from __future__ import annotations

from py2neo import Graph

from config import logger
from utils.util import get_neo4j_config, search_knowledge


NEO4J_CONFIG = get_neo4j_config()
DEFAULT_NEO4J_URL = NEO4J_CONFIG["url"]
DEFAULT_NEO4J_USER = NEO4J_CONFIG["user"]
DEFAULT_NEO4J_PASSWORD = NEO4J_CONFIG["password"]


def retrieve_wrapper(query: str):
    logger.info("开始搜索数据库，输入: {}", query)
    graph = Graph(DEFAULT_NEO4J_URL, auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD))
    result = search_knowledge(graph, query)

    if not result or not result.get("entity_info"):
        logger.error("搜索数据库失败: {}", query)
        return None

    logger.info("任务完成，查询目标={}, 耗时={}", result["target_entity"], result.get("processing_time_ms"))
    return {
        "target_entity": result["target_entity"],
        "entity_info": result.get("entity_info", ""),
        "relationships": result.get("relationships", ""),
        "processing_time_ms": result.get("processing_time_ms", ""),
    }


if __name__ == "__main__":
    target = "T-90S"
    graph = Graph(DEFAULT_NEO4J_URL, auth=(DEFAULT_NEO4J_USER, DEFAULT_NEO4J_PASSWORD))
    logger.info("{}", search_knowledge(graph, target))
