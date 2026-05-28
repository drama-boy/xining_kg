# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from time import time

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import init_config, logger
from utils.myllm import llm_gemma
from utils.util import (
    compact_json_dumps,
    compact_properties_for_prompt,
    fit_prompt_context,
    dedup_list,
    extract_place_name,
    format_mapping,
    format_nested_item,
    format_result,
    format_value_list,
    normalize_name,
    normalize_text,
    parse_date_like,
    parse_json_object,
    summarize_properties,
)

TUILI_CONFIG = init_config.get("tuili", {})
NEO4J_CONFIG = init_config.get("neo4j", {})

DEFAULT_TARGET = str(TUILI_CONFIG.get("default_target", "T-90S"))
DEFAULT_USE_LLM = bool(TUILI_CONFIG.get("use_llm", True))
DEFAULT_NEO4J_URL = str(NEO4J_CONFIG.get("url", "bolt://localhost:7687"))
DEFAULT_NEO4J_USER = str(NEO4J_CONFIG.get("user", "neo4j"))
DEFAULT_NEO4J_PASSWORD = str(NEO4J_CONFIG.get("password", "12345678"))
LLM_CONTEXT_MAX_CHARS = int(TUILI_CONFIG.get("llm_context_max_chars", 18000))
LLM_NEIGHBOR_EDGE_LIMIT = int(TUILI_CONFIG.get("llm_neighbor_edge_limit", 16))
LLM_EVENT_CONTEXT_LIMIT = int(TUILI_CONFIG.get("llm_event_context_limit", 20))
LLM_TIMELINE_LIMIT = int(TUILI_CONFIG.get("llm_timeline_limit", 20))
LLM_RELATION_PATH_LIMIT = int(TUILI_CONFIG.get("llm_relation_path_limit", 20))


HIGH_RISK_KEYWORDS = (
    "坦克",
    "主战坦克",
    "装甲",
    "导弹",
    "火箭炮",
    "反坦克",
    "自行火炮",
    "步兵战车",
    "攻击",
    "打击",
)
MEDIUM_RISK_KEYWORDS = (
    "无人机",
    "侦察",
    "防空",
    "保障",
    "哨所",
    "工事",
    "指挥",
)

ALLY_RELATIONS = {
    "同方协同",
    "协同",
    "合作",
    "联合",
    "数据链互通",
    "引导",
    "支援",
    "保障",
    "装备",
    "拥有",
    "隶属",
    "指挥",
    "使用",
    "部署",
    "部署于",
}
OPPONENT_RELATIONS = {
    "对抗",
    "敌对阵营",
    "伏击",
    "击毁",
    "命中",
    "攻击",
    "打击",
    "拦截",
    "压制",
    "优于",
    "劣于",
}
TARGET_RELATIONS = {
    "提及装备",
    "涉及装备",
    "涉及车辆",
    "涉及兵种",
    "涉及工事",
    "涉及舰艇",
    "涉及航空器",
    "涉及设施",
    "提及人员",
    "提及组织",
    "涉及人员",
    "涉及组织",
}
ASSOCIATION_OBJECT_LABELS = {"车辆", "装备设备", "舰艇", "航空器", "设施", "工事", "兵种"}


@dataclass
class GraphEntity:
    id: str
    label: str
    labels: List[str]
    name: str
    properties: Dict[str, Any]


@dataclass
class GraphEdge:
    source: str
    relation: str
    target: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InferenceResult:
    target: str
    matched_entities: List[Dict[str, Any]]
    risk_level: str
    risk_reason: str
    associations: Dict[str, Any]
    activity_patterns: Dict[str, Any]
    prediction: Dict[str, Any]
    result: Dict[str, Any] = field(default_factory=dict)
    llm_summary: str = ""
    report_sections: List[Dict[str, Any]] = field(default_factory=list)
    report_lines: List[str] = field(default_factory=list)
    report_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target,
            "matched_entities": self.matched_entities,
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "associations": self.associations,
            "activity_patterns": self.activity_patterns,
            "prediction": self.prediction,
            "result": self.result,
            "llm_summary": self.llm_summary,
            "report_sections": self.report_sections,
            "report_lines": self.report_lines,
            "report_text": self.report_text,
        }


class KGInferenceEngine:
    def __init__(self, graph: Dict[str, Any]) -> None:
        self.graph = graph
        self.nodes = [GraphEntity(**node) for node in graph.get("nodes", [])]
        self.edges = [GraphEdge(**edge) for edge in graph.get("edges", [])]
        self.node_by_id = {node.id: node for node in self.nodes}
        self.nodes_by_name = defaultdict(list)
        self.nodes_by_label = defaultdict(list)
        for node in self.nodes:
            self.nodes_by_name[normalize_name(node.name)].append(node)
            self.nodes_by_label[node.label].append(node)

        self.out_edges = defaultdict(list)
        self.in_edges = defaultdict(list)
        for edge in self.edges:
            self.out_edges[edge.source].append(edge)
            self.in_edges[edge.target].append(edge)

    @classmethod
    def from_file(cls, path: Path) -> "KGInferenceEngine":
        with path.open("r", encoding="utf-8") as file:
            graph = json.load(file)
        return cls(graph)

    @classmethod
    def from_neo4j(
        cls,
        url: str = DEFAULT_NEO4J_URL,
        user: str = DEFAULT_NEO4J_USER,
        password: str = DEFAULT_NEO4J_PASSWORD,
    ) -> "KGInferenceEngine":
        from py2neo import Graph

        neo4j_graph = Graph(url, auth=(user, password))
        nodes = cls._load_nodes_from_neo4j(neo4j_graph)
        edges = cls._load_edges_from_neo4j(neo4j_graph)
        return cls({"nodes": nodes, "edges": edges})

    @staticmethod
    def _load_nodes_from_neo4j(neo4j_graph: Any) -> List[Dict[str, Any]]:
        query = """
        MATCH (n)
        RETURN
            coalesce(n.id, toString(id(n))) AS id,
            coalesce(n.name, n.`名称`, toString(id(n))) AS name,
            labels(n) AS labels,
            properties(n) AS properties
        """
        nodes = []
        for record in neo4j_graph.run(query).data():
            labels = record.get("labels") or []
            properties = dict(record.get("properties") or {})
            label = labels[0] if labels else KGInferenceEngine._primary_label_from_properties(properties)
            node_id = str(record.get("id") or record.get("name") or "")
            name = str(record.get("name") or properties.get("名称") or node_id)
            properties.setdefault("id", node_id)
            properties.setdefault("name", name)
            nodes.append({
                "id": node_id,
                "label": label,
                "labels": labels or [label],
                "name": name,
                "properties": properties,
            })
        return nodes

    @staticmethod
    def _load_edges_from_neo4j(neo4j_graph: Any) -> List[Dict[str, Any]]:
        query = """
        MATCH (source)-[r]->(target)
        RETURN
            coalesce(source.id, toString(id(source))) AS source,
            type(r) AS relation,
            coalesce(target.id, toString(id(target))) AS target,
            properties(r) AS properties
        """
        edges = []
        for record in neo4j_graph.run(query).data():
            edges.append({
                "source": str(record.get("source") or ""),
                "relation": str(record.get("relation") or ""),
                "target": str(record.get("target") or ""),
                "properties": dict(record.get("properties") or {}),
            })
        return edges

    @staticmethod
    def _primary_label_from_properties(properties: Dict[str, Any]) -> str:
        category = properties.get("类别") or properties.get("label") or properties.get("labels")
        if isinstance(category, list) and category:
            return str(category[0])
        if category:
            return str(category)
        return "Entity"

    def infer(self, target: str, use_llm: bool = True) -> InferenceResult:
        matched = self.match_entities(target)
        if not matched:
            result = InferenceResult(
                target=target,
                matched_entities=[],
                risk_level="未知",
                risk_reason="未在图谱中找到与输入一致或近似的实体。",
                associations={"team": [], "opponent": [], "events": [], "timeline": []},
                activity_patterns={"frequencies": {}, "locations": []},
                prediction={"next_action": "", "time_window": "", "location": ""},
            )
            self._attach_result_dict(result, use_llm=use_llm)
            self._attach_report(result)
            return result

        primary = matched[0]
        risk_level, risk_reason = self.assess_risk(primary)
        associations = self.analyze_associations(primary)
        activity_patterns = self.analyze_activity_patterns(primary)
        prediction = self.predict_behavior(primary, activity_patterns)

        result = InferenceResult(
            target=target,
            matched_entities=matched,
            risk_level=risk_level,
            risk_reason=risk_reason,
            associations=associations,
            activity_patterns=activity_patterns,
            prediction=prediction,
        )
        self._attach_result_dict(result, use_llm=use_llm)
        self._attach_report(result)
        return result

    def _attach_result_dict(self, result: InferenceResult, use_llm: bool = True) -> None:
        if use_llm:
            result.result = self.build_llm_summary(result)
        else:
            result.result = self.build_result_dict(result)
        result.llm_summary = str(result.result.get("llm_summary", ""))
        result.result["activitypattern"] = normalize_text(result.result.get("activitypattern")) or "图谱信息不足，暂无法判断。"
        result.result["forecast"] = normalize_text(result.result.get("forecast")) or "图谱信息不足，暂无法判断。"

    def build_result_dict(self, result: InferenceResult) -> Dict[str, Any]:
        return {
            "target": result.target,
            "hit_target": self._build_hit_target_items(result.matched_entities),
            "risk_level": result.risk_level,
            "risk_reason": result.risk_reason,
            "timeline": self._build_result_timeline(result.associations.get("timeline", [])),
            "relatedobj": {
                "team": result.associations.get("team", []),
                "opponent": result.associations.get("opponent", []),
            },
            "activitypattern": "",
            "forecast": "",
        }

    @staticmethod
    def _build_hit_target_items(matched_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "name": item.get("name", ""),
                "label": item.get("label", ""),
                "score": item.get("score", 0),
                "properties": item.get("properties", {}),
            }
            for item in matched_entities
        ]

    @staticmethod
    def _build_result_timeline(timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result_timeline = []
        for item in timeline or []:
            event = item.get("event", "")
            behavior = item.get("behavior", "")
            behavior_or_event = KGInferenceEngine._format_behavior_or_event(event, behavior)
            result_timeline.append({
                "时间": item.get("time", ""),
                "地点": item.get("locations", []),
                "行为或事件": behavior_or_event,
            })
        return result_timeline

    @staticmethod
    def _format_behavior_or_event(event: Any, behavior: Any) -> str:
        event_text = normalize_text(event)
        behavior_text = KGInferenceEngine._short_behavior_name(event_text, behavior)
        return event_text or behavior_text

    @staticmethod
    def _short_behavior_name(event: Any, behavior: Any) -> str:
        event_text = normalize_text(event)
        behavior_text = normalize_text(behavior)
        if not behavior_text or behavior_text == event_text:
            return ""
        for separator in ("-", "：", ":", "—"):
            prefix = f"{event_text}{separator}"
            if event_text and behavior_text.startswith(prefix):
                return behavior_text[len(prefix):].strip()
        return behavior_text

    def build_llm_context(self, result: InferenceResult) -> Dict[str, Any]:
        matched_entity = result.matched_entities[0] if result.matched_entities else None
        graph_context = {}
        if matched_entity:
            graph_context = self._collect_neighbor_context(str(matched_entity.get("id", "")))

        context = {
            "target": result.target,
            "matched_entities": [
                {
                    "id": item.get("id", ""),
                    "name": item.get("name", ""),
                    "label": item.get("label", ""),
                    "score": item.get("score", 0),
                    "properties": compact_properties_for_prompt(item.get("properties", {}), max_items=10, max_text_chars=120),
                }
                for item in result.matched_entities[:3]
            ],
            "risk_level": result.risk_level,
            "risk_reason": result.risk_reason,
            "graph_context": graph_context,
            "event_context": [
                self._compact_event_context(item)
                for item in self._collect_event_context_for_result(result)[:LLM_EVENT_CONTEXT_LIMIT]
            ],
            "timeline": result.associations.get("timeline", [])[:LLM_TIMELINE_LIMIT],
            "relation_paths": result.associations.get("relation_paths", [])[:LLM_RELATION_PATH_LIMIT],
            "activity_statistics": {
                "frequencies": dict(
                    sorted(
                        (result.activity_patterns.get("frequencies", {}) or {}).items(),
                        key=lambda item: (-item[1], item[0]),
                    )[:10]
                ),
                "locations": (result.activity_patterns.get("locations", []) or [])[:10],
                "times": (result.activity_patterns.get("times", []) or [])[:10],
            },
            "prediction_seed": result.prediction,
        }
        return fit_prompt_context(context, max_chars=LLM_CONTEXT_MAX_CHARS)

    def _collect_neighbor_context(self, node_id: str) -> Dict[str, Any]:
        node = self.node_by_id.get(node_id)
        if not node:
            return {}

        incoming = []
        outgoing = []
        neighbor_ids = set()
        incoming_edges = self._rank_edges_for_prompt(self.in_edges[node.id], node.id)[:LLM_NEIGHBOR_EDGE_LIMIT]
        outgoing_edges = self._rank_edges_for_prompt(self.out_edges[node.id], node.id)[:LLM_NEIGHBOR_EDGE_LIMIT]
        for edge in incoming_edges:
            source = self.node_by_id.get(edge.source)
            if not source:
                continue
            neighbor_ids.add(source.id)
            incoming.append({
                "from": self._node_prompt_brief(source),
                "relation": edge.relation,
                "to": self._node_prompt_brief(node),
                "properties": compact_properties_for_prompt(edge.properties, max_items=6, max_text_chars=80),
            })

        for edge in outgoing_edges:
            target = self.node_by_id.get(edge.target)
            if not target:
                continue
            neighbor_ids.add(target.id)
            outgoing.append({
                "from": self._node_prompt_brief(node),
                "relation": edge.relation,
                "to": self._node_prompt_brief(target),
                "properties": compact_properties_for_prompt(edge.properties, max_items=6, max_text_chars=80),
            })

        return {
            "target_node": self._node_prompt_brief(node),
            "incoming_edges": incoming,
            "incoming_edges_omitted": max(len(self.in_edges[node.id]) - len(incoming_edges), 0),
            "outgoing_edges": outgoing,
            "outgoing_edges_omitted": max(len(self.out_edges[node.id]) - len(outgoing_edges), 0),
            "neighbor_nodes": [
                self._node_prompt_brief(self.node_by_id[neighbor_id])
                for neighbor_id in sorted(neighbor_ids)
                if neighbor_id in self.node_by_id
            ],
        }

    def _collect_event_context_for_result(self, result: InferenceResult) -> List[Dict[str, Any]]:
        if not result.matched_entities:
            return []
        node = self.node_by_id.get(str(result.matched_entities[0].get("id", "")))
        if not node:
            return []
        return self._collect_event_context(node)

    @staticmethod
    def _node_brief(node: GraphEntity) -> Dict[str, Any]:
        return {
            "id": node.id,
            "label": node.label,
            "labels": node.labels,
            "name": node.name,
            "properties": node.properties,
        }

    @staticmethod
    def _node_prompt_brief(node: GraphEntity) -> Dict[str, Any]:
        return {
            "id": node.id,
            "label": node.label,
            "name": node.name,
            "properties": compact_properties_for_prompt(node.properties, max_items=8, max_text_chars=120),
        }

    @staticmethod
    def _compact_event_context(context: Dict[str, Any]) -> Dict[str, Any]:
        relations = context.get("relations", []) or []
        return {
            "event": context.get("event", ""),
            "behavior": context.get("behavior", ""),
            "relation": context.get("relation", ""),
            "times": (context.get("times", []) or [])[:5],
            "locations": (context.get("locations", []) or [])[:5],
            "targets": (context.get("targets", []) or [])[:8],
            "relations": relations[:5],
            "relations_omitted": max(len(relations) - 5, 0),
        }

    def _rank_edges_for_prompt(self, edges: Sequence[GraphEdge], center_id: str) -> List[GraphEdge]:
        def key(edge: GraphEdge) -> Tuple[int, int, str, str]:
            other_id = edge.source if edge.target == center_id else edge.target
            other = self.node_by_id.get(other_id)
            relation_rank = 0 if edge.relation in TARGET_RELATIONS or edge.relation in ALLY_RELATIONS or edge.relation in OPPONENT_RELATIONS else 1
            label_rank = self._node_priority(other) if other else 9
            path_name = other.name if other else ""
            return (relation_rank, label_rank, edge.relation, path_name)

        return sorted(edges, key=key)

    def _attach_report(self, result: InferenceResult) -> None:
        sections = self.build_report_sections(result)
        lines = self.build_report_lines(sections)
        result.report_sections = sections
        result.report_lines = lines
        result.report_text = "\n".join(lines)

    def build_report_sections(self, result: InferenceResult) -> List[Dict[str, Any]]:
        matched_items = []
        for item in result.matched_entities[:5]:
            props = item.get("properties", {}) or {}
            matched_items.append({
                "name": item.get("name", ""),
                "label": item.get("label", ""),
                "score": item.get("score", 0),
                "summary": summarize_properties(props),
            })

        timeline = result.associations.get("timeline", []) or []
        relation_paths = result.associations.get("relation_paths", []) or []
        locations = result.activity_patterns.get("locations", []) or []
        times = result.activity_patterns.get("times", []) or []
        frequencies = result.activity_patterns.get("frequencies", {}) or {}

        sections: List[Dict[str, Any]] = [
            {
                "title": "目标概览",
                "items": [
                    {"label": "目标名称", "value": result.target},
                    {"label": "命中实体", "value": matched_items},
                ],
            },
            {
                "title": "风险研判",
                "items": [
                    {"label": "风险等级", "value": result.risk_level},
                    {"label": "研判依据", "value": result.risk_reason},
                ],
            },
            {
                "title": "关联分析",
                "items": [
                    {"label": "队友", "value": result.associations.get("team", [])},
                    {"label": "对手", "value": result.associations.get("opponent", [])},
                    {"label": "相关事件", "value": result.associations.get("events", [])},
                    {"label": "相关行为", "value": result.associations.get("behaviors", [])},
                    {"label": "共现目标", "value": result.associations.get("co_targets", [])},
                    {"label": "时序脉络", "value": timeline},
                    {"label": "关联路径", "value": relation_paths},
                ],
            },
            {
                "title": "活动规律",
                "items": [
                    {"label": "行为频次", "value": frequencies},
                    {"label": "高频地点", "value": locations},
                    {"label": "时间分布", "value": times},
                ],
            },
            {
                "title": "行为预测",
                "items": [
                    {"label": "下一步行为", "value": result.prediction.get("next_action", "")},
                    {"label": "时间窗口", "value": result.prediction.get("time_window", "")},
                    {"label": "预测地点", "value": result.prediction.get("location", "")},
                    {"label": "置信度", "value": result.prediction.get("confidence", "")},
                    {"label": "预测依据", "value": result.prediction.get("basis", {})},
                ],
            },
        ]
        if result.llm_summary:
            sections.append({
                "title": "大模型总结",
                "items": [
                    {"label": "总结", "value": result.llm_summary},
                ],
            })
        return sections

    def build_report_lines(self, sections: List[Dict[str, Any]]) -> List[str]:
        lines: List[str] = []
        section_index = 1
        for section in sections:
            lines.append(f"{section_index}. {section['title']}")
            item_index = 1
            for item in section.get("items", []):
                label = item.get("label", "")
                value = item.get("value", "")
                if label == "命中实体":
                    lines.append(f"   {item_index}. {label}：")
                    for idx, entry in enumerate(value or [], start=1):
                        summary = entry.get("summary", "")
                        score = entry.get("score", "")
                        label_text = entry.get("label", "")
                        name = entry.get("name", "")
                        lines.append(f"      - {idx}. {label_text} / {name} / 评分 {score}")
                        if summary:
                            lines.append(f"        {summary}")
                elif label in {"时序脉络", "关联路径"}:
                    lines.append(f"   {item_index}. {label}：")
                    for idx, entry in enumerate(value or [], start=1):
                        lines.append(f"      - {idx}. {format_nested_item(entry)}")
                elif label in {"队友", "对手", "相关事件", "相关行为", "共现目标"}:
                    lines.append(f"   {item_index}. {label}：{format_value_list(value)}")
                elif label in {"行为频次"}:
                    lines.append(f"   {item_index}. {label}：{format_mapping(value)}")
                elif label in {"高频地点", "时间分布"}:
                    lines.append(f"   {item_index}. {label}：")
                    for idx, entry in enumerate(value or [], start=1):
                        lines.append(f"      - {idx}. {format_nested_item(entry)}")
                elif label == "预测依据":
                    lines.append(f"   {item_index}. {label}：")
                    lines.append(f"      - {format_nested_item(value)}")
                else:
                    lines.append(f"   {item_index}. {label}：{format_nested_item(value)}")
                item_index += 1
            section_index += 1
        return lines

    def match_entities(self, target: str) -> List[Dict[str, Any]]:
        norm = normalize_name(target)
        candidates: List[Tuple[int, GraphEntity]] = []

        for node in self.nodes:
            score = self._match_score(norm, node)
            if score > 0:
                candidates.append((score, node))

        candidates.sort(key=lambda item: (-item[0], self._node_priority(item[1]), item[1].name))
        seen = set()
        output: List[Dict[str, Any]] = []
        for score, node in candidates:
            if node.id in seen:
                continue
            seen.add(node.id)
            output.append({
                "score": score,
                "id": node.id,
                "label": node.label,
                "name": node.name,
                "properties": node.properties,
            })
            if len(output) >= 5:
                break
        return output

    def assess_risk(self, entity: Dict[str, Any]) -> Tuple[str, str]:
        properties = entity.get("properties", {})
        label = entity.get("label", "")
        name = entity.get("name", "")
        text = " ".join([str(label), str(name), json.dumps(properties, ensure_ascii=False)])

        if any(keyword in text for keyword in HIGH_RISK_KEYWORDS):
            return "高危", "实体属性命中攻击性或打击性关键词，按逻辑链判为高危。"
        if any(keyword in text for keyword in MEDIUM_RISK_KEYWORDS):
            return "中危", "实体属性命中侦察、保障或防空相关关键词，按逻辑链判为中危。"

        target_type = normalize_text(properties.get("类型"))
        if target_type and any(keyword in target_type for keyword in ("坦克", "导弹", "火箭炮", "步兵战车")):
            return "高危", "类型属性显示为作战平台，按逻辑链判为高危。"
        if target_type and any(keyword in target_type for keyword in ("无人机", "侦察", "防空", "保障")):
            return "中危", "类型属性显示为侦察或保障平台，按逻辑链判为中危。"

        return "低危", "未命中高危或中危特征，默认判为低危。"

    def analyze_associations(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        node = self.node_by_id.get(entity["id"])
        if not node:
            return {"team": [], "opponent": [], "events": [], "timeline": [], "co_targets": []}

        related_events = []
        related_behaviors = []
        related_targets = []
        relation_paths = []

        for edge in self.in_edges[node.id]:
            source_node = self.node_by_id.get(edge.source)
            if not source_node:
                continue
            relation_paths.append({
                "from": source_node.name,
                "relation": edge.relation,
                "to": node.name,
                "source_label": source_node.label,
            })
            if source_node.label == "事件":
                related_events.append(source_node.name)
            if source_node.label == "行为":
                related_behaviors.append(source_node.name)

        for edge in self.out_edges[node.id]:
            target_node = self.node_by_id.get(edge.target)
            if not target_node:
                continue
            relation_paths.append({
                "from": node.name,
                "relation": edge.relation,
                "to": target_node.name,
                "target_label": target_node.label,
            })
            if edge.relation == "位于":
                related_targets.append(target_node.name)

        event_context = self._collect_event_context(node)
        timeline = self._build_timeline(event_context)
        teammates, opponents = self._split_allies_opponents(node, event_context)

        return {
            "team": teammates,
            "opponent": opponents,
            "events": dedup_list(related_events),
            "behaviors": dedup_list(related_behaviors),
            "co_targets": dedup_list(related_targets),
            "timeline": timeline,
            "relation_paths": relation_paths[:20],
        }

    def analyze_activity_patterns(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        node = self.node_by_id.get(entity["id"])
        if not node:
            return {"frequencies": {}, "locations": [], "times": []}

        contexts = self._collect_event_context(node)
        freq_counter = Counter()
        location_counter = Counter()
        time_counter = Counter()

        for ctx in contexts:
            for relation in ctx["relations"]:
                freq_counter[relation["behavior"]] += 1
            for loc in ctx["locations"]:
                location_counter[loc] += 1
            for time_text in ctx["times"]:
                time_counter[time_text] += 1

        return {
            "frequencies": dict(freq_counter),
            "locations": [{"name": name, "count": count} for name, count in location_counter.most_common()],
            "times": [{"name": name, "count": count} for name, count in time_counter.most_common()],
        }

    def predict_behavior(self, entity: Dict[str, Any], activity_patterns: Dict[str, Any]) -> Dict[str, Any]:
        node = self.node_by_id.get(entity["id"])
        if not node:
            return {"next_action": "", "time_window": "", "location": "", "confidence": 0.0}

        event_contexts = self._collect_event_context(node)
        location_counter = Counter()
        time_values: List[datetime] = []
        for ctx in event_contexts:
            for loc in ctx["locations"]:
                location_counter[loc] += 1
            for time_text in ctx["times"]:
                parsed = parse_date_like(time_text)
                if parsed:
                    time_values.append(parsed)

        predicted_location = ""
        if location_counter:
            predicted_location = location_counter.most_common(1)[0][0]

        next_action = self._predict_action_label(node, event_contexts)
        time_window = self._predict_time_window(time_values)
        confidence = 0.55
        if predicted_location:
            confidence += 0.1
        if next_action:
            confidence += 0.1
        if time_window:
            confidence += 0.1

        return {
            "next_action": next_action,
            "time_window": time_window,
            "location": predicted_location,
            "confidence": round(min(confidence, 0.95), 2),
            "basis": {
                "recent_locations": activity_patterns.get("locations", [])[:3],
                "recent_frequencies": sorted(activity_patterns.get("frequencies", {}).items(), key=lambda item: (-item[1], item[0]))[:5],
            },
        }

    def build_llm_summary(self, result: InferenceResult) -> Dict[str, Any]:
        result_dict = self.build_result_dict(result)
        if llm_gemma is None:
            return result_dict
        llm_input = self.build_llm_context(result)
        prompt = (
            "你是一个军事知识图谱推理助手。下面给出的是目标实体在Neo4j知识图谱中的相关节点、入边、出边、关联路径、事件上下文和统计信息。\n"
            "请根据这些图谱信息生成两个中文自然语言字段，不要使用固定模板照抄，也不要输出分析过程。\n"
            "要求：\n"
            "1. 只输出JSON对象，不要输出解释、Markdown或代码块。\n"
            "2. JSON必须只有两个键：activitypattern 和 forecast。\n"
            "3. activitypattern 的值必须是一段话，概括目标实体在图谱中的时间范围、地点范围、活动类型、次数、关联对象等规律。\n"
            "4. forecast 的值必须是一段话，基于图谱中的最近时间、地点、行为、关联装备和prediction_seed推断下一步可能行动。\n"
            "5. 两个键的值都必须是字符串，不能是字典或列表。\n"
            "6. 不要编造与图谱明显无关的目标、地点、时间；如果图谱信息不足，请写“图谱信息不足，暂无法判断”。\n\n"
            f"图谱上下文：{compact_json_dumps(llm_input)}"
        )
        try:
            llm_text = (llm_gemma(prompt) or "").strip()
            llm_fields = parse_json_object(llm_text)
            huodonguilv = normalize_text(llm_fields.get("activitypattern"))
            forecast = normalize_text(llm_fields.get("forecast"))
            if huodonguilv:
                result_dict["activitypattern"] = huodonguilv
            if forecast:
                result_dict["forecast"] = forecast
            result_dict["llm_summary"] = llm_text
        except Exception as exc:
            logger.warning("接口二LLM摘要生成失败，使用规则结果兜底: {}", exc)
            result_dict["llm_summary"] = ""
        if not normalize_text(result_dict.get("activitypattern")):
            result_dict["activitypattern"] = self._fallback_activitypattern(result)
        if not normalize_text(result_dict.get("forecast")):
            result_dict["forecast"] = self._fallback_forecast(result)
        return result_dict

    @staticmethod
    def _fallback_activitypattern(result: InferenceResult) -> str:
        timeline = result.associations.get("timeline", []) or []
        locations = result.activity_patterns.get("locations", []) or []
        frequencies = result.activity_patterns.get("frequencies", {}) or {}
        times = [item.get("time", "") for item in timeline if item.get("time")]
        location_names = [item.get("name", "") for item in locations[:3] if item.get("name")]
        top_behaviors = [name for name, _ in sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[:3]]

        parts = []
        if times:
            parts.append(f"时间范围集中在{times[0]}至{times[-1]}")
        if location_names:
            parts.append(f"高频地点包括{'、'.join(location_names)}")
        if top_behaviors:
            parts.append(f"主要活动类型为{'、'.join(top_behaviors)}")
        if result.associations.get("team"):
            parts.append(f"关联队友包括{'、'.join(result.associations.get('team', [])[:3])}")
        if result.associations.get("opponent"):
            parts.append(f"关联对手包括{'、'.join(result.associations.get('opponent', [])[:3])}")
        if not parts:
            return "图谱信息不足，暂无法判断。"
        return f"{result.target}在图谱中的活动规律显示，" + "，".join(parts) + "。"

    @staticmethod
    def _fallback_forecast(result: InferenceResult) -> str:
        prediction = result.prediction or {}
        next_action = normalize_text(prediction.get("next_action"))
        time_window = normalize_text(prediction.get("time_window"))
        location = normalize_text(prediction.get("location"))
        basis = []
        if next_action:
            basis.append(f"下一步可能{next_action}")
        if time_window:
            basis.append(f"时间窗口为{time_window}")
        if location:
            basis.append(f"重点位置为{location}")
        if not basis:
            return "图谱信息不足，暂无法判断。"
        return f"基于既有关联事件、地点频次和行为统计，{result.target}" + "，".join(basis) + "。"

    def _match_score(self, norm_target: str, node: GraphEntity) -> int:
        name = normalize_name(node.name)
        props_text = normalize_name(" ".join(str(v) for v in node.properties.values()))
        score = 0
        if norm_target == name:
            score += 100
        if norm_target and norm_target in name:
            score += 80
        if name and name in norm_target:
            score += 70
        if norm_target and norm_target in props_text:
            score += 30
        if self._is_vehicle_like(node) and self._normalize_vehicle_alias(norm_target) == self._normalize_vehicle_alias(name):
            score += 40
        if self._normalize_vehicle_alias(norm_target) and self._normalize_vehicle_alias(norm_target) == self._normalize_vehicle_alias(props_text):
            score += 20
        return score

    @staticmethod
    def _node_priority(node: GraphEntity) -> int:
        order = {
            "车辆": 0,
            "装备设备": 1,
            "舰艇": 2,
            "航空器": 3,
            "兵种": 4,
            "组织机构": 5,
            "人员": 6,
            "行为": 7,
            "事件": 8,
            "地点": 9,
            "时间": 10,
        }
        return order.get(node.label, 9)

    @staticmethod
    def _is_vehicle_like(node: GraphEntity) -> bool:
        text = " ".join([node.label, node.name, json.dumps(node.properties, ensure_ascii=False)])
        return any(
            keyword in text
            for keyword in ("坦克", "装甲", "无人机", "火箭炮", "战车", "导弹", "舰", "艇", "防空", "火炮", "雷达")
        )

    @staticmethod
    def _normalize_vehicle_alias(text: str) -> str:
        value = normalize_name(text)
        value = re.sub(r"[“”\"'‘’]", "", value)
        value = re.sub(r"(坦克|无人机|战车|导弹|火箭炮|系统|装备|设备)$", "", value)
        value = value.replace("型号", "")
        return value

    def _collect_event_context(self, node: GraphEntity) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        visited_events = set()

        def add_context(event_node: GraphEntity, behavior_node: Optional[GraphEntity], via_relation: str) -> None:
            event_id = event_node.id
            if event_id in visited_events:
                return
            visited_events.add(event_id)

            times = []
            locations = []
            relations = []
            targets = []

            for edge in self.out_edges[event_node.id]:
                target = self.node_by_id.get(edge.target)
                if not target:
                    continue
                if edge.relation in {"提及时间", "发生时间"} or target.label == "时间":
                    times.append(target.name)
                elif edge.relation in {"提及地点", "发生地点"} or target.label == "地点":
                    locations.append(target.name)
                elif edge.relation in TARGET_RELATIONS:
                    targets.append(target.name)
                elif edge.relation == "包含行为" and target.label == "行为":
                    relations.append({
                        "event": event_node.name,
                        "behavior": self._short_behavior_name(event_node.name, target.name) or target.name,
                        "via": edge.relation,
                    })

            if behavior_node:
                for edge in self.out_edges[behavior_node.id]:
                    target = self.node_by_id.get(edge.target)
                    if not target:
                        continue
                    if edge.relation == "发生地点":
                        loc = extract_place_name(target)
                        if loc:
                            locations.append(loc)
                    elif edge.relation == "发生时间":
                        times.append(target.name)
                    elif edge.relation.startswith("涉及"):
                        targets.append(target.name)

            contexts.append({
                "event": event_node.name,
                "behavior": self._short_behavior_name(event_node.name, behavior_node.name) if behavior_node else "",
                "relation": via_relation,
                "times": dedup_list(times),
                "locations": dedup_list(locations),
                "targets": dedup_list(targets),
                "relations": relations,
            })

        # Direct event relations
        for edge in self.in_edges[node.id]:
            source = self.node_by_id.get(edge.source)
            if not source:
                continue
            if source.label == "事件":
                behavior_node = self._find_behavior_for_event(source)
                add_context(source, behavior_node, edge.relation)
            elif source.label == "行为":
                parent_event = self._find_event_for_behavior(source)
                if parent_event:
                    add_context(parent_event, source, edge.relation)

        # If node itself is an event/behavior, add context around it.
        if node.label == "事件":
            behavior_node = self._find_behavior_for_event(node)
            add_context(node, behavior_node, "self")
        elif node.label == "行为":
            parent_event = self._find_event_for_behavior(node)
            if parent_event:
                add_context(parent_event, node, "self")

        # When the target is a vehicle/equipment, pull its event context.
        if node.label in {"车辆", "装备设备", "舰艇", "航空器", "设施", "人员", "组织机构", "兵种", "工事", "地点"}:
            for edge in self.in_edges[node.id]:
                source = self.node_by_id.get(edge.source)
                if source and source.label in {"事件", "行为"}:
                    event_node = source if source.label == "事件" else self._find_event_for_behavior(source)
                    behavior_node = self._find_behavior_for_event(event_node) if event_node else None
                    if event_node:
                        add_context(event_node, behavior_node or source if source.label == "行为" else None, edge.relation)

        return contexts

    def _find_behavior_for_event(self, event_node: GraphEntity) -> Optional[GraphEntity]:
        for edge in self.out_edges[event_node.id]:
            target = self.node_by_id.get(edge.target)
            if target and target.label == "行为":
                return target
        return None

    def _find_event_for_behavior(self, behavior_node: GraphEntity) -> Optional[GraphEntity]:
        for edge in self.in_edges[behavior_node.id]:
            source = self.node_by_id.get(edge.source)
            if source and source.label == "事件":
                return source
        return None

    def _build_timeline(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        items: List[Tuple[Optional[datetime], Dict[str, Any]]] = []
        for ctx in contexts:
            for time_text in ctx.get("times", []):
                parsed = parse_date_like(time_text)
                items.append((parsed, {
                    "time": time_text,
                    "event": ctx.get("event", ""),
                    "behavior": ctx.get("behavior", ""),
                    "locations": ctx.get("locations", []),
                    "targets": ctx.get("targets", []),
                }))
        items.sort(key=lambda pair: (pair[0] is None, pair[0] or datetime.max, pair[1]["time"]))
        seen = set()
        timeline = []
        for _, item in items:
            key = (item["time"], item["event"], item["behavior"])
            if key in seen:
                continue
            seen.add(key)
            timeline.append(item)
        return timeline

    def _split_allies_opponents(self, node: GraphEntity, contexts: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        allies = []
        opponents = []
        scope_nodes = [node] + self._related_variant_nodes(node)
        scope_ids = {scope_node.id for scope_node in scope_nodes}

        for scope_node in scope_nodes:
            for edge in self.in_edges[scope_node.id]:
                source = self.node_by_id.get(edge.source)
                if not source or source.id in scope_ids:
                    continue
                if not self._is_association_object(source, edge.relation):
                    continue
                relation_class = self._classify_relation(node, source, edge.relation)
                if relation_class == "team":
                    allies.append(source.name)
                elif relation_class == "opponent":
                    opponents.append(source.name)

            for edge in self.out_edges[scope_node.id]:
                target = self.node_by_id.get(edge.target)
                if not target or target.id in scope_ids:
                    continue
                if not self._is_association_object(target, edge.relation):
                    continue
                relation_class = self._classify_relation(node, target, edge.relation)
                if relation_class == "team":
                    allies.append(target.name)
                elif relation_class == "opponent":
                    opponents.append(target.name)

        expanded_contexts = list(contexts or [])
        for variant_node in scope_nodes[1:]:
            expanded_contexts.extend(self._collect_event_context(variant_node))

        for ctx in expanded_contexts:
            for target in ctx.get("targets", []):
                target_node = self._find_node_by_name(target)
                if target_node and target_node.id in scope_ids:
                    continue
                if not target_node or not self._is_association_object(target_node):
                    continue
                relation_class = self._classify_side_relation(node, target_node, target)
                if relation_class == "team":
                    allies.append(target)
                elif relation_class == "opponent":
                    opponents.append(target)

        node_names = {normalize_name(scope_node.name) for scope_node in scope_nodes}
        return (
            [item for item in dedup_list(allies) if normalize_name(item) not in node_names],
            [item for item in dedup_list(opponents) if normalize_name(item) not in node_names],
        )

    def _related_variant_nodes(self, node: GraphEntity) -> List[GraphEntity]:
        family_key = self._node_family_key(node)
        if not family_key:
            return []
        variants = []
        for candidate in self.nodes:
            if candidate.id == node.id or not self._is_vehicle_like(candidate):
                continue
            if self._node_family_key(candidate) == family_key:
                variants.append(candidate)
        variants.sort(key=lambda item: (self._node_priority(item), item.name))
        return variants

    def _node_family_key(self, node: GraphEntity) -> str:
        candidates = [
            node.name,
            node.properties.get("name", ""),
            node.properties.get("model", ""),
        ]
        for candidate in candidates:
            family_key = self._vehicle_family_key(candidate)
            if family_key:
                return family_key
        return ""

    @staticmethod
    def _vehicle_family_key(text: Any) -> str:
        alias = KGInferenceEngine._normalize_vehicle_alias(normalize_text(text))
        compact = re.sub(r"[^0-9a-z]+", "", alias.lower())
        if not compact:
            return ""
        for pattern in (r"(t\d{2,3})[a-z]*$", r"(vt\d{1,3})[a-z]*$", r"(ztz\d{2,3})[a-z]*$"):
            match = re.match(pattern, compact)
            if match:
                return match.group(1)
        return ""

    def _classify_relation(self, node: GraphEntity, other: GraphEntity, relation: str) -> str:
        if other.id == node.id:
            return ""
        if relation in OPPONENT_RELATIONS:
            return "opponent"
        if relation in ALLY_RELATIONS:
            return "team"
        return self._classify_side_relation(node, other, other.name)

    def _classify_side_relation(self, node: GraphEntity, other: Optional[GraphEntity], fallback_name: str = "") -> str:
        node_side = self._node_side(node)
        other_side = self._node_side(other) if other else self._side_from_text(fallback_name, allow_unknown=False)
        if node_side and other_side:
            if node_side == other_side:
                return "team"
            return "opponent"
        return ""

    def _is_association_object(self, node: Optional[GraphEntity], relation: str = "") -> bool:
        if not node:
            return False
        if node.label in {"地点", "时间", "风险等级", "事件", "行为", "组织机构", "人员"}:
            return False
        if node.label in ASSOCIATION_OBJECT_LABELS:
            return True
        if node.label == "实体":
            if self._side_from_text(node.name, allow_unknown=False):
                return False
            return relation in ALLY_RELATIONS or relation in OPPONENT_RELATIONS or self._is_vehicle_like(node)
        return self._is_vehicle_like(node)

    def _find_node_by_name(self, name: str) -> Optional[GraphEntity]:
        norm = normalize_name(name)
        if not norm:
            return None
        exact = self.nodes_by_name.get(norm)
        if exact:
            return exact[0]
        alias = self._normalize_vehicle_alias(norm)
        for node in self.nodes:
            names = [
                node.name,
                node.properties.get("name", ""),
                node.properties.get("model", ""),
                node.properties.get("type", ""),
                node.properties.get("person_name", ""),
            ]
            for candidate in names:
                candidate_norm = normalize_name(candidate)
                if not candidate_norm:
                    continue
                candidate_alias = self._normalize_vehicle_alias(candidate_norm)
                if norm == candidate_norm or (alias and alias == candidate_alias):
                    return node
                if len(norm) >= 4 and (norm in candidate_norm or candidate_norm in norm):
                    return node
        return None

    def _node_side(self, node: Optional[GraphEntity]) -> str:
        if not node:
            return ""
        properties = node.properties or {}
        for key in ("side", "所属方", "阵营"):
            side = self._side_from_text(properties.get(key), allow_unknown=True)
            if side:
                return side
        if node.label == "组织机构":
            side = self._side_from_text(node.name, allow_unknown=False)
            if side:
                return side
        organization = normalize_text(properties.get("organization") or properties.get("所属组织"))
        return self._side_from_text(organization, allow_unknown=False)

    @staticmethod
    def _side_from_text(value: Any, allow_unknown: bool = False) -> str:
        text = normalize_name(value)
        if not text:
            return ""
        if any(keyword in text for keyword in ("中方", "中国", "解放军", "中国人民解放军")):
            return "中方"
        if any(keyword in text for keyword in ("印军", "印度", "印度陆军")):
            return "印军"
        if any(keyword in text for keyword in ("巴军", "巴基斯坦", "巴基斯坦陆军")):
            return "巴军"
        return text if allow_unknown else ""

    def _predict_action_label(self, node: GraphEntity, contexts: List[Dict[str, Any]]) -> str:
        all_text = " ".join(
            [node.name]
            + [ctx.get("event", "") for ctx in contexts]
            + [t for ctx in contexts for t in ctx.get("targets", [])]
        )
        if any(keyword in all_text for keyword in ("无人机", "侦察", "目标识别")):
            return "持续侦察与目标识别"
        if any(keyword in all_text for keyword in ("坦克", "装甲", "伏击", "打击")):
            return "机动或火力对抗"
        if any(keyword in all_text for keyword in ("演习", "联训", "部署")):
            return "继续演习部署"
        return "保持监视和跟踪"

    def _predict_time_window(self, times: List[datetime]) -> str:
        if not times:
            return "未能从图谱中提取明确时间窗口"
        latest = max(times)
        next_time = latest + timedelta(days=30)
        return f"{next_time.year}年{next_time.month}月下旬附近"


def inference_wrapper(
    target: str = DEFAULT_TARGET,
    neo4j_url: str = DEFAULT_NEO4J_URL,
    neo4j_user: str = DEFAULT_NEO4J_USER,
    neo4j_password: str = DEFAULT_NEO4J_PASSWORD,
) -> Dict[str, Any]:
    t1 = time()
    engine = KGInferenceEngine.from_neo4j(
        url=neo4j_url,
        user=neo4j_user,
        password=neo4j_password,
    )
    logger.info("开始进行关联分析，输入: {}", target)
    resultall = engine.infer(target, use_llm=DEFAULT_USE_LLM)
    t2 = time()
    resultall.result["processing_time_s"] = t2 - t1
    logger.info("任务完成，耗时={}秒", resultall.result.get("processing_time_s"))
    return resultall.result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于知识图谱的推理分析")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="输入目标名称，例如 T-90S")
    return parser



def main() -> None:
    args = build_arg_parser().parse_args()
    engine = KGInferenceEngine.from_neo4j(
        url=DEFAULT_NEO4J_URL,
        user=DEFAULT_NEO4J_USER,
        password=DEFAULT_NEO4J_PASSWORD,
    )
    resultall = engine.infer(args.target, use_llm=DEFAULT_USE_LLM)
    logger.info("{}", format_result(resultall))


if __name__ == "__main__":
    main()
