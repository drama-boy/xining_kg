# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import init_config, logger
from utils.util import (
    as_list,
    clean_text,
    edge_to_link,
    get_neo4j_config,
    import_to_neo4j,
    load_mechanism_events,
    merge_values,
    normalize_node_properties,
    normalize_properties,
    resolve_path,
    write_json,
)


FILES_CONFIG = init_config.get("files", {})
KG_CONFIG = init_config.get("knowledge_graph", {})

DATA_DIR = resolve_path(FILES_CONFIG.get("data_dir"), PROJECT_ROOT / "data")
TEMP_DIR = resolve_path(FILES_CONFIG.get("temp_dir"), DATA_DIR / "temp")
OUTPUT_DIR = resolve_path(FILES_CONFIG.get("output_dir"), DATA_DIR / "output")

DEFAULT_OUTPUT_PREFIX = str(KG_CONFIG.get("output_prefix", "kg"))
DEFAULT_EXTRACTED_PREFIX = str(KG_CONFIG.get("extracted_prefix", "extracted_"))
DEFAULT_SHOW_PROGRESS = bool(KG_CONFIG.get("show_progress", False))
DEFAULT_WRITE_TO_NEO4J = bool(KG_CONFIG.get("write_to_neo4j", True))
DEFAULT_CLEAR_NEO4J = bool(KG_CONFIG.get("clear_neo4j", False))


Node = Dict[str, Any]
Edge = Dict[str, Any]


def make_output_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{DEFAULT_OUTPUT_PREFIX}_{timestamp}.json"


class MechanismKGBuilder:
    """Create a node-property graph for mechanism thinking maps."""

    LABELS = {
        "event": "事件",
        "behavior": "行为",
        "predicate": "述谓结构",
        "time": "时间",
        "place": "地点",
        "person": "人员",
        "organization": "组织机构",
        "equipment": "装备设备",
        "fortification": "工事",
        "vehicle": "车辆",
        "unit": "兵种",
        "risk": "风险等级",
    }

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[Tuple[str, str, str], Edge] = {}
        self.current_source_file = ""

    def build(self, mechanism_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Build the graph from mechanism extraction output."""
        for index, event in enumerate(mechanism_events or [], start=1):
            self.current_source_file = clean_text(event.get("来源文件")) or clean_text(event.get("from_file"))
            self._add_mechanism_event(event, index)

        self.current_source_file = ""

        raw_nodes = sorted(self.nodes.values(), key=lambda item: item["id"])
        raw_edges = sorted(
            self.edges.values(),
            key=lambda item: (item["source"], item["relation"], item["target"]),
        )
        node_id_map = {node["id"]: index for index, node in enumerate(raw_nodes, start=1)}
        nodes = [self._export_node(node, node_id_map[node["id"]]) for node in raw_nodes]
        edges = [
            self._export_edge(edge, node_id_map[edge["source"]], node_id_map[edge["target"]])
            for edge in raw_edges
            if edge["source"] in node_id_map and edge["target"] in node_id_map
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "links": [edge_to_link(edge) for edge in edges],
        }

    def _add_mechanism_event(self, event: Dict[str, Any], index: int) -> None:
        topic = clean_text(event.get("事件主题")) or f"事件{index}"
        behavior_text = clean_text(event.get("核心行为")) or clean_text(event.get("述谓结构")) or topic
        behavior_name = f"{topic}-{behavior_text}"

        event_id = self._node_id("event", topic)
        self.add_node(
            event_id,
            self.LABELS["event"],
            topic,
            {
                "name": topic,
                "event_topic": topic,
                "from_file": self.current_source_file,
            },
        )

        behavior_id = self._node_id("behavior", behavior_name)
        self.add_node(
            behavior_id,
            self.LABELS["behavior"],
            behavior_name,
            {
                "name": behavior_name,
                "event_topic": topic,
                "core_behavior": behavior_text,
                "risk_level": clean_text(event.get("风险等级")),
            },
        )
        self.add_edge(event_id, "包含行为", behavior_id)

        predicate = clean_text(event.get("述谓结构"))
        if predicate:
            predicate_id = self._node_id("predicate", predicate)
            self.add_node(
                predicate_id,
                self.LABELS["predicate"],
                predicate,
                {
                    "name": predicate,
                    "desc": predicate,
                    "event_topic": topic,
                },
            )
            self.add_edge(behavior_id, "具有述谓结构", predicate_id)

        for item in as_list(event.get("时间")):
            time_text = clean_text(item)
            if not time_text:
                continue
            time_id = self._node_id("time", time_text)
            self.add_node(time_id, self.LABELS["time"], time_text, {"name": time_text, "time": time_text})
            self.add_edge(behavior_id, "发生时间", time_id)

        place_id = self._add_place(event.get("地点信息"), topic)
        if place_id:
            self.add_edge(behavior_id, "发生地点", place_id)

        risk = clean_text(event.get("风险等级"))
        if risk:
            risk_id = self._node_id("risk", risk)
            self.add_node(risk_id, self.LABELS["risk"], risk, {"name": risk, "level": risk})
            self.add_edge(behavior_id, "风险等级", risk_id)

        for person in as_list(event.get("人员信息")):
            person_id = self._add_person(person, topic)
            if person_id:
                self.add_edge(behavior_id, "涉及人员", person_id)

        target_system = event.get("目标体系") or {}
        if not isinstance(target_system, dict):
            return

        for fortification in as_list(target_system.get("工事")):
            node_id = self._add_target_node("fortification", fortification, topic, place_id)
            if node_id:
                self.add_edge(behavior_id, "涉及工事", node_id)

        for vehicle in as_list(target_system.get("车辆")):
            node_id = self._add_target_node("vehicle", vehicle, topic, place_id)
            if node_id:
                self.add_edge(behavior_id, "涉及车辆", node_id)

        for unit in as_list(target_system.get("兵种")):
            node_id = self._add_target_node("unit", unit, topic, place_id)
            if node_id:
                self.add_edge(behavior_id, "涉及兵种", node_id)

    def _add_place(self, place: Any, topic: str) -> Optional[str]:
        if isinstance(place, dict):
            name = clean_text(place.get("规范化地名")) or clean_text(place.get("名称"))
            if not name:
                return None
            node_id = self._node_id("place", name)
            self.add_node(
                node_id,
                self.LABELS["place"],
                name,
                {
                    "name": name,
                    "normalized_place": name,
                    "coordinates": clean_text(place.get("经纬度")),
                    "altitude": place.get("海拔"),
                    "event_topic": topic,
                },
            )
            return node_id

        name = clean_text(place)
        if not name:
            return None
        node_id = self._node_id("place", name)
        self.add_node(node_id, self.LABELS["place"], name, {"name": name, "normalized_place": name})
        return node_id

    def _add_person(self, person: Any, topic: str) -> Optional[str]:
        if isinstance(person, dict):
            name = clean_text(person.get("姓名")) or clean_text(person.get("名称"))
            if not name:
                return None
            node_id = self._node_id("person", name)
            self.add_node(
                node_id,
                self.LABELS["person"],
                name,
                {
                    "name": name,
                    "person_name": name,
                    "rank": clean_text(person.get("军衔")),
                    "position": clean_text(person.get("职务")),
                    "event_topic": topic,
                },
            )
            return node_id

        name = clean_text(person)
        if not name:
            return None
        node_id = self._node_id("person", name)
        self.add_node(node_id, self.LABELS["person"], name, {"name": name, "person_name": name})
        return node_id

    def _add_target_node(self, kind: str, target: Any, topic: str, place_id: Optional[str]) -> Optional[str]:
        if not isinstance(target, dict):
            name = clean_text(target)
            if not name:
                return None
            node_id = self._node_id(kind, name)
            self.add_node(node_id, self.LABELS[kind], name, {"name": name, "event_topic": topic})
            return node_id

        target_type = clean_text(target.get("类型"))
        model = clean_text(target.get("型号"))
        name = model if model and model != "未知" else target_type
        if not name:
            return None

        node_id = self._node_id(kind, name)
        properties = {
            "name": name,
            "type": target_type,
            "model": model,
            "quantity": target.get("数量"),
            "event_topic": topic,
        }
        if kind in {"fortification", "vehicle"} and place_id:
            place_node = self.nodes.get(place_id, {})
            properties.setdefault("coordinates", place_node.get("properties", {}).get("coordinates"))
            properties.setdefault("altitude", place_node.get("properties", {}).get("altitude"))

        self.add_node(node_id, self.LABELS[kind], name, properties)
        if place_id:
            self.add_edge(node_id, "位于", place_id)
        return node_id

    def add_node(self, node_id: str, label: str, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        clean_properties = normalize_node_properties(normalize_properties(properties or {}), label, self.current_source_file)
        clean_properties["name"] = clean_text(clean_properties.get("name")) or name
        if node_id not in self.nodes:
            self.nodes[node_id] = {
                "id": node_id,
                "label": label,
                "labels": [label],
                "name": name,
                "properties": clean_properties,
            }
            return

        if label not in self.nodes[node_id].setdefault("labels", [self.nodes[node_id]["label"]]):
            self.nodes[node_id]["labels"].append(label)

        existing = self.nodes[node_id]["properties"]
        for key, value in clean_properties.items():
            if value in ("", None, []):
                continue
            if key not in existing or existing[key] in ("", None, []):
                existing[key] = value
            elif existing[key] != value:
                existing[key] = merge_values(existing[key], value)

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        edge_key = (source, relation, target)
        if edge_key not in self.edges:
            self.edges[edge_key] = {
                "source": source,
                "relation": relation,
                "target": target,
                "properties": normalize_properties(properties or {}),
            }

    @staticmethod
    def _export_node(node: Node, output_id: int) -> Node:
        exported = dict(node)
        exported["id"] = output_id
        return exported

    @staticmethod
    def _export_edge(edge: Edge, source_id: int, target_id: int) -> Edge:
        exported = dict(edge)
        exported["source"] = source_id
        exported["target"] = target_id
        return exported

    @staticmethod
    def _node_id(kind: str, name: str) -> str:
        return clean_text(name)


def build_graph_from_files(
    input_paths: Iterable[Any],
    show_progress: Optional[bool] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mechanism_events: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, Any]] = []
    input_paths = list(input_paths)
    show_progress = DEFAULT_SHOW_PROGRESS if show_progress is None else bool(show_progress)

    iterator = input_paths
    if show_progress and input_paths:
        try:
            from tqdm import tqdm

            iterator = tqdm(input_paths, desc="构建知识图谱", total=len(input_paths))
        except Exception:
            iterator = input_paths

    for input_path in iterator:
        path = Path(input_path)
        try:
            file_mechanism_events, summary = load_mechanism_events(path)
            mechanism_events.extend(file_mechanism_events)
            source_summaries.append(summary)
        except Exception as exc:
            failed_files.append({
                "source_file": str(path),
                "error": str(exc),
            })
            logger.exception("处理输入文件失败: {}", path)

    builder = MechanismKGBuilder()
    graph_data = builder.build(mechanism_events=mechanism_events)
    return graph_data, mechanism_events, source_summaries, failed_files


def build_knowledge_graph(
    input_paths: Iterable[Any],
) -> Dict[str, Any]:
    start_time = time.time()
    input_paths = list(input_paths)
    if not input_paths:
        raise ValueError("请提供至少一个输入文件")

    run_id = uuid.uuid4().hex
    output_path = make_output_file()
    extracted_path = OUTPUT_DIR / f"{DEFAULT_EXTRACTED_PREFIX}{run_id}.json"

    graph_data, mechanism_events, source_summaries, failed_files = build_graph_from_files(
        input_paths,
        show_progress=DEFAULT_SHOW_PROGRESS,
    )
    write_json(extracted_path, mechanism_events)
    write_json(output_path, graph_data)

    write_to_neo4j = DEFAULT_WRITE_TO_NEO4J
    clear_neo4j = DEFAULT_CLEAR_NEO4J
    imported_to_neo4j = False
    if write_to_neo4j:
        neo4j_config = get_neo4j_config()
        import_to_neo4j(
            graph_data,
            url=neo4j_config["url"],
            user=neo4j_config["user"],
            password=neo4j_config["password"],
            clear=clear_neo4j,
        )
        imported_to_neo4j = True

    processing_time_ms = (time.time() - start_time)
    return {
        "output_file": str(output_path),
        "extracted_file": str(extracted_path),
        "events_count": len(mechanism_events),
        "nodes_count": len(graph_data["nodes"]),
        "edges_count": len(graph_data["edges"]),
        "triples_count": len(graph_data["edges"]),
        "graph": graph_data,
        "graph_data": graph_data,
        "nodes": graph_data["nodes"],
        "edges": graph_data["edges"],
        "links": graph_data["links"],
        "processing_time_ms": processing_time_ms,
        "evaluation": {
            "source_files": source_summaries,
            "failed_files": failed_files,
            "files_count": len(input_paths),
            "imported_to_neo4j": imported_to_neo4j,
            "write_to_neo4j": write_to_neo4j,
            "clear_neo4j": clear_neo4j if write_to_neo4j else False,
        },
    }


def kg_wrapper(input_paths: Iterable[Any]) -> Dict[str, Any]:
    try:
        result = build_knowledge_graph(input_paths)
        logger.info("知识图谱构建完成: {}", result)
        return result
    except Exception as exc:
        logger.exception("知识图谱构建失败: {}", exc)
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建机理思维知识图谱")
    parser.add_argument("--input-files", type=Path, nargs="+", required=True, help="csv/txt/pdf/docx/json 输入文件路径")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = build_knowledge_graph(args.input_files)
    logger.info("知识图谱构建完成：{}", result["output_file"])
    logger.info("抽取结果：{}", result["extracted_file"])
    logger.info("事件数量：{}", result["events_count"])
    logger.info("节点数量：{}", result["nodes_count"])
    logger.info("关系数量：{}", result["edges_count"])


if __name__ == "__main__":
    main()
