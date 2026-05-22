# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent
DEFAULT_EVENT_ENTITY_FILE = ROOT / "data" / "out-5.20.1.json"
DEFAULT_MECHANISM_FILE = ROOT / "data" / "out-5.20.2.json"
DEFAULT_OUTPUT_FILE = ROOT / "data" / "kg-5.20.json"


Node = Dict[str, Any]
Edge = Dict[str, Any]
EventHandler = Callable[[Dict[str, Any], int], None]


@dataclass(frozen=True)
class ExtractionSource:
    """One extraction result and the method that knows how to add it to the graph."""

    name: str
    events: Iterable[Dict[str, Any]]
    handler: EventHandler


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

    def build(
        self,
        sources: Optional[Iterable[ExtractionSource]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build the graph from configured extraction outputs."""
        for source in sources or []:
            self._add_records(source.events, source.handler)

        return {
            "nodes": sorted(self.nodes.values(), key=lambda item: item["id"]),
            "edges": sorted(
                self.edges.values(),
                key=lambda item: (item["source"], item["relation"], item["target"]),
            ),
        }

    def _add_records(self, events: Iterable[Dict[str, Any]], handler: EventHandler) -> None:
        for index, event in enumerate(events or [], start=1):
            handler(event, index)

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
                "名称": topic,
                "事件主题": topic,
                "来源文件": "out-5.20.2.json",
            },
        )

        behavior_id = self._node_id("behavior", behavior_name)
        self.add_node(
            behavior_id,
            self.LABELS["behavior"],
            behavior_name,
            {
                "名称": behavior_name,
                "事件主题": topic,
                "核心行为": behavior_text,
                "风险等级": clean_text(event.get("风险等级")),
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
                    "名称": predicate,
                    "内容": predicate,
                    "事件主题": topic,
                },
            )
        self.add_edge(behavior_id, "具有述谓结构", predicate_id)

        for item in as_list(event.get("时间")):
            time_text = clean_text(item)
            if not time_text:
                continue
            time_id = self._node_id("time", time_text)
            self.add_node(time_id, self.LABELS["time"], time_text, {"名称": time_text, "时间": time_text})
            self.add_edge(behavior_id, "发生时间", time_id)

        place_id = self._add_place(event.get("地点信息"), topic)
        if place_id:
            self.add_edge(behavior_id, "发生地点", place_id)

        risk = clean_text(event.get("风险等级"))
        if risk:
            risk_id = self._node_id("risk", risk)
            self.add_node(risk_id, self.LABELS["risk"], risk, {"名称": risk, "等级": risk})
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

    def _add_event_entity_summary(self, event: Dict[str, Any], index: int) -> None:
        topic = clean_text(event.get("事件主题")) or f"实体抽取事件{index}"
        event_id = self._node_id("event", topic)
        self.add_node(
            event_id,
            self.LABELS["event"],
            topic,
            {
                "名称": topic,
                "事件主题": topic,
                "来源文件": "out-5.20.1.json",
            },
        )

        for item in as_list(event.get("时间")):
            time_text = clean_text(item)
            if time_text:
                time_id = self._node_id("time", time_text)
                self.add_node(time_id, self.LABELS["time"], time_text, {"名称": time_text, "时间": time_text})
                self.add_edge(event_id, "提及时间", time_id)

        for place_name in as_list(event.get("地名")):
            place_text = clean_text(place_name)
            if place_text:
                place_id = self._node_id("place", place_text)
                self.add_node(place_id, self.LABELS["place"], place_text, {"名称": place_text, "规范化地名": place_text})
                self.add_edge(event_id, "提及地点", place_id)

        for person_name in as_list(event.get("人名")):
            person_text = clean_text(person_name)
            if person_text:
                person_id = self._node_id("person", person_text)
                self.add_node(person_id, self.LABELS["person"], person_text, {"名称": person_text, "姓名": person_text})
                self.add_edge(event_id, "提及人员", person_id)

        for org_name in as_list(event.get("组织机构")):
            org_text = clean_text(org_name)
            if org_text:
                org_id = self._node_id("organization", org_text)
                self.add_node(org_id, self.LABELS["organization"], org_text, {"名称": org_text})
                self.add_edge(event_id, "提及组织", org_id)

        for equipment_name in as_list(event.get("装备设备")):
            equipment_text = clean_text(equipment_name)
            if equipment_text:
                equipment_id = self._node_id("equipment", equipment_text)
                self.add_node(
                    equipment_id,
                    self.LABELS["equipment"],
                    equipment_text,
                    {
                        "名称": equipment_text,
                        "型号": infer_model(equipment_text),
                    },
                )
                self.add_edge(event_id, "提及装备", equipment_id)

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
                    "名称": name,
                    "规范化地名": name,
                    "经纬度": clean_text(place.get("经纬度")),
                    "海拔": place.get("海拔"),
                    "事件主题": topic,
                },
            )
            return node_id

        name = clean_text(place)
        if not name:
            return None
        node_id = self._node_id("place", name)
        self.add_node(node_id, self.LABELS["place"], name, {"名称": name, "规范化地名": name})
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
                    "名称": name,
                    "姓名": name,
                    "军衔": clean_text(person.get("军衔")),
                    "职务": clean_text(person.get("职务")),
                    "事件主题": topic,
                },
            )
            return node_id

        name = clean_text(person)
        if not name:
            return None
        node_id = self._node_id("person", name)
        self.add_node(node_id, self.LABELS["person"], name, {"名称": name, "姓名": name})
        return node_id

    def _add_target_node(self, kind: str, target: Any, topic: str, place_id: Optional[str]) -> Optional[str]:
        if not isinstance(target, dict):
            name = clean_text(target)
            if not name:
                return None
            node_id = self._node_id(kind, name)
            self.add_node(node_id, self.LABELS[kind], name, {"名称": name, "事件主题": topic})
            return node_id

        target_type = clean_text(target.get("类型"))
        model = clean_text(target.get("型号"))
        name = model if model and model != "未知" else target_type
        if not name:
            return None

        node_id = self._node_id(kind, name)
        properties = {
            "名称": name,
            "类型": target_type,
            "型号": model,
            "数量": target.get("数量"),
            "事件主题": topic,
        }
        if kind in {"fortification", "vehicle"} and place_id:
            place_node = self.nodes.get(place_id, {})
            properties.setdefault("经纬度", place_node.get("properties", {}).get("经纬度"))
            properties.setdefault("海拔", place_node.get("properties", {}).get("海拔"))

        self.add_node(node_id, self.LABELS[kind], name, properties)
        if place_id:
            self.add_edge(node_id, "位于", place_id)
        return node_id

    def add_node(self, node_id: str, label: str, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        clean_properties = normalize_properties(properties or {})
        clean_properties["名称"] = clean_text(clean_properties.get("名称")) or name
        clean_properties["类别"] = merge_values(clean_properties.get("类别"), label)
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
    def _node_id(kind: str, name: str) -> str:
        return clean_text(name)


def infer_model(name: str) -> str:
    text = clean_text(name)
    if not text:
        return ""
    patterns = [
        r"T-90MS?",
        r"MQ-9B",
        r"99AE",
        r"PHL-03",
        r"BMP-2",
        r"翼龙-2",
        r"绿箭-8E",
        r"苍鹭",
        r"阿卡什",
        r"155毫米榴弹炮",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return text


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def normalize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in properties.items():
        if value == "":
            normalized[key] = ""
        elif isinstance(value, str):
            normalized[key] = value.strip()
        elif isinstance(value, list):
            normalized[key] = [normalize_scalar(item) for item in value if normalize_scalar(item) not in ("", None)]
        else:
            normalized[key] = value
    return normalized


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def merge_values(left: Any, right: Any) -> Any:
    values: List[Any] = []
    for item in as_list(left) + as_list(right):
        if item in ("", None, []):
            continue
        if item not in values:
            values.append(item)
    if len(values) == 1:
        return values[0]
    return values


def slugify(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[:：/\\|]", "_", text)
    return text


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def import_to_neo4j(graph_data: Dict[str, List[Dict[str, Any]]], url: str, user: str, password: str, clear: bool = False) -> None:
    from py2neo import Graph, Node as NeoNode, Relationship


    graph = Graph(url, auth=(user, password))
    if clear:
        graph.delete_all()

    neo_nodes: Dict[str, Any] = {}
    for node in graph_data["nodes"]:
        labels = node.get("labels") or [node["label"]]
        primary_label = labels[0]
        properties = to_neo4j_properties(node["properties"])
        properties["id"] = node["id"]
        properties["name"] = node["name"]
        neo_node = NeoNode(*labels, **properties)
        graph.merge(neo_node, primary_label, "name")
        neo_nodes[node["id"]] = neo_node

    for edge in graph_data["edges"]:
        source = neo_nodes.get(edge["source"])
        target = neo_nodes.get(edge["target"])
        if not source or not target:
            continue
        relationship = Relationship(source, edge["relation"], target, **to_neo4j_properties(edge.get("properties", {})))
        graph.merge(relationship)


def to_neo4j_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Neo4j does not store null values; keep only scalar/list values it accepts."""
    cleaned: Dict[str, Any] = {}
    for key, value in properties.items():
        if value in ("", None, []):
            continue
        if isinstance(value, list):
            values = [item for item in value if item not in ("", None)]
            if values:
                cleaned[key] = values
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="构建机理思维知识图谱")
    parser.add_argument("--event-entity-file", type=Path, default=DEFAULT_EVENT_ENTITY_FILE, help="out-5.20.1.json 路径")
    parser.add_argument("--mechanism-file", type=Path, default=DEFAULT_MECHANISM_FILE, help="out-5.20.2.json 路径")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="输出图谱 JSON 路径")
    parser.set_defaults(to_neo4j=True)
    parser.add_argument("--to-neo4j", dest="to_neo4j", action="store_true", help="将图谱写入 Neo4j")
    parser.add_argument("--no-neo4j", dest="to_neo4j", action="store_false", help="只生成 JSON，不写入 Neo4j")
    parser.add_argument("--neo4j-url", default="bolt://localhost:7687", help="Neo4j 地址，例如 bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j 用户名")
    parser.add_argument("--neo4j-password", default="12345678", help="Neo4j 密码")
    parser.add_argument("--clear-neo4j", action="store_true", help="写入前清空 Neo4j")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    builder = MechanismKGBuilder()
    source_configs = [
        ("mechanism_events", args.mechanism_file, builder._add_mechanism_event),
        ("event_entities", args.event_entity_file, builder._add_event_entity_summary),
    ]
    sources = [
        ExtractionSource(name=name, events=load_json(path), handler=handler)
        for name, path, handler in source_configs
    ]
    graph_data = builder.build(sources)
    write_json(args.output_file, graph_data)

    print(f"知识图谱构建完成：{args.output_file}")
    print(f"节点数量：{len(graph_data['nodes'])}")
    print(f"关系数量：{len(graph_data['edges'])}")

    if args.to_neo4j:
        import_to_neo4j(
            graph_data,
            url=args.neo4j_url,
            user=args.neo4j_user,
            password=args.neo4j_password,
            clear=args.clear_neo4j,
        )
        print("Neo4j 导入完成")


if __name__ == "__main__":
    main()
