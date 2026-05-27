# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
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
    build_node_description,
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
DEFAULT_WRITE_TO_NEO4J = bool(KG_CONFIG.get("write_to_neo4j", True))
DEFAULT_CLEAR_NEO4J = bool(KG_CONFIG.get("clear_neo4j", False))


Node = Dict[str, Any]
Edge = Dict[str, Any]


def make_output_file() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{DEFAULT_OUTPUT_PREFIX}_{timestamp}.json"


class MechanismKGBuilder:
    """Create a node-property graph for mechanism thinking maps."""

    PRIORITY_NODE_TYPES = {
        "反坦克锥",
        "弹药库",
        "油库",
        "指挥部",
        "有限掩体",
        "加农炮",
        "圆周掩体",
        "坦克",
        "正方型掩体",
        "C型掩体",
        "装甲车",
        "运输车",
        "堑壕",
    }

    PRIORITY_TYPE_RULES = (
        ("反坦克锥", ("反坦克锥", "阻车锥", "反坦克障碍锥", "龙牙")),
        ("弹药库", ("弹药库", "弹药仓库", "弹药储备库", "弹药储存库", "弹药补给库")),
        ("油库", ("油库", "油料库", "燃油库", "燃料库", "油料仓库")),
        ("指挥部", ("指挥部", "指挥所", "指挥中心", "临时指挥中心", "作战指挥中心", "前线指挥所")),
        ("有限掩体", ("有限掩体",)),
        ("圆周掩体", ("圆周掩体", "环形掩体", "圆形掩体")),
        ("正方型掩体", ("正方型掩体", "正方形掩体", "方形掩体")),
        ("C型掩体", ("C型掩体", "C形掩体", "c型掩体", "c形掩体")),
        ("堑壕", ("堑壕", "战壕", "壕沟", "交通壕")),
        ("装甲车", ("装甲车", "装甲车辆", "步兵战车", "战车", "装甲输送车")),
        ("运输车", ("运输车", "运输车辆", "输送车", "补给车", "后勤车", "卡车", "军用卡车")),
        ("加农炮", ("加农炮", "自行加农炮", "牵引式加农炮")),
        ("坦克", ("坦克", "主战坦克", "轻型坦克", "T-", "VT-", "MBT", "99AE", "T90", "T-90")),
    )

    GENERIC_CONFLICT_TYPES = {
        "车辆",
        "主战坦克",
        "轻型坦克",
        "装甲车辆",
        "运输车辆",
        "工事",
        "防御工事",
        "设施",
        "装备",
        "装备设备",
        "仓库",
        "仓储设施",
        "储存设施",
        "库房",
        "掩体",
        "防护工事",
        "火炮",
        "炮",
        "指挥中心",
        "指挥所",
        "指挥机构",
        "后勤设施",
        "补给设施",
        "障碍物",
    }

    CONTEXT_PRIORITY_TYPES = {
        "车辆": {"坦克", "装甲车", "运输车"},
        "工事": {"反坦克锥", "有限掩体", "圆周掩体", "正方型掩体", "C型掩体", "堑壕"},
        "装备设备": {"反坦克锥", "加农炮", "坦克", "装甲车", "运输车"},
        "设施": {"弹药库", "油库", "指挥部"},
        "组织机构": {"指挥部"},
        "实体": PRIORITY_NODE_TYPES,
    }

    LABELS = {
        "event": "事件",
        "behavior": "行为",
        "time": "时间",
        "place": "地点",
        "person": "人员",
        "organization": "组织机构",
        "equipment": "装备设备",
        "vessel": "舰艇",
        "aircraft": "航空器",
        "facility": "设施",
        "entity": "实体",
        "fortification": "工事",
        "vehicle": "车辆",
        "unit": "兵种",
        "risk": "风险等级",
    }

    TARGET_CATEGORY_MAP = {
        "工事": ("fortification", "涉及工事"),
        "车辆": ("vehicle", "涉及车辆"),
        "兵种": ("unit", "涉及兵种"),
        "舰艇": ("vessel", "涉及舰艇"),
        "航空器": ("aircraft", "涉及航空器"),
        "装备设备": ("equipment", "涉及装备"),
        "设施": ("facility", "涉及设施"),
    }

    TARGET_HANDLED_KEYS = {
        "name",
        "type",
        "model",
        "quantity",
        "quantity_value",
        "quantity_unit",
        "side",
        "organization",
        "location",
        "mission",
        "status",
        "purpose",
        "caliber",
        "range",
        "accuracy",
        "response_time",
        "response_time_value",
        "response_time_comparison",
        "duration",
        "speed",
        "height",
        "distance",
        "movement_direction",
        "coordinates",
        "altitude",
        "event_topic",
        "from_file",
        "label",
        "labels",
        "id",
    }

    ORG_HANDLED_KEYS = {
        "name",
        "type",
        "side",
        "parent_organization",
        "location",
        "mission",
        "coordinates",
        "altitude",
        "event_topic",
        "from_file",
        "label",
        "labels",
        "id",
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
        behavior_text = clean_text(event.get("核心行为")) or topic
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

        for place in as_list(event.get("地点列表")):
            extra_place_id = self._add_place(place, topic)
            if extra_place_id and extra_place_id != place_id:
                self.add_edge(behavior_id, "涉及地点", extra_place_id)

        risk = clean_text(event.get("风险等级"))
        if risk:
            risk_id = self._node_id("risk", risk)
            self.add_node(risk_id, self.LABELS["risk"], risk, {"name": risk, "level": risk})
            self.add_edge(behavior_id, "风险等级", risk_id)

        for person in as_list(event.get("人员信息")):
            person_id = self._add_person(person, topic)
            if person_id:
                self.add_edge(behavior_id, "涉及人员", person_id)

        for organization in as_list(event.get("组织机构")):
            organization_id = self._add_organization(organization, topic)
            if organization_id:
                self.add_edge(behavior_id, "涉及组织", organization_id)

        target_system = event.get("目标体系") or {}
        if not isinstance(target_system, dict):
            target_system = {}

        for category, (kind, relation) in self.TARGET_CATEGORY_MAP.items():
            for target in as_list(target_system.get(category)):
                node_id = self._add_target_node(kind, target, topic, place_id)
                if node_id:
                    self.add_edge(behavior_id, relation, node_id)

        for organization in as_list(target_system.get("组织机构")):
            organization_id = self._add_organization(organization, topic)
            if organization_id:
                self.add_edge(behavior_id, "涉及组织", organization_id)

        for person in as_list(target_system.get("人员")):
            person_id = self._add_person(person, topic)
            if person_id:
                self.add_edge(behavior_id, "涉及人员", person_id)

        self._add_explicit_relationships(event)

    def _add_place(self, place: Any, topic: str) -> Optional[str]:
        if isinstance(place, dict):
            name = clean_text(place.get("规范化地名")) or clean_text(place.get("名称"))
            coordinates = clean_text(place.get("经纬度"))
            if not name:
                name = coordinates
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
                    "coordinates": coordinates,
                    "altitude": place.get("海拔"),
                    "direction": clean_text(place.get("方位")),
                    "distance": clean_text(place.get("距离")),
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
                    "side": clean_text(person.get("所属方")),
                    "organization": clean_text(person.get("所属组织")),
                    "commanded_target": clean_text(person.get("指挥对象")),
                    "event_topic": topic,
                },
            )
            self._attach_ownership_edges(node_id, person)
            return node_id

        name = clean_text(person)
        if not name:
            return None
        node_id = self._node_id("person", name)
        self.add_node(node_id, self.LABELS["person"], name, {"name": name, "person_name": name})
        return node_id

    def _add_organization(self, organization: Any, topic: str) -> Optional[str]:
        if isinstance(organization, dict):
            name = clean_text(organization.get("名称")) or clean_text(organization.get("姓名"))
            if not name:
                return None

            node_id = self._node_id("organization", name)
            properties = self._collect_properties(organization, handled_keys=self.ORG_HANDLED_KEYS)
            properties.update({
                "name": name,
                "type": clean_text(organization.get("类型")) or self.LABELS["organization"],
                "side": clean_text(organization.get("所属方")),
                "parent_organization": clean_text(organization.get("上级组织")),
                "location": clean_text(organization.get("位置")),
                "mission": clean_text(organization.get("任务")),
                "event_topic": topic,
            })
            self.add_node(node_id, self.LABELS["organization"], name, properties)
            self._attach_ownership_edges(node_id, organization)
            self._attach_location_edge(node_id, organization, topic, None)
            return node_id

        name = clean_text(organization)
        if not name:
            return None
        node_id = self._node_id("organization", name)
        self.add_node(node_id, self.LABELS["organization"], name, {"name": name, "event_topic": topic})
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
        explicit_name = clean_text(target.get("名称")) or clean_text(target.get("姓名"))
        name = model if model and model != "未知" else explicit_name or target_type
        if not name:
            return None

        node_id = self._node_id(kind, name)
        properties = self._collect_properties(target, handled_keys=self.TARGET_HANDLED_KEYS)
        normalized_type = self._normalize_target_type(kind, target_type, model, " ".join([name, explicit_name]))
        side = clean_text(target.get("所属方"))
        organization = clean_text(target.get("所属组织"))
        location = clean_text(target.get("位置")) or self._place_name_from_id(place_id)
        quantity_value = target.get("数量")
        quantity_text = self._format_quantity(quantity_value, kind, normalized_type, model, name, location)
        response_time_value = clean_text(target.get("响应时间")) or clean_text(target.get("反应时间"))
        properties.update({
            "name": name,
            "type": normalized_type,
            "model": model,
            "quantity": quantity_text,
            "quantity_value": quantity_value,
            "quantity_unit": self._infer_quantity_unit(kind, normalized_type, model, name) if quantity_text else "",
            "side": side,
            "organization": organization,
            "location": location,
            "mission": clean_text(target.get("任务")),
            "status": clean_text(target.get("状态")),
            "purpose": clean_text(target.get("用途")),
            "caliber": clean_text(target.get("口径")),
            "range": clean_text(target.get("射程")),
            "accuracy": clean_text(target.get("精度")),
            "response_time": self._format_response_time(
                response_time_value,
                kind,
                normalized_type,
                model,
                name,
                clean_text(target.get("任务")),
                clean_text(target.get("用途")),
            ),
            "response_time_value": response_time_value,
            "duration": clean_text(target.get("耗时")),
            "speed": clean_text(target.get("速度")),
            "height": clean_text(target.get("高度")),
            "distance": clean_text(target.get("距离")),
            "movement_direction": clean_text(target.get("运动方向")),
            "event_topic": topic,
        })
        if kind in {"fortification", "vehicle", "equipment", "vessel", "aircraft", "facility"} and place_id:
            place_node = self.nodes.get(place_id, {})
            properties.setdefault("coordinates", clean_text(target.get("经纬度")) or place_node.get("properties", {}).get("coordinates"))
            properties.setdefault("altitude", target.get("海拔") or place_node.get("properties", {}).get("altitude"))

        self.add_node(node_id, self.LABELS[kind], name, properties)
        self._attach_location_edge(node_id, target, topic, place_id)
        self._attach_ownership_edges(node_id, target)
        return node_id

    @staticmethod
    def _flatten_type_values(value: Any) -> List[str]:
        values: List[str] = []
        for item in as_list(value):
            text = clean_text(item)
            if text and text not in values:
                values.append(text)
        return values

    @classmethod
    def _priority_type_from_text(cls, *values: Any) -> str:
        text = " ".join(clean_text(value) for value in values if clean_text(value))
        if not text:
            return ""
        text_lower = text.lower()
        for canonical, keywords in cls.PRIORITY_TYPE_RULES:
            if canonical == "坦克" and "反坦克" in text and "反坦克锥" not in text:
                continue
            if any(keyword.lower() in text_lower for keyword in keywords):
                return canonical
        return ""

    @classmethod
    def _type_conflicts_with_priority(cls, value: str, priority_types: List[str]) -> bool:
        text = clean_text(value)
        if not text:
            return True
        if text in cls.GENERIC_CONFLICT_TYPES:
            return True
        mapped_type = cls._priority_type_from_text(text)
        if mapped_type:
            return True
        for priority_type in priority_types:
            if priority_type in text or text in priority_type:
                return True
        return False

    @classmethod
    def _priority_type_from_context(cls, label: str, *values: Any) -> str:
        priority_type = cls._priority_type_from_text(*values)
        allowed_types = cls.CONTEXT_PRIORITY_TYPES.get(clean_text(label), set())
        if priority_type in allowed_types:
            return priority_type
        return ""

    @classmethod
    def _normalize_type_property(
        cls,
        value: Any,
        label: str,
        name: str,
        model: Any = "",
    ) -> Any:
        raw_values = cls._flatten_type_values(value)
        priority_values: List[str] = []
        other_values: List[str] = []

        for item in raw_values:
            priority_type = cls._priority_type_from_text(item)
            if priority_type:
                if priority_type not in priority_values:
                    priority_values.append(priority_type)
            elif item not in other_values:
                other_values.append(item)

        context_priority_type = cls._priority_type_from_context(label, name, model)
        if context_priority_type and context_priority_type not in priority_values:
            priority_values.append(context_priority_type)

        if priority_values:
            merged = priority_values + [
                item
                for item in other_values
                if not cls._type_conflicts_with_priority(item, priority_values)
            ]
        else:
            merged = other_values or [clean_text(label)]

        merged = [item for item in merged if item]
        if not merged:
            return ""
        if len(merged) == 1:
            return merged[0]
        return merged

    @classmethod
    def _normalize_target_type(cls, kind: str, target_type: str, model: str, name: str) -> Any:
        raw_type = clean_text(target_type)
        raw_model = clean_text(model)
        raw_name = clean_text(name)
        text = " ".join([raw_type, raw_model, raw_name])
        priority_type = cls._priority_type_from_text(text)
        if priority_type:
            return cls._normalize_type_property(
                [priority_type, raw_type],
                cls.LABELS.get(kind, kind),
                "",
                "",
            )

        if kind == "vehicle":
            if raw_type and not any(keyword in raw_type for keyword in ("连", "排", "营", "旅", "团", "师", "军", "队")):
                return clean_text(cls._normalize_type_property(raw_type, "车辆", raw_name, raw_model)) or raw_type
            if any(keyword in text for keyword in ("无人机", "UAV", "MQ-", "翼龙", "彩虹", "侦察机")):
                return "无人机"
            if any(keyword in text for keyword in ("运输车", "运输车辆", "输送车", "补给车", "卡车")):
                return "运输车"
            if any(keyword in text for keyword in ("步兵战车", "装甲车", "装甲车辆", "战车")):
                return "装甲车"
            if any(keyword in text for keyword in ("坦克", "T-", "VT-", "MBT", "主战坦克")):
                return "坦克"
            if any(keyword in text for keyword in ("火箭炮",)):
                return "火箭炮"
            return raw_type or "车辆"

        if kind == "equipment":
            if raw_type and raw_type not in {"未知", "其他"}:
                return raw_type
            if any(keyword in text for keyword in ("防空", "导弹", "火箭炮", "火炮", "雷达", "传感器", "通信", "数据链")):
                return raw_type or "装备设备"
            return raw_type or "装备设备"

        if kind == "fortification":
            return raw_type or "工事"

        if kind == "facility":
            return raw_type or "设施"

        if kind == "aircraft":
            return raw_type or "航空器"

        if kind == "vessel":
            return raw_type or "舰艇"

        if kind == "unit":
            return raw_type or "兵种"

        return raw_type or MechanismKGBuilder.LABELS.get(kind, kind)

    def _place_name_from_id(self, place_id: Optional[str]) -> str:
        if not place_id:
            return ""
        place_node = self.nodes.get(place_id) or {}
        properties = place_node.get("properties", {}) or {}
        return (
            clean_text(properties.get("normalized_place"))
            or clean_text(properties.get("name"))
            or clean_text(place_node.get("name"))
        )

    @classmethod
    def _format_quantity(
        cls,
        quantity: Any,
        kind: str,
        target_type: str,
        model: str,
        name: str,
        location: str,
    ) -> str:
        quantity_text = clean_text(quantity)
        if not quantity_text or quantity_text in {"未知", "不详", "unknown", "None"}:
            return ""
        count_text = quantity_text
        if not re.search(r"(辆|架|门|套|个|枚|具|艘|人|名|支|台|部|连|排|营|旅|团|师|军)$", count_text):
            count_text = f"{count_text}{cls._infer_quantity_unit(kind, target_type, model, name)}"
        if location:
            return f"{location}有{count_text}"
        return f"{name}数量{count_text}"

    @staticmethod
    def _infer_quantity_unit(kind: str, target_type: str, model: str, name: str) -> str:
        text = " ".join([kind, clean_text(target_type), clean_text(model), clean_text(name)])
        if kind == "aircraft" or any(keyword in text for keyword in ("无人机", "UAV", "MQ-", "翼龙", "彩虹", "飞机", "直升机")):
            return "架"
        if kind == "vessel" or any(keyword in text for keyword in ("舰", "艇", "船")):
            return "艘"
        if any(keyword in text for keyword in ("坦克", "装甲", "战车", "车辆", "车")) or kind == "vehicle":
            return "辆"
        if any(keyword in text for keyword in ("火箭炮", "火炮", "榴弹炮", "炮")):
            return "门"
        if any(keyword in text for keyword in ("发射器", "发射具")):
            return "具"
        if any(keyword in text for keyword in ("导弹", "弹药")):
            return "枚"
        if any(keyword in text for keyword in ("系统", "雷达", "中心", "站")):
            return "套"
        return "个"

    @classmethod
    def _format_response_time(
        cls,
        response_time: Any,
        kind: str,
        target_type: str,
        model: str,
        name: str,
        mission: str,
        purpose: str,
    ) -> str:
        raw = clean_text(response_time)
        if not raw or raw in {"未知", "不详", "unknown", "None"}:
            return ""
        if "响应时间" in raw or "反应时间" in raw:
            return raw
        subject = cls._response_time_subject(kind, target_type, model, name, mission, purpose)
        return f"{subject}响应时间{raw}"

    @staticmethod
    def _response_time_subject(
        kind: str,
        target_type: str,
        model: str,
        name: str,
        mission: str,
        purpose: str,
    ) -> str:
        text = " ".join([
            kind,
            clean_text(target_type),
            clean_text(model),
            clean_text(name),
            clean_text(mission),
            clean_text(purpose),
        ])
        if any(keyword in text for keyword in ("坦克", "装甲", "战车", "T-", "VT-", "99")):
            return "火控系统"
        if any(keyword in text for keyword in ("导弹", "防空", "发射", "火箭炮", "火炮")):
            return "武器系统"
        if any(keyword in text for keyword in ("无人机", "侦察", "数据链", "链路")):
            return "任务链路"
        return "系统"

    def _attach_location_edge(
        self,
        node_id: str,
        source: Dict[str, Any],
        topic: str,
        fallback_place_id: Optional[str],
    ) -> None:
        location = clean_text(source.get("位置")) or clean_text(source.get("地点")) or clean_text(source.get("规范化地名"))
        coordinates = clean_text(source.get("经纬度"))
        place_id = None
        if location or coordinates:
            place_id = self._add_place(
                {
                    "名称": location or coordinates,
                    "经纬度": coordinates,
                    "海拔": source.get("海拔"),
                    "距离": source.get("距离"),
                    "方位": source.get("方位"),
                },
                topic,
            )
        place_id = place_id or fallback_place_id
        if place_id:
            self.add_edge(node_id, "位于", place_id)

    def _attach_ownership_edges(self, node_id: str, source: Dict[str, Any]) -> None:
        side = clean_text(source.get("所属方")) or clean_text(source.get("阵营"))
        organization = clean_text(source.get("所属组织"))
        parent_organization = clean_text(source.get("上级组织"))
        commanded_target = clean_text(source.get("指挥对象"))

        if side:
            side_id = self._ensure_generic_node(side, "organization", {"name": side, "side": side})
            self.add_edge(side_id, "拥有", node_id)

        if organization:
            organization_id = self._ensure_generic_node(
                organization,
                "organization",
                {"name": organization, "side": side},
            )
            self.add_edge(node_id, "隶属", organization_id)
            self.add_edge(organization_id, "装备", node_id)
            if side:
                side_id = self._ensure_generic_node(side, "organization", {"name": side, "side": side})
                self.add_edge(organization_id, "隶属", side_id)

        if parent_organization:
            parent_id = self._ensure_generic_node(
                parent_organization,
                "organization",
                {"name": parent_organization, "side": side},
            )
            self.add_edge(node_id, "隶属", parent_id)

        if commanded_target:
            target_id = self._resolve_node_id(commanded_target)
            if target_id:
                self.add_edge(node_id, "指挥", target_id)

    def _add_explicit_relationships(self, event: Dict[str, Any]) -> None:
        relationships = []
        for key in ("关联关系", "关系", "relationships", "edges"):
            relationships.extend(as_list(event.get(key)))

        for relationship in relationships:
            if not isinstance(relationship, dict):
                continue
            source_name = (
                clean_text(relationship.get("头实体"))
                or clean_text(relationship.get("源实体"))
                or clean_text(relationship.get("from"))
                or clean_text(relationship.get("source"))
            )
            relation = (
                clean_text(relationship.get("关系"))
                or clean_text(relationship.get("relation"))
                or clean_text(relationship.get("type"))
            )
            target_name = (
                clean_text(relationship.get("尾实体"))
                or clean_text(relationship.get("目标实体"))
                or clean_text(relationship.get("to"))
                or clean_text(relationship.get("target"))
            )
            if not source_name or not relation or not target_name:
                continue

            source_id = self._resolve_node_id(source_name)
            target_id = self._resolve_node_id(target_name)
            if not source_id or not target_id:
                continue

            properties = {}
            attrs = relationship.get("属性")
            if isinstance(attrs, dict):
                properties.update(attrs)
            evidence = clean_text(relationship.get("证据"))
            if evidence:
                properties["evidence"] = evidence
            self.add_edge(source_id, relation, target_id, properties)

    @staticmethod
    def _collect_properties(item: Dict[str, Any], handled_keys: Optional[set] = None) -> Dict[str, Any]:
        handled = set(handled_keys or set())
        return {
            str(key): value
            for key, value in (item or {}).items()
            if value not in ("", None, [])
            and str(key).isascii()
            and str(key) not in handled
        }

    def _ensure_generic_node(
        self,
        name: str,
        kind: str = "entity",
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        node_id = self._node_id(kind, name)
        label = self.LABELS.get(kind, self.LABELS["entity"])
        self.add_node(node_id, label, name, {"name": name, **(properties or {})})
        return node_id

    def _resolve_node_id(self, name: str) -> Optional[str]:
        clean_name = clean_text(name)
        if not clean_name:
            return None
        direct_id = self._node_id("entity", clean_name)
        if direct_id in self.nodes:
            return direct_id

        target_key = self._entity_key(clean_name)
        for node_id, node in self.nodes.items():
            names = [
                node.get("name", ""),
                node.get("properties", {}).get("name", ""),
                node.get("properties", {}).get("model", ""),
                node.get("properties", {}).get("type", ""),
                node.get("properties", {}).get("person_name", ""),
            ]
            for candidate in names:
                candidate_key = self._entity_key(candidate)
                if not candidate_key:
                    continue
                if target_key == candidate_key:
                    return node_id
                if len(target_key) >= 4 and (target_key in candidate_key or candidate_key in target_key):
                    return node_id

        return self._ensure_generic_node(clean_name)

    @staticmethod
    def _entity_key(value: Any) -> str:
        text = clean_text(value).lower()
        text = text.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
        for token in (" ", "\t", "\n", "（", "）", "(", ")", "-", "_"):
            text = text.replace(token, "")
        for suffix in ("主战坦克", "坦克", "无人机", "战车", "导弹", "火箭炮系统", "火箭炮", "防空系统", "系统", "装备", "设备"):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
        return text

    def add_node(self, node_id: str, label: str, name: str, properties: Optional[Dict[str, Any]] = None) -> None:
        clean_properties = normalize_node_properties(normalize_properties(properties or {}), label, self.current_source_file)
        clean_properties["name"] = clean_text(clean_properties.get("name")) or name
        clean_properties["type"] = self._normalize_type_property(
            clean_properties.get("type"),
            label,
            clean_properties["name"],
            clean_properties.get("model", ""),
        )
        clean_properties["desc"] = build_node_description(clean_properties, label)
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
            if key == "desc":
                continue
            if value in ("", None, []):
                continue
            if key not in existing or existing[key] in ("", None, []):
                existing[key] = value
            elif existing[key] != value:
                existing[key] = merge_values(existing[key], value)
            if key == "type":
                existing["type"] = self._normalize_type_property(
                    existing.get("type"),
                    self.nodes[node_id]["label"],
                    existing.get("name", name),
                    existing.get("model", ""),
                )
        desc_source = {key: value for key, value in existing.items() if key != "desc"}
        existing["desc"] = build_node_description(desc_source, self.nodes[node_id]["label"])

    def add_edge(
        self,
        source: str,
        relation: str,
        target: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        if source not in self.nodes or target not in self.nodes:
            return
        if source == target:
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
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    mechanism_events: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []
    failed_files: List[Dict[str, Any]] = []
    input_paths = list(input_paths)

    iterator = input_paths
    if input_paths:
        try:
            from tqdm import tqdm

            iterator = tqdm(input_paths, desc="读取输入文件", total=len(input_paths), unit="file", dynamic_ncols=True)
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
    parser.add_argument("--input-files", type=Path, nargs="+", required=True, help="csv/txt/pdf/docx/md/markdown/图片/音视频 输入文件路径")
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
