# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import init_config


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FILES_CONFIG = init_config.get("files", {})
KG_CONFIG = init_config.get("knowledge_graph", {})
NEO4J_CONFIG = init_config.get("neo4j", {})
SPLITTER_CONFIG = KG_CONFIG.get("splitter", {})


def resolve_path(value: Any, fallback: Path) -> Path:
    if value in (None, ""):
        return fallback
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def normalize_suffixes(values: Iterable[Any]) -> Tuple[str, ...]:
    suffixes: List[str] = []
    for value in values or []:
        suffix = str(value).strip().lower()
        if not suffix:
            continue
        if not suffix.startswith("."):
            suffix = f".{suffix}"
        if suffix not in suffixes:
            suffixes.append(suffix)
    return tuple(suffixes)


DEFAULT_SUPPORTED_SUFFIXES = normalize_suffixes(KG_CONFIG.get("supported_suffixes", [".csv", ".docx", ".pdf", ".txt", ".png", ".jpg", ".jpeg"]))
TEXT_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("text_suffixes", [".txt", ".md", ".csv", ".pdf", ".docx"]))
IMAGE_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("image_suffixes", [".png", ".jpg", ".jpeg"]))
AUDIO_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("audio_suffixes", [".mp3", ".wav", ".m4a"]))
VIDEO_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("video_suffixes", [".mp4", ".avi", ".mov", ".mkv"]))
SUPPORTED_SOURCE_SUFFIXES = tuple(dict.fromkeys(
    DEFAULT_SUPPORTED_SUFFIXES
    + TEXT_SOURCE_SUFFIXES
    + IMAGE_SOURCE_SUFFIXES
    + AUDIO_SOURCE_SUFFIXES
    + VIDEO_SOURCE_SUFFIXES
))

DEFAULT_CHUNK_SIZE = int(SPLITTER_CONFIG.get("chunk_size", 1800))
DEFAULT_CHUNK_OVERLAP = int(SPLITTER_CONFIG.get("chunk_overlap", 120))
DEFAULT_SPLITTER_SEPARATORS = tuple(SPLITTER_CONFIG.get("separators", ["\n", "。", "？", "；", "!", "！", "?"]))


def get_neo4j_config() -> Dict[str, str]:
    return {
        "url": str(NEO4J_CONFIG.get("url", "bolt://localhost:7687")),
        "user": str(NEO4J_CONFIG.get("user", "neo4j")),
        "password": str(NEO4J_CONFIG.get("password", "12345678")),
    }


def build_split_pattern(separators: Iterable[str]) -> str:
    parts = []
    for separator in separators:
        if not separator:
            continue
        if separator in {"\n", "\r\n"}:
            parts.append(r"\r?\n+")
        else:
            parts.append(re.escape(separator))
    if not parts:
        return r"\r?\n+"
    return "(?:" + "|".join(parts) + ")"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def normalize_scalar(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


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


PROPERTY_KEY_MAP = {
    "类型": "type",
    "类别": "category",
    "名称": "name",
    "姓名": "person_name",
    "描述": "desc",
    "说明": "desc",
    "内容": "content",
    "来源文件": "from_file",
    "经纬度": "coordinates",
    "规范化地名": "normalized_place",
    "海拔": "altitude",
    "军衔": "rank",
    "职务": "position",
    "数量": "quantity",
    "数量值": "quantity_value",
    "数量单位": "quantity_unit",
    "型号": "model",
    "事件主题": "event_topic",
    "核心行为": "core_behavior",
    "目标体系": "target_system",
    "地点信息": "location_info",
    "地点列表": "location_list",
    "人员信息": "personnel_info",
    "风险等级": "risk_level",
    "所属方": "side",
    "阵营": "side",
    "所属组织": "organization",
    "上级组织": "parent_organization",
    "指挥对象": "commanded_target",
    "位置": "location",
    "地点": "location",
    "方位": "direction",
    "方向": "direction",
    "距离": "distance",
    "任务": "mission",
    "状态": "status",
    "用途": "purpose",
    "口径": "caliber",
    "射程": "range",
    "精度": "accuracy",
    "命中精度": "accuracy",
    "响应时间": "response_time",
    "反应时间": "response_time",
    "响应时间值": "response_time_value",
    "耗时": "duration",
    "速度": "speed",
    "高度": "height",
    "运动方向": "movement_direction",
    "装备": "equipment",
    "可用": "available",
    "证据": "evidence",
    "响应时间对比": "response_time_comparison",
    "印军响应时间": "indian_response_time",
    "巴军响应时间": "pakistan_response_time",
    "中方响应时间": "china_response_time",
}


DESCRIPTION_FIELDS = [
    ("type", "类型"),
    ("model", "型号"),
    ("side", "所属方"),
    ("organization", "所属组织"),
    ("location", "位置"),
    ("quantity", "数量"),
    ("mission", "任务"),
    ("status", "状态"),
    ("purpose", "用途"),
    ("rank", "军衔"),
    ("position", "职务"),
    ("event_topic", "事件"),
    ("time", "时间"),
    ("coordinates", "坐标"),
    ("altitude", "海拔"),
]


def _description_value(value: Any) -> str:
    if value in ("", None, [], {}):
        return ""
    if isinstance(value, list):
        return "、".join(text for text in (_description_value(item) for item in value) if text)
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            text = _description_value(item)
            if text:
                pairs.append(f"{key}:{text}")
        return "；".join(pairs)
    return clean_text(value)


def build_node_description(properties: Dict[str, Any], label: str = "") -> str:
    props = properties or {}
    existing = clean_text(props.get("desc")) or clean_text(props.get("description"))
    if existing:
        return existing

    label_text = clean_text(label) or clean_text(props.get("label")) or clean_text(props.get("type")) or "实体"
    name = clean_text(props.get("name")) or _description_value(props.get("person_name"))
    type_text = _description_value(props.get("type"))
    subject = name or type_text or label_text
    prefix = f"{label_text}节点：{subject}"

    details = []
    seen_values = {subject}
    for key, title in DESCRIPTION_FIELDS:
        value_text = _description_value(props.get(key))
        if not value_text or value_text in seen_values:
            continue
        if key == "type" and value_text == label_text:
            continue
        details.append(f"{title}{value_text}")
        seen_values.add(value_text)

    if details:
        return f"{prefix}；" + "；".join(details)
    return prefix


def normalize_property_key(key: Any) -> str:
    key_text = clean_text(key)
    if not key_text:
        return ""
    mapped_key = PROPERTY_KEY_MAP.get(key_text)
    if mapped_key:
        return mapped_key
    if key_text.isascii():
        return key_text
    return ""


def normalize_node_properties(properties: Dict[str, Any], label: str, source_file: str) -> Dict[str, Any]:

    result: Dict[str, Any] = {}
    for key, value in (properties or {}).items():
        if value in ("", None, []):
            continue
        mapped_key = normalize_property_key(key)
        if not mapped_key:
            continue
        if mapped_key in {"accuracy", "precision"}:
            text = clean_text(value)
            if text and not re.search(r"\d|%|km|m|米|秒|度|倍", text):
                continue
        result[mapped_key] = value

    if not clean_text(result.get("type")):
        result["type"] = label
    result["name"] = clean_text(result.get("name"))
    if source_file:
        result["from_file"] = merge_values(result.get("from_file"), source_file)
    result["desc"] = build_node_description(result, label)
    return {
        key: value
        for key, value in result.items()
        if key.isascii() and value not in ("", None, [])
    }


def edge_to_link(edge: Dict[str, Any]) -> Dict[str, Any]:
    link = {
        "source": edge.get("source", ""),
        "target": edge.get("target", ""),
        "relation": edge.get("relation", ""),
    }
    properties = edge.get("properties") or {}
    if properties:
        link["properties"] = properties
    return link


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def read_csv_file(path: Path) -> str:
    import pandas as pd

    last_error: Optional[Exception] = None
    for encoding in ("utf-8", "utf-8-sig", "gb18030"):
        try:
            dataframe = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_error = exc
    else:
        if last_error:
            raise last_error
        dataframe = pd.read_csv(path)

    lines: List[str] = []
    dataframe = dataframe.fillna("")
    for row_index, row in dataframe.iterrows():
        fields = []
        for column, value in row.items():
            text = clean_text(value)
            if text:
                fields.append(f"{column}: {text}")
        if fields:
            lines.append(f"row {row_index + 1}: " + "；".join(fields))
    return "\n".join(lines)


def read_pdf_file(path: Path) -> str:
    import fitz

    pages: List[str] = []
    with fitz.open(path) as document:
        for page in document:
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(text)
    return "\n\n".join(pages)


def read_docx_file(path: Path) -> str:
    import mammoth

    try:
        with path.open("rb") as file:
            result = mammoth.extract_raw_text(file)
        return result.value or ""
    except Exception:
        return read_text_file(path)


def read_source_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        return read_text_file(path)
    if suffix == ".csv":
        return read_csv_file(path)
    if suffix == ".pdf":
        return read_pdf_file(path)
    if suffix == ".docx":
        return read_docx_file(path)
    supported = ", ".join(sorted(TEXT_SOURCE_SUFFIXES))
    raise ValueError(f"不支持的文本文件类型: {suffix or '无后缀'}，支持: {supported}")


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[Iterable[str]] = None,
) -> List[str]:
    text = re.sub(r"\r\n?", "\n", clean_text(text))
    text = re.sub(r"[ \t]+", " ", text)
    if not text:
        return []

    chunk_size = max(int(chunk_size), 1)
    chunk_overlap = max(min(int(chunk_overlap), chunk_size - 1), 0)
    step = max(chunk_size - chunk_overlap, 1)
    split_pattern = build_split_pattern(separators or DEFAULT_SPLITTER_SEPARATORS)
    pieces = [piece.strip() for piece in re.split(split_pattern, text) if piece.strip()]

    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if len(piece) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            for start in range(0, len(piece), step):
                chunk = piece[start : start + chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            continue

        candidate = piece if not current else f"{current}\n{piece}"
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = piece

    if current:
        chunks.append(current)
    return chunks


def make_news_chunks(source_name: str, text: str) -> List[Dict[str, str]]:
    chunks = split_text(
        text,
        chunk_size=int(SPLITTER_CONFIG.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        chunk_overlap=int(SPLITTER_CONFIG.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)),
        separators=SPLITTER_CONFIG.get("separators", DEFAULT_SPLITTER_SEPARATORS),
    )
    return [
        {
            "title": f"{source_name}#{index}",
            "content": chunk,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def extract_json_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("events", "data", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        return [data]
    return []


def attach_source(events: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    output = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        item.setdefault("来源文件", source_name)
        item.setdefault("from_file", source_name)
        output.append(item)
    return output


def extract_events_with_grapher(news_listdict_in: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    if not news_listdict_in:
        return []
    from utils.txt_to_json import graph_cls

    grapher = graph_cls()
    abstract_listdict = grapher.graph_main(news_listdict_in)
    return [item for item in abstract_listdict if isinstance(item, dict)]


def events_from_text_file(path: Path, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    text = read_source_file(path)
    news_listdict_in = make_news_chunks(source_name, text)
    events = extract_events_with_grapher(news_listdict_in)
    return attach_source(events, source_name), len(news_listdict_in)


def call_converter(module_name: str, function_names: Iterable[str], path: Path) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"缺少转换模块 {module_name}: {exc}") from exc

    for function_name in function_names:
        converter = getattr(module, function_name, None)
        if callable(converter):
            return converter(path)

    names = ", ".join(function_names)
    raise AttributeError(f"{module_name} 中未找到可调用函数: {names}")


def normalize_converter_output(output: Any, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    chunks_count = 0
    if isinstance(output, tuple) and output:
        output = output[0]

    if isinstance(output, str):
        news_listdict_in = make_news_chunks(source_name, output)
        chunks_count = len(news_listdict_in)
        return attach_source(extract_events_with_grapher(news_listdict_in), source_name), chunks_count

    records = extract_json_records(output)
    if records:
        if all("content" in item for item in records):
            news_listdict_in = [
                {
                    "title": clean_text(item.get("title")) or f"{source_name}#{index}",
                    "content": clean_text(item.get("content")),
                }
                for index, item in enumerate(records, start=1)
                if clean_text(item.get("content"))
            ]
            chunks_count = len(news_listdict_in)
            return attach_source(extract_events_with_grapher(news_listdict_in), source_name), chunks_count

        return attach_source(records, source_name), chunks_count

    return [], chunks_count


def events_from_image_file(path: Path, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    output = call_converter("utils.image_to_json", ("images_to_json", "image_to_json", "convert"), path)
    return normalize_converter_output(output, source_name)


def events_from_audio_file(path: Path, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    output = call_converter("utils.video_to_json", ("audio_to_json", "media_to_json", "video_to_json", "convert"), path)
    return normalize_converter_output(output, source_name)


def events_from_video_file(path: Path, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    output = call_converter("utils.video_to_json", ("video_to_json", "media_to_json", "convert"), path)
    return normalize_converter_output(output, source_name)


def file_to_mechanism_events(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    suffix = path.suffix.lower()
    source_name = path.name

    if suffix in TEXT_SOURCE_SUFFIXES:
        return events_from_text_file(path, source_name)
    if suffix in IMAGE_SOURCE_SUFFIXES:
        return events_from_image_file(path, source_name)
    if suffix in AUDIO_SOURCE_SUFFIXES:
        return events_from_audio_file(path, source_name)
    if suffix in VIDEO_SOURCE_SUFFIXES:
        return events_from_video_file(path, source_name)

    supported = ", ".join(sorted(SUPPORTED_SOURCE_SUFFIXES))
    raise ValueError(f"不支持的文件类型: {suffix or '无后缀'}，支持: {supported}")


def load_mechanism_events(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    suffix = path.suffix.lower()
    source_name = path.name
    if suffix not in SUPPORTED_SOURCE_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SOURCE_SUFFIXES))
        raise ValueError(f"不支持的文件类型: {suffix or '无后缀'}，支持: {supported}")

    mechanism_events, chunks_count = file_to_mechanism_events(path)
    summary = {
        "source_file": str(path),
        "source_name": source_name,
        "source_type": suffix.lstrip("."),
        "chunks_count": chunks_count,
        "events_count": len(mechanism_events),
    }
    return mechanism_events, summary


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
    cleaned: Dict[str, Any] = {}
    for key, value in properties.items():
        normalized_value = to_neo4j_value(value)
        if normalized_value in ("", None, []):
            continue
        cleaned[key] = normalized_value
    return cleaned


def to_neo4j_value(value: Any) -> Any:
    if value in ("", None, []):
        return None
    if isinstance(value, list):
        items: List[Any] = []
        item_types = set()
        for item in value:
            normalized_item = to_neo4j_value(item)
            if normalized_item in ("", None, []):
                continue
            if isinstance(normalized_item, list):
                normalized_item = str(normalized_item)
            if isinstance(normalized_item, dict):
                normalized_item = json.dumps(normalized_item, ensure_ascii=False)
            items.append(normalized_item)
            item_types.add(type(normalized_item))
        if not items:
            return None
        if len(item_types) > 1:
            return [str(item) for item in items]
        return items
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def get_entity_info(graph: Any, entity_name: str) -> Dict[str, Any]:
    query = """
    MATCH (e {name: $name})
    RETURN properties(e) as props
    """
    result = graph.run(query, name=entity_name).data()
    return result[0]["props"] if result else {}


def search_knowledge(graph: Any, entity_name: str) -> Dict[str, Any]:
    start_time = time.time()
    result = {
        "target_entity": entity_name,
        "entity_info": None,
        "related_entities": [],
        "relationships": [],
        "triples": [],
        "processing_time_ms": None,
    }

    query = """
    MATCH (n {name: $name})
    OPTIONAL MATCH (n)-[r_out]->(m_out)
    OPTIONAL MATCH (m_in)-[r_in]->(n)
    RETURN
        labels(n) as target_labels,
        n as target,
        collect(DISTINCT {type: type(r_out), to: m_out.name, direction: 'outgoing'}) as outgoing,
        collect(DISTINCT {type: type(r_in), from: m_in.name, direction: 'incoming'}) as incoming
    """
    result_data = graph.run(query, name=entity_name).data()

    if not result_data or not result_data[0]["target"]:
        result["error"] = f"未找到实体: {entity_name}"
        return result

    target = result_data[0]["target"]
    result["entity_info"] = {
        "name": target.get("name"),
        "labels": list(target.labels) if hasattr(target, "labels") else ["Entity"],
        "properties": dict(target),
    }

    related_set = set()
    for rel in result_data[0]["outgoing"]:
        if rel.get("to"):
            triple = {"from": entity_name, "type": rel["type"], "to": rel["to"]}
            result["triples"].append(triple)
            result["relationships"].append(triple)
            if rel["to"] not in related_set:
                related_set.add(rel["to"])
                result["related_entities"].append({
                    "name": rel["to"],
                    "properties": get_entity_info(graph, rel["to"]),
                    "relation": rel["type"],
                })

    for rel in result_data[0]["incoming"]:
        if rel.get("from"):
            triple = {"from": rel["from"], "type": rel["type"], "to": entity_name}
            result["triples"].append(triple)
            result["relationships"].append(triple)
            if rel["from"] not in related_set:
                related_set.add(rel["from"])
                result["related_entities"].append({
                    "name": rel["from"],
                    "properties": get_entity_info(graph, rel["from"]),
                    "relation": rel["type"],
                    "direction": "incoming",
                })

    result["processing_time_ms"] = (time.time() - start_time) * 1000
    return result


def normalize_name(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("“", "").replace("”", "").replace('"', "").replace("'", "")
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def truncate_text(value: Any, max_chars: int = 240) -> str:
    text = normalize_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return f"{text[:max_chars]}...(已截断{omitted}字)"


def compact_for_prompt(
    value: Any,
    max_list_items: int = 10,
    max_dict_items: int = 16,
    max_text_chars: int = 240,
    max_depth: int = 4,
    _depth: int = 0,
) -> Any:
    if value in (None, "", [], {}):
        return value
    if _depth >= max_depth:
        if isinstance(value, (dict, list)):
            return truncate_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), max_text_chars)
        return truncate_text(value, max_text_chars) if isinstance(value, str) else value
    if isinstance(value, str):
        return truncate_text(value, max_text_chars)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        limit = max(int(max_list_items), 0)
        output = [
            compact_for_prompt(
                item,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_text_chars=max_text_chars,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
            for item in value[:limit]
        ]
        omitted = len(value) - limit
        if omitted > 0:
            output.append({"_omitted_count": omitted})
        return output
    if isinstance(value, dict):
        limit = max(int(max_dict_items), 0)
        output: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= limit:
                break
            output[str(key)] = compact_for_prompt(
                item,
                max_list_items=max_list_items,
                max_dict_items=max_dict_items,
                max_text_chars=max_text_chars,
                max_depth=max_depth,
                _depth=_depth + 1,
            )
        omitted = len(value) - limit
        if omitted > 0:
            output["_omitted_count"] = omitted
        return output
    return truncate_text(value, max_text_chars)


def compact_properties_for_prompt(
    properties: Dict[str, Any],
    preferred_keys: Optional[Iterable[str]] = None,
    max_items: int = 12,
    max_text_chars: int = 160,
) -> Dict[str, Any]:
    if not isinstance(properties, dict) or not properties:
        return {}

    default_keys = (
        "name",
        "type",
        "model",
        "quantity",
        "side",
        "organization",
        "location",
        "event_topic",
        "core_behavior",
        "time",
        "coordinates",
        "altitude",
        "mission",
        "status",
        "purpose",
        "risk_level",
        "desc",
    )
    selected_keys: List[str] = []
    for key in list(preferred_keys or default_keys) + list(default_keys) + list(properties.keys()):
        if key in properties and key not in selected_keys and properties.get(key) not in ("", None, [], {}):
            selected_keys.append(key)
        if len(selected_keys) >= max_items:
            break

    compacted = {
        key: compact_for_prompt(
            properties[key],
            max_list_items=5,
            max_dict_items=6,
            max_text_chars=max_text_chars,
            max_depth=2,
        )
        for key in selected_keys
    }
    omitted = len([key for key, item in properties.items() if item not in ("", None, [], {})]) - len(selected_keys)
    if omitted > 0:
        compacted["_omitted_property_count"] = omitted
    return compacted


def compact_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fit_prompt_context(value: Any, max_chars: int = 18000) -> Any:
    profiles = (
        (12, 16, 220, 4),
        (8, 12, 160, 4),
        (5, 8, 120, 3),
        (3, 6, 80, 3),
        (2, 4, 60, 2),
    )
    for max_list_items, max_dict_items, max_text_chars, max_depth in profiles:
        compacted = compact_for_prompt(
            value,
            max_list_items=max_list_items,
            max_dict_items=max_dict_items,
            max_text_chars=max_text_chars,
            max_depth=max_depth,
        )
        if len(compact_json_dumps(compacted)) <= max_chars:
            return compacted

    text = compact_json_dumps(compacted)
    return {
        "_truncated": True,
        "summary": truncate_text(text, max(max_chars - 100, 100)),
    }


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = normalize_text(text)
    if not raw:
        return {}
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def dedup_list(values: Iterable[Any]) -> List[Any]:
    seen = set()
    output = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, dict) else str(value)
        if key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def format_scalar(value: Any) -> str:
    if isinstance(value, list):
        return "、".join(format_scalar(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def summarize_properties(properties: Dict[str, Any]) -> str:
    if not properties:
        return ""
    preferred_keys = [
        "name",
        "type",
        "model",
        "quantity",
        "side",
        "organization",
        "location",
        "event_topic",
        "coordinates",
        "altitude",
        "caliber",
        "range",
        "accuracy",
        "response_time",
        "duration",
        "speed",
        "height",
        "distance",
    ]
    parts = []
    for key in preferred_keys:
        if key in properties and properties[key] not in ("", None, []):
            parts.append(f"{key}={format_scalar(properties[key])}")
    if not parts:
        for key, value in list(properties.items())[:5]:
            if value not in ("", None, []):
                parts.append(f"{key}={format_scalar(value)}")
    return "；".join(parts)


def format_value_list(value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "无"
        return "、".join(format_scalar(item) for item in value)
    if value in ("", None):
        return "无"
    return format_scalar(value)


def format_mapping(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "无"
        return "；".join(f"{key}:{format_scalar(item)}" for key, item in value.items())
    return format_scalar(value)


def format_nested_item(value: Any) -> str:
    if isinstance(value, dict):
        if not value:
            return "无"
        pairs = []
        for key, item in value.items():
            if item in ("", None, [], {}):
                continue
            pairs.append(f"{key}:{format_scalar(item)}")
        return "；".join(pairs) if pairs else "无"
    if isinstance(value, list):
        if not value:
            return "无"
        return "；".join(format_nested_item(item) for item in value)
    if value in ("", None):
        return "无"
    return format_scalar(value)


def extract_place_name(node: Any) -> str:
    if getattr(node, "label", "") not in {"地点", "place"}:
        return getattr(node, "name", "")
    props = getattr(node, "properties", {}) or {}
    for key in ("normalized_place", "name", "地点", "规范化地名", "名称"):
        value = props.get(key)
        if value:
            return str(value)
    return getattr(node, "name", "")


def parse_date_like(text: str) -> Optional[datetime]:
    raw = normalize_text(text)
    if not raw:
        return None
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if match:
        try:
            return datetime.strptime(match.group(0), "%Y-%m-%d")
        except ValueError:
            pass
    match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", raw)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return None
    match = re.search(r"(\d{4})年(\d{1,2})月", raw)
    if match:
        try:
            return datetime(int(match.group(1)), int(match.group(2)), 1)
        except ValueError:
            return None
    return None


def format_result(result: Any) -> str:
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    return json.dumps(result, ensure_ascii=False, indent=2)
