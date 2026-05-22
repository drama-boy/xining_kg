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


DEFAULT_SUPPORTED_SUFFIXES = normalize_suffixes(KG_CONFIG.get("supported_suffixes", [".csv", ".docx", ".json", ".pdf", ".txt"]))
JSON_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("json_suffixes", [".json"]))
TEXT_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("text_suffixes", [".txt", ".csv", ".pdf", ".docx"]))
IMAGE_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("image_suffixes", [".png", ".jpg", ".jpeg"]))
AUDIO_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("audio_suffixes", [".mp3", ".wav", ".m4a"]))
VIDEO_SOURCE_SUFFIXES = normalize_suffixes(KG_CONFIG.get("video_suffixes", [".mp4", ".avi", ".mov", ".mkv"]))
SUPPORTED_SOURCE_SUFFIXES = tuple(dict.fromkeys(
    DEFAULT_SUPPORTED_SUFFIXES
    + JSON_SOURCE_SUFFIXES
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


def normalize_node_properties(properties: Dict[str, Any], label: str, source_file: str) -> Dict[str, Any]:
    key_map = {
        "类型": "type",
        "名称": "name",
        "来源文件": "from_file",
        "经纬度": "coordinates",
        "海拔": "altitude",
        "军衔": "rank",
        "职务": "position",
        "数量": "quantity",
        "型号": "model",
        "事件主题": "event_topic",
        "核心行为": "core_behavior",
        "述谓结构": "predicate",
        "目标体系": "target_system",
        "地点信息": "location_info",
        "人员信息": "personnel_info",
        "风险等级": "risk_level",
    }

    result: Dict[str, Any] = {}
    for key, value in (properties or {}).items():
        if value in ("", None, []):
            continue
        result[key_map.get(str(key), str(key))] = value

    if not clean_text(result.get("type")):
        result["type"] = label
    result["name"] = clean_text(result.get("name"))
    if source_file:
        result["from_file"] = merge_values(result.get("from_file"), source_file)
    if "coordinates" not in result and result.get("location"):
        result["coordinates"] = result.pop("location")
    if not clean_text(result.get("desc")):
        for candidate_key in ("core_behavior", "event_topic", "predicate", "model", "time", "level", "name"):
            candidate = clean_text(result.get(candidate_key))
            if candidate:
                result["desc"] = candidate
                break
    return {key: value for key, value in result.items() if value not in ("", None, [])}


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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    if suffix == ".txt":
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


def is_mechanism_event(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    return any(key in record for key in ("述谓结构", "目标体系", "地点信息", "核心行为", "风险等级"))


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


def json_data_to_text(data: Any) -> str:
    records = extract_json_records(data)
    if not records:
        return json.dumps(data, ensure_ascii=False, indent=2)

    sections: List[str] = []
    for index, record in enumerate(records, start=1):
        title = clean_text(record.get("title")) or clean_text(record.get("标题")) or f"record {index}"
        content = (
            clean_text(record.get("content"))
            or clean_text(record.get("text"))
            or clean_text(record.get("正文"))
            or clean_text(record.get("内容"))
        )
        if not content:
            content = json.dumps(record, ensure_ascii=False)
        sections.append(f"{title}\n{content}")
    return "\n\n".join(sections)


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


def events_from_json_file(path: Path, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    data = load_json(path)
    records = extract_json_records(data)
    mechanism_events = attach_source([item for item in records if is_mechanism_event(item)], source_name)
    if mechanism_events:
        return mechanism_events, 0

    news_listdict_in = make_news_chunks(source_name, json_data_to_text(data))
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

    records = extract_json_records(output)
    mechanism_events = [item for item in records if is_mechanism_event(item)]
    if mechanism_events:
        return attach_source(mechanism_events, source_name), chunks_count

    if isinstance(output, str):
        news_listdict_in = make_news_chunks(source_name, output)
        chunks_count = len(news_listdict_in)
        return attach_source(extract_events_with_grapher(news_listdict_in), source_name), chunks_count

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

        text = json_data_to_text(records)
        news_listdict_in = make_news_chunks(source_name, text)
        chunks_count = len(news_listdict_in)
        return attach_source(extract_events_with_grapher(news_listdict_in), source_name), chunks_count

    return [], chunks_count


def events_from_image_file(path: Path, source_name: str) -> Tuple[List[Dict[str, Any]], int]:
    output = call_converter("utils.images_to_json", ("images_to_json", "image_to_json", "convert"), path)
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

    if suffix in JSON_SOURCE_SUFFIXES:
        return events_from_json_file(path, source_name)
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
    preferred_keys = ["name", "type", "model", "quantity", "event_topic", "coordinates", "altitude"]
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
