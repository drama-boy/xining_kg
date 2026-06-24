# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config import init_config, logger
from utils.myllm import llm_gemma
from utils.util import clean_text, read_source_file, split_text


SUPPORTED_QUERY_SUFFIXES = init_config.get("related_data", {}).get("supported_query_suffixes", [".csv", ".docx", ".txt", ".pdf", ".md"])
MAX_BATCH_CHARS = init_config.get("related_data", {}).get("max_batch_chars", 12000)
MAX_EVENT_CHARS = init_config.get("related_data", {}).get("max_content_chars", 100)
MAX_RESULTS = init_config.get("related_data", {}).get("max_results", 3)
KG_CONFIG = init_config.get("knowledge_graph", {})
MAX_FILE_WORKERS = KG_CONFIG.get("max_workers", 5)
MAX_LLM_WORKERS = init_config.get("llm", {}).get("query_llm_workers", 3)


def _file_item_parts(item: Any) -> Tuple[Path, str]:
    """解析文件路径和原名。"""
    if isinstance(item, dict):
        path = Path(str(item.get("path") or item.get("file_path") or ""))
        filename = clean_text(item.get("filename")) or path.name
        return path, Path(filename).stem
    path = Path(str(item))
    return path, path.stem


def _trim_text(text: Any, max_chars: int = MAX_EVENT_CHARS) -> str:
    """截取完整短句。"""
    content = re.sub(r"\s+", " ", clean_text(text))
    if not content:
        return ""
    sentence_endings = ("。", "！", "？", ".", "!", "?")
    clause_endings = ("；", ";", "，", ",")
    window = content[:max_chars]
    if len(content) <= max_chars and content.endswith(sentence_endings + clause_endings):
        return content
    strong_positions = [window.rfind(mark) for mark in ("。", "！", "？", ".", "!", "?")]
    strong_cut = max(strong_positions)
    if strong_cut >= 30:
        return window[: strong_cut + 1]
    weak_positions = [window.rfind(mark) for mark in ("；", ";", "，", ",")]
    weak_cut = max(weak_positions)
    if weak_cut >= 30:
        return window[: weak_cut + 1]
    return window.rstrip()


def _normalize_result_item(item: Dict[str, Any]) -> Dict[str, str]:
    """标准化抽取结果字段。"""
    return {
        "file_name": clean_text(item.get("file_name")),
        "time": clean_text(item.get("time")),
        "coordinate": clean_text(item.get("coordinate")),
        "place": clean_text(item.get("place")),
        "event": _trim_text(item.get("event")),
    }


def _parse_llm_results(text: str) -> List[Dict[str, str]]:
    """解析模型返回列表。"""
    raw = clean_text(text)
    if not raw:
        return []
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("[")
    end = raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    results = []
    for item in data:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_result_item(item)
        if normalized["file_name"] and normalized["event"]:
            results.append(normalized)
    return results


def _make_batches(chunks: List[Dict[str, str]]) -> List[List[Dict[str, str]]]:
    """按长度聚合文本块。"""
    batches: List[List[Dict[str, str]]] = []
    current: List[Dict[str, str]] = []
    current_chars = 0
    for chunk in chunks:
        item_chars = len(chunk["content"]) + len(chunk["file_name"]) + 64
        if current and current_chars + item_chars > MAX_BATCH_CHARS:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


def _fallback_search(chunks: List[Dict[str, str]], request: str) -> List[Dict[str, str]]:
    """关键词兜底匹配。"""
    keywords = [word for word in re.split(r"[\s,，。；;、]+", request) if word]
    results = []
    for chunk in chunks:
        if any(keyword in chunk["content"] for keyword in keywords):
            results.append({
                "file_name": chunk["file_name"],
                "time": "",
                "coordinate": "",
                "place": "",
                "event": _trim_text(chunk["content"]),
            })
    return results


def _query_batch(batch: List[Dict[str, str]], request: str) -> List[Dict[str, str]]:
    """调用模型查询批次。"""
    context = "\n\n".join(
        f"[{index}] 文件名:{item['file_name']}\n内容:{item['content']}"
        for index, item in enumerate(batch, start=1)
    )
    prompt = (
        "你是一个信息抽取专家。请从以下给定的多条文本中，抽取每个文本所描述的**实际发生的事件**，并提取五个关键要素：文件名、事件时间、坐标、地点、事件描述。\n"
        "1. **文件名（file_name）**：直接从输入文本块中的“[编号] 文件名:...”获取，无需抽取。\n"
        "2. **事件时间（time）**：事件发生的具体时间（如日期、时刻等）。如果文本中没有明确时间，填空字符串\"\"。\n"
        "3. **坐标（coordinate）**：事件发生位置的经纬度或坐标表示（如“32.1°N, 45.2°E”）。如果文本中没有，填空字符串。\n"
        "4. **地点（place）**：事件发生的地名（如“某高地”、“某区域”）。如果文本中没有，填空字符串。\n"
        "5. **事件描述（event）**：简明扼要描述事件，格式为“[主体] + [执行/进行] + [任务]”，长度控制在50～100个中文字符，以完整句子结尾。注意不要重复描述地点（地点已单独提取）。\n"
        "\n抽取规则：\n"
        "- 只抽取**实际行动事件**（如巡逻、警戒、攻击、运输等），忽略诱因（“因……导致”、“由于……”）、故障现象（“弹匣变形”、“无法击发”）、后果或评估（“可靠性缺陷”、“内部总结”）。\n"
        "- 如果整段文本属于内部总结、评估或非实际行动事件，则**不抽取**，即该条不输出任何对象。\n"
        "- 同一文件可能包含多个独立事件，每个事件生成一个 JSON 对象，全部放入数组。\n"
        "\n输出格式：\n"
        "只返回 JSON 数组，不要解释，不要 Markdown。数组元素格式为：\n"
        "{\"file_name\":\"文件名\",\"time\":\"时间\",\"coordinate\":\"坐标\",\"place\":\"地点\",\"event\":\"事件描述\"}\n"
        "如果没有抽取到任何事件，返回空数组 []。\n"
        "\n示例：\n"
        "输入：\n"
        "\"[1] 文件名:日志A\\n内容:2024年1月15日，印军第28步兵师后勤基地士兵在执行弹药库外围警戒任务时，因INSAS突击步枪在低温环境下出现弹匣变形导致无法顺利上膛。\"\n"
        "输出：\n"
        "[{\"file_name\":\"日志A\",\"time\":\"2024年1月15日\",\"coordinate\":\"\",\"place\":\"弹药库外围\",\"event\":\"印军第28步兵师后勤基地士兵执行警戒任务。\"}]\n"
        "\n请注意：示例中时间明确，地点为“弹药库外围”，坐标缺失，事件描述不含地点。\n\n"
        f"request: {request}\n\n"
        f"文本块:\n{context}"
    )
    llm_text = llm_gemma(prompt)
    return _parse_llm_results(llm_text)


def _load_file_chunks(item: Any) -> Tuple[List[Dict[str, str]], Optional[Dict[str, str]]]:
    """读取单文件文本块。"""
    path, filename = _file_item_parts(item)
    try:
        if path.suffix.lower() not in SUPPORTED_QUERY_SUFFIXES:
            raise ValueError(f"不支持的文件类型: {path.suffix}")
        text = read_source_file(path)
        chunks = [
            {
                "file_name": filename,
                "chunk_id": str(index),
                "content": content,
            }
            for index, content in enumerate(split_text(text, chunk_size=1400, chunk_overlap=120), start=1)
            if content
        ]
        return chunks, None
    except Exception as exc:
        logger.exception("文件问询读取失败: {}", path)
        return [], {"file_name": filename, "error": str(exc)}


def _limit_results(results: List[Dict[str, str]], max_results: int = MAX_RESULTS) -> List[Dict[str, str]]:
    """限制结果并均衡文件。"""
    cleaned = []
    for item in results:
        normalized = _normalize_result_item(item)
        if normalized["file_name"] and normalized["event"]:
            cleaned.append(normalized)
    seen = set()
    deduped = []
    for item in cleaned:
        key = (item["file_name"], item["time"], item["coordinate"], item["place"], item["event"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    selected: List[Dict[str, str]] = []
    selected_files = set()
    remaining: List[Dict[str, str]] = []
    for item in deduped:
        if item["file_name"] not in selected_files and len(selected) < max_results:
            selected.append(item)
            selected_files.add(item["file_name"])
        else:
            remaining.append(item)

    for item in remaining:
        if len(selected) >= max_results:
            break
        selected.append(item)
    return selected


def relevant_wrapper(files: Iterable[Any], request: str) -> Dict[str, Any]:
    """多文件大模型问询。"""
    start_time = time.time()
    request = clean_text(request)
    if not request:
        raise ValueError("request不能为空")

    chunks: List[Dict[str, str]] = []
    failed_files: List[Dict[str, str]] = []
    file_items = list(files or [])
    if len(file_items) > 1 and MAX_FILE_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_FILE_WORKERS, len(file_items))) as executor:
            for file_chunks, failed_file in executor.map(_load_file_chunks, file_items):
                chunks.extend(file_chunks)
                if failed_file:
                    failed_files.append(failed_file)
    else:
        for item in file_items:
            file_chunks, failed_file = _load_file_chunks(item)
            chunks.extend(file_chunks)
            if failed_file:
                failed_files.append(failed_file)

    results: List[Dict[str, str]] = []
    batches = _make_batches(chunks)
    if len(batches) > 1 and MAX_LLM_WORKERS > 1:
        with ThreadPoolExecutor(max_workers=min(MAX_LLM_WORKERS, len(batches))) as executor:
            for batch_results in executor.map(lambda batch: _query_batch(batch, request), batches):
                results.extend(batch_results)
    else:
        for batch in batches:
            results.extend(_query_batch(batch, request))

    if not results:
        results = _fallback_search(chunks, request)

    formatted_results = _limit_results(results)
    return {
        "request": request,
        "files_count": len(file_items),
        "results": formatted_results,
        "failed_files": failed_files,
        "processing_time_s": time.time() - start_time,
    }
