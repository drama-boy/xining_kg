# -*- coding: utf-8 -*-
from __future__ import annotations

import base64
import json
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from config import init_config, logger


ROOT = Path(__file__).resolve().parents[1]
FILES_CONFIG = init_config.get("files", {})
KG_CONFIG = init_config.get("knowledge_graph", {})
LLM_CONFIG = init_config.get("llm", {})


def _resolve_path(value: Any, fallback: Path) -> Path:
    if value in (None, ""):
        return fallback
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _normalize_suffixes(values: Iterable[Any]) -> tuple[str, ...]:
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


TEMP_DIR = _resolve_path(FILES_CONFIG.get("temp_dir"), ROOT / "data" / "temp")
OUTPUT_DIR = _resolve_path(FILES_CONFIG.get("output_dir"), ROOT / "data" / "output")
DEFAULT_IMAGE_PATH = TEMP_DIR / "example.jpg"
DEFAULT_YOLO_JSON_PATH = OUTPUT_DIR / "yolo_json_example.json"
YOLO_MODEL_PATH = _resolve_path(KG_CONFIG.get("yolo_model_path"), ROOT / "utils" / "image_model.pt")
IMAGE_SUFFIXES = _normalize_suffixes(KG_CONFIG.get("image_suffixes", [".png", ".jpg", ".jpeg"]))
API_BASE = str(LLM_CONFIG.get("base_url", "http://192.168.1.42:30002/v1")).rstrip("/")
MODEL_NAME = str(LLM_CONFIG.get("model", "/mnt/gemma-4-26B-A4B-it"))


SYSTEM_PROMPT = """
你是军事视觉信息抽取专家。现在会同时给你一张图片和 YOLO 检测的 JSON。
请以图片内容为主、YOLO 检测结果为辅助，抽取事件、实体、属性和关系，输出可直接用于知识图谱构建的标准 JSON 数组。

要求：
1. 只返回 JSON 数组，不要解释、Markdown 或代码块。
2. 字段名、层级和数据类型必须与模板一致。
3. 只抽取图片或 YOLO JSON 中能直接确认的信息；未知字段填 ""、null 或 []，不要编造。
4. 坐标、海拔、类别、置信度、bbox、速度、帧号等信息优先使用 YOLO JSON。
5. 同一实体的属性尽量合并到同一个对象中。
6. YOLO 权重可能只识别少量英文类别，例如 car、trunk、tank、camp、bunker；这些类别只用于定位和参考，不能直接写入“类型”字段。所有“类型”字段必须使用中文语义类型，例如 car 转为“车辆”或“装甲车”，trunk 转为“运输车”，tank 转为“坦克”，camp 转为“营地”，bunker 转为“掩体”或更具体的标准掩体类型。英文 YOLO 类别只能放在“关联关系”的“属性”或“证据”中。
7. 风险等级必须按判别规则填写：高危表示出现攻击/打击/开火/命中/伏击/战斗部署，或出现坦克、装甲车、火炮、导弹、弹药库、油库、指挥部等关键武器或高价值军事目标，或多个武装目标集中并具有明显作战威胁；中危表示出现侦察、巡逻、集结、运输、营地、防御工事、掩体、车辆停放、保障活动等军事活动或潜在威胁，但未见明确攻击行为；低危表示仅有静态、零散、低威胁目标或背景设施，缺少武装行动、敏感目标和紧张态势。证据不足时选择较低等级，不要无依据升高风险。
8. “时间”字段只能填写图片画面中清晰可见的时间，或输入数据中明确标注为拍摄时间/事件发生时间的值。严禁使用系统当前日期、当前时间、文件生成时间、文件修改时间、文件名中的日期、YOLO 推理时间、YOLO JSON 生成时间或处理流水时间作为事件发生时间。没有明确时间证据时，“时间”必须填 []，不要填今天日期、当前年份或任何推测日期。
9. “类型”字段优先使用以下13个标准类别：反坦克锥、弹药库、油库、指挥部、有限掩体、加农炮、圆周掩体、坦克、正方型掩体、C型掩体、装甲车、运输车、堑壕。
    如果标准类别与其他类型存在包含关系、并列关系或重合关系，必须以这13个标准类别为准；不冲突的其他类型可保留为更细补充。
10. 所有输出对象中名为“类型”的字段，最终都会作为知识图谱节点属性 type 使用，必须为中文，不得出现 car、trunk、truck、tank、camp、bunker 等英文检测标签。
11. 禁止把画面构图区域或视觉描述词当作知识图谱节点，例如“画面前景”“画面中景”“前景区域”“中景区域”“远景”“背景”“左侧区域”“右侧区域”“图像中心”等。此类词只能作为目标对象的“位置”“方位”或关系“证据”中的描述，不能作为“名称”、地点名称、组织名称、头实体或尾实体。
"""


OUTPUT_TEMPLATE: Dict[str, Any] = {
    "事件主题": "",
    "时间": [],
    "地点信息": {
        "规范化地名": "",
        "经纬度": "",
        "海拔": None,
    },
    "地点列表": [],
    "人员信息": [],
    "组织机构": [],
    "目标体系": {
        "工事": [],
        "车辆": [],
        "兵种": [],
        "舰船": [],
        "航空器": [],
        "装备设备": [],
        "设施": [],
        "组织机构": [],
        "人员": [],
    },
    "关联关系": [],
    "核心行为": "",
    "风险等级": "",
}


OUTPUT_SCHEMA = """
[
  {
    "事件主题": "4到15字概括核心事件",
    "时间": [],
    "地点信息": {
      "规范化地名": "主地点全称",
      "经纬度": "主地点坐标，例如 29.250936, 113.127266",
      "海拔": null
    },
    "地点列表": [
      {"名称": "", "经纬度": "", "海拔": null, "方位": "", "距离": ""}
    ],
    "人员信息": [
      {"姓名": "", "军衔": "", "职务": "", "所属方": "", "所属组织": "", "指挥对象": ""}
    ],
    "组织机构": [
      {"名称": "", "类型": "", "所属方": "", "上级组织": "", "位置": "", "任务": ""}
    ],
    "目标体系": {
      "工事": [
        {"名称": "", "类型": "", "数量": null, "所属方": "", "位置": "", "经纬度": "", "状态": "", "用途": ""}
      ],
      "车辆": [
        {"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "经纬度": "", "海拔": null, "任务": "", "速度": "", "运动方向": "", "状态": "", "用途": ""}
      ],
      "兵种": [
        {"名称": "", "类型": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "任务": "", "装备": ""}
      ],
      "舰船": [
        {"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "任务": ""}
      ],
      "航空器": [
        {"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "经纬度": "", "海拔": null, "任务": "", "速度": "", "运动方向": "", "状态": "", "用途": ""}
      ],
      "装备设备": [
        {"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "经纬度": "", "任务": "", "状态": "", "用途": ""}
      ],
      "设施": [
        {"名称": "", "类型": "", "数量": null, "所属方": "", "位置": "", "经纬度": "", "距离": "", "用途": ""}
      ],
      "组织机构": [
        {"名称": "", "类型": "", "所属方": "", "上级组织": "", "位置": "", "任务": ""}
      ],
      "人员": [
        {"姓名": "", "军衔": "", "职务": "", "所属方": "", "所属组织": ""}
      ]
    },
    "关联关系": [
      {"头实体": "", "关系": "", "尾实体": "", "属性": {}, "证据": ""}
    ],
    "核心行为": "识别、部署、打击、协同、指挥、前移、伏击等核心动作",
    "风险等级": "高危/中危/低危"
  }
]
"""


def _is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _collect_image_paths(path: Any = None) -> List[Path]:
    source = Path(path) if path not in (None, "") else DEFAULT_IMAGE_PATH
    if source.is_dir():
        images = [item for item in sorted(source.iterdir()) if _is_image_path(item)]
        if not images:
            raise FileNotFoundError(f"图片目录中未找到支持的图片: {source}")
        return images
    if not source.exists():
        raise FileNotFoundError(f"图片文件不存在: {source}")
    if not _is_image_path(source):
        supported = ", ".join(sorted(IMAGE_SUFFIXES))
        raise ValueError(f"不支持的图片文件类型: {source.suffix or '无后缀'}，支持: {supported}")
    return [source]


def _image_to_base64(path: Path) -> str:
    with path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "image/jpeg"


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {"data": data}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def _safe_stem(path: Path) -> str:
    stem = "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in path.stem)
    return stem or "image"


def _yolo_json_path_for(image_path: Path, total_images: int = 1) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"yolo_json_{_safe_stem(image_path)}_{timestamp}.json"


def _bbox_to_ints(bbox: Any) -> List[int]:
    return [int(round(float(value))) for value in bbox]


def _format_timestamp(seconds: float) -> str:
    minutes = int(seconds) // 60
    second = int(seconds) % 60
    return f"{minutes}:{second:02d}"


def _summarize_detected_classes(targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for target in targets:
        class_name = str(target.get("best_class_name") or "").strip()
        if not class_name:
            continue
        item = summary.setdefault(
            class_name,
            {
                "class_name": class_name,
                "count": 0,
                "max_confidence": None,
            },
        )
        item["count"] += 1
        confidence = target.get("best_confidence")
        if confidence is not None and (
            item["max_confidence"] is None or float(confidence) > float(item["max_confidence"])
        ):
            item["max_confidence"] = float(confidence)
    return list(summary.values())


def _compact_yolo_target(target: Dict[str, Any]) -> Dict[str, Any]:
    class_name = target.get("best_class_name")
    compact = {
        "id": target.get("global_target_id"),
        "name": class_name,
        "track_id": target.get("display_name") or target.get("display_id"),
        "class_id": target.get("best_class_id"),
        "class_name": class_name,
        "confidence": target.get("best_confidence"),
        "bbox": target.get("best_bbox"),
        "latitude": target.get("best_latitude"),
        "longitude": target.get("best_longitude"),
        "altitude": target.get("best_altitude"),
        "speed": target.get("best_speed"),
        "frame_index": target.get("best_frame_index") if target.get("best_frame_index") is not None else target.get("frame_index"),
        "frame_timestamp": target.get("best_frame_timestamp"),
    }
    return {
        key: value
        for key, value in compact.items()
        if value not in (None, "", [], {})
    }


def _compact_yolo_for_prompt(yolo_data: Dict[str, Any]) -> Dict[str, Any]:
    targets = yolo_data.get("targets") if isinstance(yolo_data, dict) else []
    if not isinstance(targets, list):
        targets = []

    compact_targets = [
        _compact_yolo_target(target)
        for target in targets
        if isinstance(target, dict)
    ]
    compact_targets.sort(
        key=lambda item: float(item.get("confidence") or 0),
        reverse=True,
    )

    detected_classes = yolo_data.get("detected_classes")
    if not detected_classes:
        detected_classes = _summarize_detected_classes(
            [target for target in targets if isinstance(target, dict)]
        )

    return {
        "type": yolo_data.get("type"),
        "task_id": yolo_data.get("task_id"),
        "image_path": yolo_data.get("image_path"),
        "time_field_note": "YOLO 摘要已移除推理/文件生成时间；不得把当前日期或处理时间当作事件发生时间。",
        "detected_class_names": yolo_data.get("detected_class_names") or [
            item.get("class_name") for item in detected_classes if isinstance(item, dict)
        ],
        "detected_classes": detected_classes,
        "target_count": len(compact_targets),
        "targets": compact_targets,
    }


def run_yolo_to_json(
    image_path: Any = None,
    yolo_json_path: Any = None,
    model_path: Any = None,
) -> Dict[str, Any]:
    image = Path(image_path) if image_path not in (None, "") else DEFAULT_IMAGE_PATH
    output = Path(yolo_json_path) if yolo_json_path not in (None, "") else _yolo_json_path_for(image)
    model = Path(model_path) if model_path not in (None, "") else YOLO_MODEL_PATH

    if not image.exists():
        raise FileNotFoundError(f"图片文件不存在: {image}")
    if not model.exists():
        raise FileNotFoundError(f"YOLO 权重文件不存在: {model}")

    from ultralytics import YOLO

    yolo_model = YOLO(str(model))
    results = yolo_model(str(image), verbose=False)
    result = results[0] if results else None
    names = getattr(result, "names", {}) if result is not None else {}
    boxes = getattr(result, "boxes", None) if result is not None else None
    now = datetime.now().isoformat()

    targets: List[Dict[str, Any]] = []
    if boxes is not None:
        xyxy = boxes.xyxy.cpu().tolist() if getattr(boxes, "xyxy", None) is not None else []
        confs = boxes.conf.cpu().tolist() if getattr(boxes, "conf", None) is not None else []
        classes = boxes.cls.cpu().tolist() if getattr(boxes, "cls", None) is not None else []
        for index, bbox in enumerate(xyxy, start=1):
            class_id = int(classes[index - 1]) if index - 1 < len(classes) else -1
            class_name = str(names.get(class_id, class_id))
            confidence = float(confs[index - 1]) if index - 1 < len(confs) else None
            target = {
                "global_target_id": index,
                "display_name": f"{class_name}{index}",
                "best_latitude": None,
                "best_longitude": None,
                "best_altitude": None,
                "best_confidence": confidence,
                "best_class_id": class_id,
                "best_class_name": class_name,
                "best_bbox": _bbox_to_ints(bbox),
                "best_speed": None,
                "best_frame_index": 0,
                "best_frame_timestamp": 0.0,
                "last_time": now,
                "observations": [
                    {
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name,
                        "display_name": class_name,
                        "bbox": _bbox_to_ints(bbox),
                        "speed_mps": None,
                        "latitude": None,
                        "longitude": None,
                        "altitude": None,
                        "timestamp": now,
                        "frame_index": 0,
                        "frame_timestamp": 0.0,
                        "image_path": str(image),
                    }
                ],
                "display_id": f"{class_name}_{index}",
                "image_path": str(image),
                "frame_index": 0,
                "timestamp_seconds": 0.0,
                "timestamp": _format_timestamp(0.0),
            }
            targets.append(target)

    yolo_data = {
        "type": "global_targets",
        "task_id": image.stem,
        "timestamp": now,
        "image_path": str(image),
        "model_path": str(model),
        "detected_class_names": [item["class_name"] for item in _summarize_detected_classes(targets)],
        "detected_classes": _summarize_detected_classes(targets),
        "targets": targets,
    }
    _write_json(output, yolo_data)
    return yolo_data


def _load_or_run_yolo(image_path: Path, yolo_json_path: Path) -> Dict[str, Any]:
    try:
        return run_yolo_to_json(image_path=image_path, yolo_json_path=yolo_json_path)
    except ImportError as exc:
        if yolo_json_path.exists():
            logger.warning("{}；使用已有 YOLO JSON: {}", exc, yolo_json_path)
            return _read_json(yolo_json_path)
        raise


def _extract_json_text(llmresult: str) -> str:
    text = (llmresult or "").strip()
    if not text:
        return ""

    if "```json" in text:
        start = text.find("```json") + len("```json")
        end = text.rfind("```")
        return text[start:end].strip() if end > start else text[start:].strip()

    if "```" in text:
        start = text.find("```") + len("```")
        end = text.rfind("```")
        return text[start:end].strip() if end > start else text[start:].strip()

    array_start = text.find("[")
    array_end = text.rfind("]")
    if array_start != -1 and array_end > array_start:
        return text[array_start:array_end + 1].strip()

    object_start = text.find("{")
    object_end = text.rfind("}")
    if object_start != -1 and object_end > object_start:
        return text[object_start:object_end + 1].strip()

    return ""


def _deepcopy_template() -> Dict[str, Any]:
    return json.loads(json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False))


def _merge_template(item: Dict[str, Any]) -> Dict[str, Any]:
    merged = _deepcopy_template()
    for key, value in item.items():
        if key == "目标体系" and isinstance(value, dict):
            merged["目标体系"].update(value)
        elif key == "地点信息" and isinstance(value, dict):
            merged["地点信息"].update(value)
        else:
            merged[key] = value
    return merged


class graph_cls:
    def llmstr2list(self, llmresult: str) -> List[Dict[str, Any]]:
        json_text = _extract_json_text(llmresult)
        if not json_text:
            return []

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            logger.warning("图片 JSON 解析失败: {}, 原始内容: {}", exc, json_text[:200])
            return []

        records = data if isinstance(data, list) else [data]
        return [_merge_template(item) for item in records if isinstance(item, dict)]

    def _build_prompt(self, image_path: Path, yolo_data: Dict[str, Any]) -> str:
        yolo_prompt_data = _compact_yolo_for_prompt(yolo_data)
        yolo_text = json.dumps(yolo_prompt_data, ensure_ascii=False, indent=2)
        class_names = yolo_prompt_data.get("detected_class_names") or []
        class_summary = yolo_prompt_data.get("detected_classes") or []
        return f"""
请结合图片和 YOLO 检测结果抽取军事相关知识图谱事件。

图片文件名：{image_path.name}
YOLO 检测类别：{json.dumps(class_names, ensure_ascii=False)}
YOLO 类别统计：{json.dumps(class_summary, ensure_ascii=False)}
YOLO 关键信息摘要：
{yolo_text}

抽取规则：
1. YOLO 的 best_class_name/class_name 只能作为检测标签参考，不能原样作为目标“类型”；输出中的所有“类型”字段必须使用中文语义类型，best_bbox/bbox 可作为证据属性。
2. 如果 YOLO JSON 中存在 best_latitude、best_longitude、best_altitude，请填入地点或目标经纬度、海拔。
3. 输出节点或目标对象的“名称”必须使用类别名或真实语义名称，不能使用 car12、trunk3、tank2、camp1 这种带检测序号的临时 ID；“类型”字段必须使用中文，不要写 car、trunk、tank、camp、bunker。
4. 如果需要保留检测编号，只能放到“关联关系”的“属性”或“证据”中，不能作为节点名称。
5. YOLO 类别只有少数几类时，不要只机械输出类别清单；要根据图片内容自主判断实体间关系，例如“部署于”“接近”“防护”“威胁”“隶属/组成”“位于营地内”等。
6. 如果图片内容和 YOLO 类别冲突，以图片可见内容为准，并在证据中保留 YOLO 类别。
7. 关联关系必须尽量显式抽取，至少在能判断时建立目标与位置、目标与设施/营地、目标之间的空间或作战关系。
8. “时间”字段只能来自图片中清晰可见的时间，或输入数据中明确说明为拍摄时间/事件发生时间的值；YOLO 推理时间、YOLO JSON 生成时间、处理时间、文件生成/修改时间、文件名日期、系统当前日期都不是事件发生时间，严禁填写到“时间”字段。没有明确时间证据时必须输出 "时间": []。
9. 风险等级必须按以下依据判别：高危=明确攻击/打击/开火/命中/伏击/战斗部署，或坦克、装甲车、火炮、导弹、弹药库、油库、指挥部等关键武器/高价值目标，或多个武装目标集中形成明显作战威胁；中危=侦察、巡逻、集结、运输、营地、防御工事、掩体、车辆停放、保障活动等军事活动或潜在威胁，但没有明确攻击行为；低危=静态、零散、低威胁目标或背景设施，且缺少武装行动和敏感目标。证据不足时选择较低等级。
10. 英文 YOLO 类别到中文类型的参考映射：car=车辆/装甲车，trunk=运输车，truck=运输车，tank=坦克，camp=营地，bunker=掩体/有限掩体/圆周掩体/正方型掩体/C型掩体。能判断为13个标准类别时优先使用标准类别。
11. 不要把画面区域或构图描述抽取成节点或地点，例如“画面前景”“画面中景”“前景和中景”“背景区域”“图像左侧”“图像中心”等不能出现在“名称”、地点名称、组织名称、头实体或尾实体中；如需表达空间位置，只能写入目标对象的“位置”“方位”字段，或写入关系“证据”。
12. 上面的 YOLO 信息已经是摘要，不包含冗余明细字段；不要要求读取完整文件。
13. 只输出 JSON 数组。

输出模板：
{OUTPUT_SCHEMA}
"""

    def _call_vision_llm(self, image_path: Path, prompt: str) -> str:
        image_b64 = _image_to_base64(image_path)
        mime_type = _guess_mime_type(image_path)
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_b64}",
                            },
                        },
                    ],
                },
            ],
            "temperature": 0,
            "max_tokens": 4096,
        }
        response = requests.post(f"{API_BASE}/chat/completions", json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

    def _process_image(self, item: Any, total_images: int = 1) -> List[Dict[str, Any]]:
        if isinstance(item, dict):
            raw_path = item.get("path") or item.get("image_path") or item.get("file_path") or item.get("content")
            raw_yolo_json_path = item.get("yolo_json_path")
        else:
            raw_path = item
            raw_yolo_json_path = None

        if not raw_path:
            return []

        image_path = Path(raw_path)
        yolo_json_path = Path(raw_yolo_json_path) if raw_yolo_json_path else _yolo_json_path_for(image_path, total_images)

        try:
            if raw_yolo_json_path and yolo_json_path.exists():
                yolo_data = _read_json(yolo_json_path)
            else:
                yolo_data = _load_or_run_yolo(image_path, yolo_json_path)
            prompt = self._build_prompt(image_path, yolo_data)
            llmresult = self._call_vision_llm(image_path, prompt)
            records = self.llmstr2list(llmresult)
            for record in records:
                record.setdefault("来源文件", image_path.name)
                record.setdefault("from_file", image_path.name)
                record.setdefault("source_image", image_path.name)
                record.setdefault("yolo_json", str(yolo_json_path))
            return records
        except Exception as exc:
            logger.exception("处理图片失败: {}", image_path)
            logger.warning("图片处理异常详情: {}", exc)
            return []

    def abstract_ner(self, image_list: List[Any]) -> List[Dict[str, Any]]:
        total_images = len(image_list)
        abstract_listdict: List[Dict[str, Any]] = []
        for item in image_list:
            abstract_listdict.extend(self._process_image(item, total_images=total_images))
        return abstract_listdict

    def graph_main(self, image_listdict_in: List[Any]) -> List[Dict[str, Any]]:
        return self.abstract_ner(image_listdict_in or [])


def image_to_json(path: Any = None) -> List[Dict[str, Any]]:
    image_paths = _collect_image_paths(path)
    total_images = len(image_paths)
    image_listdict_in = [
        {
            "title": image_path.name,
            "path": str(image_path),
            "yolo_json_path": str(_yolo_json_path_for(image_path, total_images)),
        }
        for image_path in image_paths
    ]
    grapher = graph_cls()
    return grapher.graph_main(image_listdict_in)


def images_to_json(path: Any = None) -> List[Dict[str, Any]]:
    return image_to_json(path)


def convert(path: Any = None) -> List[Dict[str, Any]]:
    return image_to_json(path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="图片+YOLO检测转知识图谱抽取 JSON")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_IMAGE_PATH, help="图片文件或图片目录，默认 data/temp/example.jpg")
    parser.add_argument("--yolo-json", type=Path, default=DEFAULT_YOLO_JSON_PATH, help="YOLO 输出 JSON，默认 data/output/yolo_json_example.json")
    parser.add_argument("--output", type=Path, default=TEMP_DIR / "image_to_json_output.json")
    args = parser.parse_args()

    image_paths = _collect_image_paths(args.path)
    image_listdict_in = [
        {
            "title": image_path.name,
            "path": str(image_path),
            "yolo_json_path": str(args.yolo_json if len(image_paths) == 1 else _yolo_json_path_for(image_path, len(image_paths))),
        }
        for image_path in image_paths
    ]
    output = graph_cls().graph_main(image_listdict_in)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    logger.info("图片抽取完成: {}", args.output)
