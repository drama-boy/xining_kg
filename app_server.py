# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import init_config, logger


def get_config(path: List[str], default: Any = None) -> Any:
    value: Any = init_config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


APP_HOST = get_config(["server", "host"], "0.0.0.0")
APP_PORT = get_config(["server", "port"], 5000)
UPLOAD_DIR = Path(get_config(["files", "upload_dir"], "./data/upload"))
OUTPUT_DIR = Path(get_config(["files", "output_dir"], "./data/output"))
STATIC_DIR = Path(get_config(["files", "static_dir"], "./static"))
LOG_DIR = Path(get_config(["log", "dir"], "./logs"))

for directory in (UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title="知识图谱算法 API",
    description="搭建知识图谱，图谱推理检索算法封装服务",
    version="0.1.0",
)
app.openapi_version = "3.0.2"

app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def make_upload_path(upload_file: UploadFile) -> Path:
    suffix = Path(upload_file.filename or "").suffix
    return UPLOAD_DIR / f"upload_{uuid.uuid4().hex}{suffix}"


async def save_upload_file(upload_file: UploadFile, destination: Path) -> Path:
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as buffer:
            while True:
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)
        return destination
    except Exception as exc:
        logger.exception("文件保存失败: {}", exc)
        raise HTTPException(status_code=500, detail=f"文件保存失败: {exc}") from exc


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("服务启动完成")
    logger.info("上传目录: {}", UPLOAD_DIR)
    logger.info("输出目录: {}", OUTPUT_DIR)


@app.post("/api/knowledge_graph/construction",description="支持的文件格式: txt, docx, pdf, csv, json\n")
async def knowledge_graph_construction_api(files: List[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="请上传至少一个文件，字段名为 files")

    saved_paths: List[str] = []
    try:
        for upload_file in files:
            logger.info("收到文件: name={}, type={}", upload_file.filename, upload_file.content_type)
            saved_path = await save_upload_file(upload_file, make_upload_path(upload_file))
            saved_paths.append(str(saved_path))
            logger.info("文件已保存: {}", saved_path)

        from algorithms.knowledge_graph.build_kg import build_knowledge_graph

        result = build_knowledge_graph(saved_paths)
        if not result:
            raise HTTPException(status_code=400, detail="知识图谱构建失败")

        return {
            "code": 200,
            "msg": "success",
            "files_count": len(saved_paths),
            "output_file": result.get("output_file", ""),
            "extracted_file": result.get("extracted_file", ""),
            "events_count": result.get("events_count", 0),
            "nodes_count": result.get("nodes_count", 0),
            "edges_count": result.get("edges_count", 0),
            "processing_time_s": result.get("processing_time_s", ""),
            "triples_count": result.get("triples_count", 0),
            "evaluation": result.get("evaluation", {}),
            "graph_data": result.get("graph", {}),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("construction 处理异常: {}", exc)
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc


@app.post("/api/knowledge_graph/inference")
async def knowledge_graph_inference_api(target: str = Form(...)) -> dict:
    if not target:
        raise HTTPException(status_code=400, detail="缺少 target 参数")

    try:
        from algorithms.knowledge_graph.inference import inference_wrapper

        result = inference_wrapper(target)
        if not result:
            raise HTTPException(status_code=400, detail="inference 算法处理失败")

        return {
            "code": 200,
            "msg": "success",
            "target": result.get("target", target),
            "risk_level": result.get("risk_level", ""),
            "risk_reason": result.get("risk_reason", ""),
            "timeline": result.get("timeline", ""),
            "relatedobj": result.get("relatedobj", ""),
            "activitypattern": result.get("activitypattern", ""),
            "forecast": result.get("forecast", ""),
            "processing_time_s": result.get("processing_time_s", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("inference 处理异常: {}", exc)
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc


@app.post("/api/knowledge_graph/retrieve")
async def knowledge_graph_retrieve_api(query: str = Form(...)) -> dict:
    if not query:
        raise HTTPException(status_code=400, detail="缺少 query 参数")

    try:
        from algorithms.knowledge_graph.retriever import retrieve_wrapper

        result = retrieve_wrapper(query)
        if not result:
            raise HTTPException(status_code=400, detail="retrieve 算法处理失败")

        return {
            "code": 200,
            "msg": "success",
            "target_entity": result.get("target_entity", query),
            "entity_info": result.get("entity_info", ""),
            "relationships": result.get("relationships", ""),
            "processing_time_ms": result.get("processing_time_ms", ""),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("retrieve 处理异常: {}", exc)
        raise HTTPException(status_code=500, detail=f"处理失败: {exc}") from exc


@app.get("/")
async def root() -> Any:
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"code": 200, "msg": "success", "data": "knowledge graph service"})


@app.get("/healthz")
async def health_check() -> dict:
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_server:app", host=APP_HOST, port=APP_PORT, reload=False)
