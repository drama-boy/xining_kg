import os
from datetime import datetime

import yaml
from loguru import logger


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


init_config = read_config("./config/config.yaml")

log_config = init_config.get("log", {})
log_dir = log_config.get("dir", "./logs")
log_prefix = log_config.get("file_prefix", "app_server")
date_format = log_config.get("date_format", "YYYY_MM_DD")
python_date_format = date_format.replace("YYYY", "%Y").replace("MM", "%m").replace("DD", "%d")
log_file = log_config.get("api_log_file") or f"{log_prefix}_{datetime.now().strftime(python_date_format)}.log"
os.makedirs(log_dir, exist_ok=True)

logger.add(
    os.path.join(log_dir, log_file),
    level=log_config.get("level", "INFO"),
    rotation=log_config.get("rotation", "10 MB"),
    retention=log_config.get("retention", "30 days"),
)
