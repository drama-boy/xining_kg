# -*- coding: utf-8 -*-
import json
from openai import OpenAI
import traceback
from config import init_config, logger


def llm_gemma_31b(prompt):     #model in ['qw1.5-32b','qw2-72b-int4','qw2-72b-fp16','qw2-110b-int4','qw2.5-72b']
    answer=''
    try:
        llm_config = init_config.get("llm", {})
        client = OpenAI(
            base_url=llm_config.get("base_url", "http://192.168.1.42:30002/v1"),
            api_key=llm_config.get("api_key", "xxx"),
        )
        response = client.chat.completions.create(
            model=llm_config.get("model", "/mnt/gemma-4-26B-A4B-it"),
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            temperature=llm_config.get("temperature", 0.3)
            )
        answer=response.choices[0].message.content
    except:
        logger.exception("LLM 调用失败")
    return answer
