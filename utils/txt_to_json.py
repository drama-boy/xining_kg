# -*- coding: utf-8 -*-
import json
import concurrent.futures
from functools import partial
from tqdm import tqdm

from config import init_config, logger
from utils.myllm import llm_gemma_31b

KG_CONFIG = init_config.get("knowledge_graph", {})
DEFAULT_MAX_WORKERS = int(KG_CONFIG.get("max_workers", 5))
DEFAULT_SHOW_PROGRESS = bool(KG_CONFIG.get("show_progress", False))

class graph_cls():
    def llmstr2list(self, llmresult):
        """
        【优化版】大模型输出解析：兼容所有标准JSON格式，健壮性更强
        处理：```json、```、纯JSON、带换行/空格的输出
        """
        if not llmresult:
            return []
        
        #  Step1: 剔除markdown代码块标记
        llmresult = llmresult.strip()
        if '```json' in llmresult:
            # 截取json代码块内容
            start = llmresult.find('```json') + 7
            end = llmresult.rfind('```')
            stri = llmresult[start:end].strip()
        elif '```' in llmresult:
            start = llmresult.find('```') + 3
            end = llmresult.rfind('```')
            stri = llmresult[start:end].strip()
        else:
            # 无代码块，直接提取最外层[]数组
            if '[' in llmresult and ']' in llmresult:
                stri = llmresult[llmresult.find('['):llmresult.rfind(']')+1].strip()
            else:
                return []

        # Step2: 统一转义，兼容单引号/双引号
        stri = stri.replace("'", '"')
        try:
            out = json.loads(stri)
            # 确保返回列表
            return out if isinstance(out, list) else [out]
        except json.JSONDecodeError as e:
            logger.warning("JSON解析失败: {}, 原始内容: {}", e, stri[:100])
            return []

    def _process_passage(self, passage):
        """
        【核心修改】重写提示词：适配军事机理知识图谱抽取
        严格匹配你的需求：述谓结构、目标体系、经纬度、风险等级、时空信息
        """
        prompt = f'''
        任务：从以下军事文本中抽取**事件知识图谱要素**，一篇文本可能包含多个事件或者多个主题，输出严格遵循要求。
        输出格式：仅返回标准JSON数组，格式为 List[dict]，无多余文字、无解释。

        必选抽取字段（严格按要求提取）：
        1. 事件主题：4-10字概括核心事件（如：印巴坦克伏击战、中印边境联合演训）
        2. 时间：标准时间字符串（精确到年月日，无则留空数组）
        3. 地点信息：dict格式，包含3个子字段
           - 规范化地名：全称，禁止简称（如：拉达克地区、克什米尔山谷）
           - 经纬度：坐标字符串（如：33°20′45″N 78°45′32″E）
           - 海拔：数字（单位：米，无则填null）
        4. 人员信息：list[dict]，包含 姓名、军衔、职务（无则留空数组）
        5. 目标体系：dict格式，严格匹配你的分类，包含数量+型号
           - 工事：[{{'类型':'','数量':0}},...]（前沿哨所、碉堡、炮兵阵地等）
           - 车辆：[{{'类型':'','型号':'','数量':0}},...]（主战坦克、无人机、步兵战车等）
           - 兵种：[{{'类型':'','数量':0}},...]（装甲兵、山地步兵旅等）
        6. 风险等级：高危/中危/低危（攻击性装备、对峙事件为高危）
        7. 核心行为：事件核心动作（目标识别、火力引导、伏击击毁、演习部署等）

        待抽取文本：
        {passage}
        '''
        try:
            llmresult = llm_gemma_31b(prompt)
            return self.llmstr2list(llmresult)
        except Exception as e:
            logger.exception("处理文本失败: {}", e)
            return []

    def abstract_ner(self, cutlist):
        """并行处理文本，保留原逻辑不变"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
            func = partial(self._process_passage)
            mapped = executor.map(func, cutlist)
            iterator = tqdm(mapped, total=len(cutlist)) if DEFAULT_SHOW_PROGRESS else mapped
            results = list(iterator)
        
        abstract_listdict = []
        for sublist in results:
            abstract_listdict.extend(sublist)
        return abstract_listdict

    def graph_main(self, news_listdict_in):
        """主入口函数，保留原逻辑不变"""
        cutlist = [news_dict_in['content'] for news_dict_in in news_listdict_in]
        abstract_listdict = self.abstract_ner(cutlist)
        return abstract_listdict


if __name__ == "__main__":
    filepath = "./data/news.json"
    with open(filepath, "r", encoding="utf-8") as f:
        news_listdict_in = json.load(f)
    grapher = graph_cls()
    abstract_listdict = grapher.graph_main(news_listdict_in)
    with open("./data/out-5.20.2.json", "w", encoding="utf-8") as f:
        json.dump(abstract_listdict, f, ensure_ascii=False, indent=1)

