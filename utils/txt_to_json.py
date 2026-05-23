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
        严格匹配你的需求：实体类型、全面属性、显式关系、经纬度、风险等级、时空信息
        """
        prompt = f'''
        你是军事机理知识图谱抽取助手。请从下面文本中抽取事件、实体、属性和关系。
        一段文本可能包含多个事件、多个地点、多个阵营和多类装备；不要只抽取工事、车辆、兵种。

        输出要求：
        1. 仅返回标准 JSON 数组，格式为 List[dict]；不要输出解释、Markdown、代码块或多余文本。
        2. JSON 必须使用双引号，不能使用单引号。
        3. 只抽取文本中明示或由句式直接指向的信息；未知字段用 ""、null 或 []，不要编造。
        4. 所有实体都要尽量补全属性，尤其是型号、数量、所属方、所属组织、位置、经纬度、海拔、任务、状态、用途、口径、射程、精度(仅当文本中出现可量化指标时，如“命中精度95%”或“误差2米”)、响应时间、耗时、速度、高度、距离、运动方向等。
        5. 同一实体在文本中出现多个属性时，要合并到同一个实体对象中，不要漏掉性能指标。
        6. 列表字段没有内容时输出 []；不要保留空模板对象。

        每个事件对象必须包含以下字段：
        {{
          "事件主题": "4到15字概括核心事件，如：班公湖演习、T90S无人机协同",
          "时间": ["标准时间字符串，尽量精确到年月日或时分"],
          "地点信息": {{
            "规范化地名": "事件主地点全称",
            "经纬度": "主地点坐标，如 33°32′18″N 78°55′42″E",
            "海拔": null
          }},
          "地点列表": [
            {{"名称": "所有出现的地点/机场/阵地/指挥中心/补给中心等", "经纬度": "", "海拔": null, "方位": "", "距离": ""}}
          ],
          "人员信息": [
            {{"姓名": "", "军衔": "", "职务": "", "所属方": "", "所属组织": "", "指挥对象": ""}}
          ],
          "组织机构": [
            {{"名称": "如中方、印军、第114装甲团、第8山地步兵旅", "类型": "", "所属方": "", "上级组织": "", "位置": "", "任务": ""}}
          ],
          "目标体系": {{
            "工事": [
              {{"名称": "", "类型": "", "数量": null, "所属方": "", "位置": "", "经纬度": "", "状态": "", "用途": ""}}
            ],
            "车辆": [
              {{"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "经纬度": "", "海拔": null, "任务": "", "速度": "", "运动方向": "", "口径": "", "射程": "", "精度": "", "响应时间": "", "耗时": "", "用途": ""}}
            ],
            "兵种": [
              {{"名称": "", "类型": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "任务": "", "装备": ""}}
            ],
            "舰艇": [
              {{"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "任务": "", "口径": "", "射程": "", "精度": ""}}
            ],
            "装备设备": [
              {{"名称": "", "类型": "", "型号": "", "数量": null, "所属方": "", "所属组织": "", "位置": "", "经纬度": "", "任务": "", "高度": "", "距离": "", "口径": "", "射程": "", "精度": "", "响应时间": "", "耗时": "", "用途": ""}}
            ],
            "设施": [
              {{"名称": "", "类型": "", "数量": null, "所属方": "", "位置": "", "经纬度": "", "距离": "", "用途": ""}}
            ],
            "组织机构": [
              {{"名称": "", "类型": "", "所属方": "", "上级组织": "", "位置": "", "任务": ""}}
            ],
            "人员": [
              {{"姓名": "", "军衔": "", "职务": "", "所属方": "", "所属组织": ""}}
            ]
          }},
          "关联关系": [
            {{"头实体": "", "关系": "", "尾实体": "", "属性": {{}}, "证据": ""}}
          ],
          "述谓结构": "主体 谓词 客体；如：99AE坦克 精确打击 模拟目标",
          "核心行为": "事件核心动作，如部署、侦察、打击、协同、指挥、前移、伏击、救援",
          "风险等级": "高危/中危/低危"
        }}

        实体类型抽取规则：
        - 车辆包括主战坦克、装甲车、步兵战车、无人机、工程保障车辆等。
        - 装备设备包括导弹、火箭炮、防空系统、传感器、通信/数据链系统、火炮、雷达、后勤装备等；如果不确定是否是车辆，放入装备设备。
        - 舰艇、航空器、组织机构、人员、设施都要作为独立实体抽取。
        - “中方、印军、巴军”等阵营/军队名称必须作为组织机构抽取；第114装甲团、第8山地步兵旅、防空导弹连、后勤补给中心等也要抽取。
        - 人员如“李云龙大校”“某军长/师长/排长”必须抽取姓名、军衔、职务和所属组织。

        关系抽取规则：
        - 关系类型不要限制在对抗、合作、隶属、驻守；按原文语义扩展，如：部署于、装备、使用、隶属、指挥、协同、数据链互通、引导、打击、侦察、起飞自、建立、前移至、支援、保障、伏击、击毁、命中、优于、劣于、对抗、同方协同、敌对阵营。
        - 必须显式抽取阵营关系：例如“印军 装备 T-90S”“第114装甲团 装备 T-90S”“T-90S 同方协同 MQ-9B”“T-90S 对抗 99AE主战坦克”“T-90S 对抗 PHL-03远程火箭炮系统”“T-90S 对抗 翼龙-2无人机”。
        - 如果文本出现中印双方、印巴双方等对峙/冲突/演习对抗场景，要把不同阵营的主要武器装备建立“对抗”或“敌对阵营”关系，把同阵营武器装备建立“同方协同”关系。
        - 关系属性中保留数量、距离、耗时、响应时间、射程、精度、速度、高度等能够限定关系的指标。

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

