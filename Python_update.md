## 1. 最初版本python
```
import os
from dotenv import load_dotenv

# 1. 先加载 .env
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。")

# 2. 强制 OpenAI SDK 使用 DeepSeek API（关键补丁）
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

print(f"已为 OpenAI SDK 设置 base_url = {openai.base_url}")
print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils


# 2. 给 LlamaIndex 打一个“小补丁”，让它认识 deepseek-chat
_orig_ctx_func = openai_utils.openai_modelname_to_contextsize


def _patched_openai_modelname_to_contextsize(model_name: str) -> int:
    # 对 deepseek 系列模型，返回一个固定的 context window
    if model_name.startswith("deepseek"):
        # DeepSeek 官方上下文一般是 8K 或 16K，这里保守给 8192
        return 8192
    # 其他模型依然走原来的逻辑
    return _orig_ctx_func(model_name)


openai_utils.openai_modelname_to_contextsize = _patched_openai_modelname_to_contextsize
print("🔧 已为 LlamaIndex 打补丁，使其支持 deepseek-chat 模型。")


# 3. 配置中文向量模型（BGE，小型中文 embedding）
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
print(f"📦 正在加载向量模型: {EMBED_MODEL_NAME} ...")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

Settings.embed_model = embed_model

# 4. 配置 DeepSeek 作为 LLM（通过 OpenAI 兼容协议）
#    注意：这里的 model 就是 deepseek 的模型名
llm = OpenAI(
    model="deepseek-chat",
    temperature=0.2,
)
Settings.llm = llm
print("🤖 已配置 deepseek-chat 作为对话模型。")


# 5. 从 ./data 目录加载法律文档
DATA_DIR = "./data"
print(f"📚 正在加载本地文档 {DATA_DIR} ...")
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"已加载文档数量: {len(documents)}")

# 6. 构建向量索引
print("🧠 正在构建向量索引（VectorStoreIndex）...")
index = VectorStoreIndex.from_documents(documents)
print("✅ 索引构建完成！可以开始提问了～")

# 7. 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,          # 每次从知识库里找 3条最相近的法条/案例
    response_mode="compact",     # 输出相对精简
)


def pretty_print_response(resp):
    """美化输出：正文 + 引用片段"""
    print("\n====== 模型回答 ======\n")
    print(str(resp))

    # 展示引用的法条片段，方便调用
    if getattr(resp, "source_nodes", None):
        print("\n====== 引用片段（Top 3）======")
        for i, sn in enumerate(resp.source_nodes[:3], 1):
            text = sn.node.get_content().strip()
            print(f"\n[{i}] score={sn.score:.3f}\n{text}\n")


# 8. 简单 REPL 循环：在终端里和 bot 对话
while True:
    user_input = input("\n💬 请输入你的问题（输入 q 退出）：\n> ").strip()
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 已退出，再见～")
        break

    if not user_input:
        continue

    try:
        resp = query_engine.query(user_input)
        pretty_print_response(resp)
    except Exception as e:
        print("❌ 查询出错：", e)
```
#### 1.1 问题：分析简单，只能输出结论+法条
#### 1.2 解决：上传案例库

## 2. 版本2.0 增加案例库
```
import os
from dotenv import load_dotenv

# 1. 先加载 .env
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。")

# 2. 强制 OpenAI SDK 使用 DeepSeek API（关键补丁）
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

print(f"已为 OpenAI SDK 设置 base_url = {openai.base_url}")
print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils


# 2. 给 LlamaIndex 打一个“小补丁”，让它认识 deepseek-chat
_orig_ctx_func = openai_utils.openai_modelname_to_contextsize


def _patched_openai_modelname_to_contextsize(model_name: str) -> int:
    # 对 deepseek 系列模型，返回一个固定的 context window
    if model_name.startswith("deepseek"):
        # DeepSeek 官方上下文一般是 8K 或 16K，这里保守给 8192
        return 8192
    # 其他模型依然走原来的逻辑
    return _orig_ctx_func(model_name)


openai_utils.openai_modelname_to_contextsize = _patched_openai_modelname_to_contextsize
print("🔧 已为 LlamaIndex 打补丁，使其支持 deepseek-chat 模型。")


# 3. 配置中文向量模型（BGE，小型中文 embedding）
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
print(f"📦 正在加载向量模型: {EMBED_MODEL_NAME} ...")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)

Settings.embed_model = embed_model

# 4. 配置 DeepSeek 作为 LLM（通过 OpenAI 兼容协议）
#    注意：这里的 model 就是 deepseek 的模型名
llm = OpenAI(
    model="deepseek-chat",
    temperature=0.2,
)
Settings.llm = llm
print("🤖 已配置 deepseek-chat 作为对话模型。")


# 5. 从 ./data 目录加载法律文档
DATA_DIR = "./data"
print(f"📚 正在加载本地文档 {DATA_DIR} ...")
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"已加载文档数量: {len(documents)}")

# 6. 构建向量索引
print("🧠 正在构建向量索引（VectorStoreIndex）...")
index = VectorStoreIndex.from_documents(documents)
print("✅ 索引构建完成！可以开始提问了～")

# 7. 创建查询引擎
#    similarity_top_k 稍微调大，更容易同时捞到“法条 + 案例”
query_engine = index.as_query_engine(
    similarity_top_k=5,          # 每次从知识库里找 5条最相近的法条/案例
    response_mode="compact",     # 输出相对精简
)


def pretty_print_response(resp):
    """美化输出：正文 + 引用片段"""
    print("\n====== 模型回答 ======\n")
    print(str(resp))

    # 展示引用的法条片段，方便调用
    if getattr(resp, "source_nodes", None):
        print("\n====== 引用片段（Top 3）======")
        for i, sn in enumerate(resp.source_nodes[:3], 1):
            text = sn.node.get_content().strip()
            print(f"\n[{i}] score={sn.score:.3f}\n{text}\n")


# 8. 简单 REPL 循环：在终端里和 bot 对话
while True:
    user_input = input("\n💬 请输入你的问题（输入 q 退出）：\n> ").strip()
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 已退出，再见～")
        break

    if not user_input:
        continue

# 🔑 核心改动：这里给模型一个“壳”，强制它按照 3 个部分来输出
    full_prompt = f"""
你是一名中国劳动法专业助理，需要基于《劳动合同法》条文和预先整理的典型案例库，
回答下面的用户问题。请严格按照下面的结构作答：

1.【结论与分析】
- 先用 1–3 句话直接给出结论（是否违法、谁承担责任、劳动者可以主张什么权利）。
- 然后用 3–6 句话结合《劳动合同法》的要件进行推理分析，不要空泛说教，要贴着法条和事实说。

2.【适用法条及条文内容】
- 列出你认为“真正起决定性作用”的条文，优先引用《劳动合同法》，如有必要可补充实施条例或司法解释。
- 每一条单独一行，格式示例：
  - 《劳动合同法》第21条【试用期内解除劳动合同】：用人单位在试用期解除劳动合同的，应当向劳动者说明理由；除本法第39条和第40条第1、2项情形外，不得解除。
- 如你从当前上下文中没有检索到明确对应条文，请如实写：“未在知识库中检索到明确对应条文，但可类推适用相关一般规定”。

3.【类案参考（如有）】
- 如果你在本次检索到的上下文中看到了我们整理的案例（通常以“案例一”“案例二”或“【案例】案例X”开头），
  请选择其中【1个】与本案最相似的，用 2–4 句话说明：
  - 该案的关键事实
  - 该案的处理结论
  - 该案结论如何支持或限制你在本案中的判断
- 如果你没有检索到合适的类案，或者类案与本案情形明显不同，请只写一句：“类案参考：无合适类案”。

请只基于提供的知识库内容（劳动合同法条文 + 案例库）来回答，避免凭空编造新的条文或案例名称。

【用户问题】
{user_input}
"""

    try:
        resp = query_engine.query(user_input)
        pretty_print_response(resp)
    except Exception as e:
        print("❌ 查询出错：", e)
```
#### 2.1 prompt 制定输出内容
#### 2.2 问题：模型回答的感觉不够模块化，也没有类案分析
```
💬 请输入你的问题（输入 q 退出）：
> 公司口头录用但一直不签合同，我可以拿双倍工资吗？

====== 模型回答 ======

如果用人单位在用工之日起超过一个月但不满一年内未与你签订书面劳动合同，你有权要求支付双倍工资。双倍工资的计算期间是从用工满一个月的次日起至满一年的前一 日止，最长不超过11个月。如果用工已满一年仍未签订合同，法律上视为双方已订立无固定期限劳动合同，但此时不再继续支付双倍工资。

此外，主张双倍工资的权利受到仲裁时效的限制，你需要在知道或应当知道权利被侵害之日起一年内提出申请。

====== 引用片段（Top 3）======

[1] score=0.683
双倍工资的支付期间为：自用工满一个月之日起至满一年之日止，**最长不超过11个月**。
2. 自用工满一年仍未签订书面劳动合同的，视为已订立无固定期限劳动合同，但法律并未规定继续支付双倍工资。
3. 因此，万某要求 2017年8月至2018年7月期间的双倍工资，没有法律依据，仲裁委员会驳回其请求是正确的。

【理由】
1. 《劳动合同法》第82条规定：用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动合同的，应当向劳动者每月支付二倍的工资。
   - 用工日：2016年8月1日；
   - 自2016年9月1日起（满一个月）至2017年7月31日（满一年）为双倍工资的法定区间，最长 11 个月。
2. 《劳动合同法》第14条规定：用人单位与劳动者连续工作满一年不订立书面劳动合同的，视为已订立无固定期限劳动合同。此时劳动合同关系性质发生变化，但法律并未再叠加新的双倍工资惩罚条款。
3. 双倍工资请求还受《劳动争议调解仲裁法》第21条关于一年仲裁时效限制：劳动者应当自“知道或应当知道权利被侵害之日”起一年内提出。即使在 11 个月区间内本来有双倍工资可主张，超过时效也可能丧失救济。
4. 万某直接主张 2017年8月至2018年7月的“双倍工资（无固定期限劳动合同未签的期间）”缺乏法律依据，因此不予支持。

【参考法律条文】
《劳动合同法》第14条、第82条
《劳动争议调解仲裁法》第21条


【案例】案例9 录用意向书是否等同于劳动合同
【英文标题】Case 9 Written offer of employment and labor contract

【案情】
朱某原在 A 公司工作，对薪资不满，准备跳槽。
2019年3月11日，朱某应聘 B 公司产品线经理职位并参加面试。
2019年3月22/23日，B 公司通过邮件向朱某发出聘用意向书（offer letter）及报到须知，内容包括：
- 表明拟录用朱某；
- 要求朱某于4月1日上午9:30到 B 公司报到；
- 报到时需携带身份证、劳动手册、原单位退工证明等材料。

朱某收到录用通知后，并未在规定时间内书面回复“接受录用”，而是先向 A 公司提出辞职。
3月29日，B 公司通知朱某：取消录用决定，不再聘用他。

朱某认为 B 公司撤回 offer 违反劳动法，主张其已与 B 公司之间成立劳动关系或劳动合同。

【争议焦点】
1.


[2] score=0.651
劳动关系认定的三要素：
   （1）用工主体合法：家具公司依法登记，具备用工主体资格；
   （2）劳动者受单位管理：虽“管理较为松散”，但家具公司仍对工作任务、工作地点、时间安排等具有组织和指挥权，杨某接受公司派工和调度；
   （3）劳动者提供劳动、单位支付报酬：杨某持续为公司提供安装劳动，按安装面积计件取得报酬，体现明显的经济从属性。
2. 自带劳动工具、计件工资、不打卡等形式是劳动关系下常见的用工方式，并不当然否定劳动关系。关键在于是否依附于用人单位的组织体系并取得劳动报酬。
3. 即便未签书面劳动合同，只要上述要素存在，即应认定为劳动关系。
4. 因此，杨某在安装家具时受伤，应按劳动关系下的工伤处理规则处理。

【参考法律条文】
《劳动合同法实施条例》第2条
人社部《劳动关系认定办法（试行）》第5条
《工伤保险条例》相关条款


【案例】案例8 用人单位未签无固定期限劳动合同是否持续支付双倍工资
【英文标题】Case 8 Whether the employer should pay double wages after the deemed conclusion of an open-ended labor contract

【案情】
2016年8月1日，万某入职某食品公司，从事检验工作，双方口头约定月工资为 3000 元。
公司负责人承诺：试用期 3 个月期满后再签书面劳动合同，但双方始终未签订任何书面劳动合同。
2018年7月31日，万某与食品公司解除劳动关系。此后，万某要求公司支付：
- 2017年8月至2018年7月期间，因未与其签订“无固定期限劳动合同”而欠付的双倍工资。
公司拒绝支付，万某申请劳动仲裁。仲裁委员会驳回其请求。

【争议焦点】
1. 用人单位在用工满一年仍未签订书面劳动合同，劳动关系被视为无固定期限劳动合同后，是否还需要继续向劳动者支付双倍工资？
2. 双倍工资的起算时间、最长期间以及诉讼/仲裁时效如何计算？

【结论】
1. 双倍工资的支付期间为：自用工满一个月之日起至满一年之日止，**最长不超过11个月**。
2. 自用工满一年仍未签订书面劳动合同的，视为已订立无固定期限劳动合同，但法律并未规定继续支付双倍工资。
3. 因此，万某要求 2017年8月至2018年7月期间的双倍工资，没有法律依据，仲裁委员会驳回其请求是正确的。

【理由】
1.


[3] score=0.620
朱某虽因信赖 B 公司的录用意向而从 A 公司辞职，但其未在规定时间内向 B 公司作出书面接受，且未到岗报到，尚未进入“用工”状态，劳动关系尚未成立。
4. B 公司撤回录用在劳动法上不构成解除劳动合同。朱某如主张损失，只能尝试依据《民法典》诚实信用原则及缔约过失责任，但根据(2019)沪02民终9985号案的裁判思路，法院通常认为双方尚未达成劳动合同，B 公司的撤回尚不构成违法。

【参考法律条文】
《劳动合同法》第10条
《民法典》总则编、合同编有关要约、要约邀请及缔约过失责任的规定
(2019) 沪02民终9985号 相关裁判观点

【案例】案例10 无固定期限劳动合同期间订立固定期限合同后拒绝续签是否合法
【英文标题】Case 10 Fixed-term labor contract concluded during an open-ended labor contract

【案情】
孙某自 2009 年 7 月 1 日起在北京某公司工作，岗位为资料管理员。
双方一直未签订书面劳动合同，但孙某持续履行工作至 2016 年。
2016 年 4 月 25 日，双方首次签订书面劳动合同，约定：
- 合同期限：2016 年 4 月 1 日至 2016 年 12 月 20 日（固定期限劳动合同）

2016 年 12 月 2 日，公司向孙某发出《不续签劳动合同通知书》，表示合同到期不再续签。
2016 年 12 月 20 日，劳动合同期满，公司终止劳动关系。
孙某起诉，请求确认公司构成违法终止劳动合同并支付赔偿金。

一审法院认为：
- 2016 年 4 月 25 日签订的固定期限劳动合同改变了之前“默示形成”的无固定期限劳动合同性质；
- 视为双方仅订立过一次固定期限劳动合同；
- 合同期满终止不违法。

孙某不服，上诉。
二审（2017）京 03 民终 10611 号认为：
- 自 2009 年 7 月 1 日起，双方已经形成无固定期限劳动合同（未书面但持续用工）；
- 2016 年 4 月 25 日又签订了书面固定期限劳动合同；
- 应视为双方已连续订立两次劳动合同；
- 孙某有权在固定期限劳动合同期满时选择续签，且有权主张无固定期限劳动合同。

【争议焦点】
1. 在已形成无固定期限劳动合同的基础上，再签订固定期限劳动合同，是否会“改写”原有合同性质？
2.
```
<img width="1517" height="618" alt="image" src="https://github.com/user-attachments/assets/02831026-7297-43b5-a3c9-164ae2f283d3" />

## 3. 版本 3.0 
```
import os
from dotenv import load_dotenv

# 1. 先加载 .env
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。")

# 2. 强制 OpenAI SDK 使用 DeepSeek API（关键补丁）
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

print(f"已为 OpenAI SDK 设置 base_url = {openai.base_url}")
print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils


# 2. 给 LlamaIndex 打一个“小补丁”，让它认识 deepseek-chat
_orig_ctx_func = openai_utils.openai_modelname_to_contextsize


def _patched_openai_modelname_to_contextsize(model_name: str) -> int:
    # 对 deepseek 系列模型，返回一个固定的 context window
    if model_name.startswith("deepseek"):
        # DeepSeek 官方上下文一般是 8K 或 16K，这里保守给 8192
        return 8192
    # 其他模型依然走原来的逻辑
    return _orig_ctx_func(model_name)


openai_utils.openai_modelname_to_contextsize = _patched_openai_modelname_to_contextsize
print("🔧 已为 LlamaIndex 打补丁，使其支持 deepseek-chat 模型。")

print("📦 正在加载向量模型: BAAI/bge-small-zh-v1.5 ...")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")
Settings.embed_model = embed_model

# 4. 配置 DeepSeek 作为 LLM（通过 OpenAI 兼容协议）
llm = OpenAI(
    model="deepseek-chat",
    temperature=0.2,
)
Settings.llm = llm
print("🤖 已配置 deepseek-chat 作为对话模型。")

# 5. 从 ./data 目录加载法律文档（法条 + 案例库）
DATA_DIR = "./data"
print(f"📚 正在加载本地文档 {DATA_DIR} ...")
documents = SimpleDirectoryReader(DATA_DIR).load_data()
print(f"已加载文档数量: {len(documents)}")

# 6. 构建向量索引
print("🧠 正在构建向量索引（VectorStoreIndex）...")
index = VectorStoreIndex.from_documents(documents)
print("✅ 索引构建完成！可以开始提问了～")

# 7. 创建查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=5,          # 稍微大一点，方便同时命中法条 + 案例
    response_mode="compact",
)


def pretty_print_response(resp):
    """只输出模型回答本身，不再打印调试用的引用片段。"""
    print("\n====== 模型回答 ======\n")
    print(str(resp))
    print("\n======================\n")


# 8. 简单 REPL 循环：在终端里和 bot 对话
while True:
    user_input = input("\n💬 请输入你的问题（输入 q 退出）：\n> ").strip()
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 已退出，再见～")
        break

    if not user_input:
        continue

    # 🔑 核心提示词：强制三段式输出 + 只允许 1 个类案 + 不要出现 score / Top3 等词
    full_prompt = f"""
你是一名中国劳动法专业助理，需要基于《劳动合同法》条文和预先整理的典型案例库，
回答下面的用户问题。请严格按照下面的结构作答，并遵守后面的规则。

【回答结构】

1.【结论与分析】
- 用 1～3 句话先给出直接结论（是否违法、谁承担责任、劳动者大致可以主张哪些权利）。
- 再用 3～6 句话进行法律论证：围绕《劳动合同法》的构成要件进行分析，
  例如权利义务、违法点、救济路径等。这里请尽量用你自己的法律推理，
  不要大段照抄案例中的“【理由】”原文，而是用“抽象规则 + 具体适用”的方式来写。

2.【适用法条及条文内容】
- 列出本案中起关键作用的法律条文，优先选择《劳动合同法》，如有必要可补充实施条例或司法解释。
- 每一条单独一行，采用如下格式（示例）：
  - 《劳动合同法》第82条【不订立书面劳动合同的法律责任】：用人单位自用工之日起超过一个月不满一年的……（可概括或适当引用原文要点）。
- 条文数量建议控制在 1～5 条之间，尽量精准，不要一股脑把所有相关条文都写上。
- 如果在当前知识库中没有检索到明确对应条文，请如实写明：“未在知识库中检索到明确对应条文，可依据一般劳动法原理进行类推适用”。

3.【类案参考（如有）】
- 如果你在本次检索到的上下文中，看到了我们整理的案例（通常以“案例一 / 案例1”“案例二 / 案例2”等开头），
  请在所有相关案例中【只选择 1 个你认为最相似、最有代表性的】进行简要介绍。
- 输出格式示例：
  - 案例名称：案例八 无固定期限劳动合同期间订立固定期限合同后拒绝续签是否合法
  - 关键事实：……
  - 裁判结论：……
  - 对本案的启示：……
- 如果没有合适的类案（或者命中的案例与本案情形明显不同），请只写一句：
  “类案参考：无合适类案”。

【重要约束】

- 不要直接输出底层检索片段的原始格式（例如【法条】、【内容】、【理由】的大段原文）。
  你可以参考它们，但需要用自己的话进行总结、重写和结构化。
- 你的回答对象是法律专业学生或法律从业者，可以适度使用专业术语，
  但整体表达要清晰、有条理。

【用户问题】
{user_input}
"""

    try:
        resp = query_engine.query(full_prompt)
        pretty_print_response(resp)
    except Exception as e:
        print("❌ 查询出错：", e)
```
#### 3.1 类案检索情况一般
#### 3.2 已隐藏底层检索片段
<img width="1328" height="456" alt="image" src="https://github.com/user-attachments/assets/4fc80109-7ea5-4b6e-98db-e9c626bef8b0" />

## 4. 版本4.0
```
import os
from dotenv import load_dotenv

# ==============================
# 1. 读取环境变量 & DeepSeek 补丁
# ==============================
load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。")

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")

print(f"已为 OpenAI SDK 设置 base_url = {openai.base_url}")
print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")

# ==============================
# 2. LlamaIndex & DeepSeek 兼容设置
# ==============================
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
import llama_index.llms.openai.utils as openai_utils

# 给 LlamaIndex 打补丁，让它认识 deepseek-chat
_orig_ctx_func = openai_utils.openai_modelname_to_contextsize


def _patched_openai_modelname_to_contextsize(model_name: str) -> int:
    if model_name.startswith("deepseek"):
        # DeepSeek 上下文一般是 8K 或 16K，这里保守给 8192
        return 8192
    return _orig_ctx_func(model_name)


openai_utils.openai_modelname_to_contextsize = _patched_openai_modelname_to_contextsize
print("🔧 已为 LlamaIndex 打补丁，使其支持 deepseek-chat 模型。")

# 配置中文向量模型（BGE，小型中文 embedding）
EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
print(f"📦 正在加载向量模型: {EMBED_MODEL_NAME} ...")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
Settings.embed_model = embed_model

# 配置 DeepSeek 作为 LLM
llm = OpenAI(
    model="deepseek-chat",
    temperature=0.2,
)
Settings.llm = llm
print("🤖 已配置 deepseek-chat 作为对话模型。")

# ==============================
# 3. 分别加载「法条库」和「案例库」
# ==============================
STATUTE_DIR = "./data/statutes"
CASE_DIR = "./data/cases"

# --- 法条库 ---
if not os.path.exists(STATUTE_DIR):
    raise ValueError("没有找到 ./data/statutes 目录，请创建并把《劳动合同法》等法条 txt 放进去。")

print(f"📚 正在加载法条文档 {STATUTE_DIR} ...")
statute_docs = SimpleDirectoryReader(STATUTE_DIR).load_data()
print(f"已加载法条文档数量: {len(statute_docs)}")

statute_index = VectorStoreIndex.from_documents(statute_docs)
statute_retriever = statute_index.as_retriever(similarity_top_k=5)  # 检索少量高相关法条


# --- 案例库 ---
case_retriever = None
best_case_enabled = False

if os.path.exists(CASE_DIR) and os.listdir(CASE_DIR):
    print(f"📚 正在加载案例文档 {CASE_DIR} ...")
    case_docs = SimpleDirectoryReader(CASE_DIR).load_data()
    print(f"已加载案例文档数量: {len(case_docs)}")
    case_index = VectorStoreIndex.from_documents(case_docs)
    # 这里 top_k 稍微放大一点，只用于选出「最像的 1 个」
    case_retriever = case_index.as_retriever(similarity_top_k=12)
    best_case_enabled = True
else:
    print("⚠ 未找到 ./data/cases 目录或目录为空，将暂不启用类案检索。")

print("🧠 向量索引构建完成！可以开始提问了～")


# ==============================
# 4. 构造最终回答的 Prompt
# ==============================
def build_final_prompt(question: str, statute_nodes, case_node_text: str | None) -> str:
    """把检索到的法条片段 & 案例片段塞进一个总 Prompt，让 LLM 按固定结构回答。"""

    statute_context_parts = []
    for i, sn in enumerate(statute_nodes[:3], 1):
        content = sn.node.get_content().strip()
        statute_context_parts.append(f"【法条片段{i}】\n{content}\n")
    statute_context = "\n".join(statute_context_parts) if statute_context_parts else "（未检索到明显相关的法条片段）"

    case_context = case_node_text.strip() if case_node_text else "（无明显相关案例，仅供一般性回答）"

    prompt = f"""
你是一名精通中国劳动法的律师助手，请基于给定的【法条材料】和【类案材料】回答用户问题，
并按照要求的结构输出，注意不要输出任何技术细节（如 TopK、score 等）。

【用户问题】
{question}

【法条材料】
{statute_context}

【类案材料】
{case_context}

请用中文输出，结构严格如下（标题和序号都要保留）：

1. 结论与分析：
- 先用 1～3 句话直接给出明确结论（例如：是否违法、能否主张双倍工资、是否构成劳动关系等）。
- 再用 3～6 句话进行简要的法律分析，重点说明：
  · 适用的是哪些法律条文（写出条款号，如《劳动合同法》第82条），
  · 对当事人有利和不利的因素各是什么，
  · 如有前提条件或例外情况，也一并说明。

2. 适用法条及条文内容：
- 只列出与你结论直接相关的 2～4 条关键法条。
- 格式示例：
  （1）《劳动合同法》第82条【未订立书面劳动合同的法律后果】：……（引用关键条文原文或高度概括，但不要超过 200 字）
- 如有需要，可以补充《劳动争议调解仲裁法》《民法典》《公司法》等相关条款，但不要堆砌无关条文。

3. 类案参考（如有）：
- 如果【类案材料】与问题高度类似，请用 1～2 段话概括：
  · 案情要点（当事人身份、核心争议）；
  · 裁判结论（法院如何认定）；
  · 对本问题的启示（用 2～3 点简要说明）。
- 如果【类案材料】相关性不高，请统一写：
  “本问题暂无特别贴近的典型案例，仅能作一般性参考，具体处理仍需结合个案事实。”

要求：
- 全程不要出现“TopK”“score”“source_nodes”等技术字段。
- 不要照搬原文中的“【理由】”标题，而是改写成自然的说理段落。
- 语言风格以专业律师风格为主，但尽量让非法律专业人士也能看懂。
"""
    return prompt


# ==============================
# 5. 命令行对话循环
# ==============================
while True:
    user_input = input("\n💬 请输入你的问题（输入 q 退出）：\n> ").strip()
    if user_input.lower() in ["q", "quit", "exit"]:
        print("👋 已退出，再见～")
        break

    if not user_input:
        continue

    try:
        # 1）先分别从「法条索引」和「案例索引」检索
        statute_nodes = statute_retriever.retrieve(user_input)

        best_case_text = None
        if best_case_enabled and case_retriever is not None:
            case_nodes = case_retriever.retrieve(user_input)
            if case_nodes:
                # 只选最相似的 1 个案例
                best_case_text = case_nodes[0].node.get_content().strip()

        # 2）用一个总 Prompt 让 LLM 按结构整合回答
        final_prompt = build_final_prompt(user_input, statute_nodes, best_case_text)
        final_resp = llm.complete(final_prompt)

        print("\n====== 模型回答 ======\n")
        print(final_resp.text.strip())

    except Exception as e:
        print("❌ 查询出错：", e)
```
#### 4.1 模型回答达到逾期
```
💬 请输入你的问题（输入 q 退出）：
> 公司口头录用但一直不签合同，我可以拿双倍工资吗？

====== 模型回答 ======

1. 结论与分析：
- 结论：如果公司口头录用后您已实际工作超过一个月但未满一年，您可以主张双倍工资；如果已工作满一年仍未签订合同，则视为已订立无固定期限劳动合同，但不能再主张双倍工资。
- 法律分析：
  · 根据《劳动合同法》第82条，用人单位自用工之日起超过一个月不满一年未订立书面劳动合同的，应当支付双倍工资。
  · 对您有利的是：只要存在实际用工且超过一个月未签合同，即可主张最长11个月的双倍工资差额。
  · 对您不利的是：双倍工资请求受一年仲裁时效限制（自知道权利被侵害日起算），且工作满一年后视为订立无固定期限合同，此时 法律未规定继续支付双倍工资。
  · 需注意：口头录用后实际提供劳动是主张权利的前提，且需保存工资支付记录、工作安排等证明劳动关系的证据。

2. 适用法条及条文内容：
（1）《劳动合同法》第10条【订立书面劳动合同】：建立劳动关系应当订立书面劳动合同。已建立劳动关系未同时订立书面合同的，应 当自用工之日起一个月内订立。
（2）《劳动合同法》第82条【未订立书面劳动合同的法律责任】：用人单位自用工之日起超过一个月不满一年未与劳动者订立书面劳动 合同的，应当向劳动者每月支付二倍的工资。
（3）《劳动合同法》第14条【无固定期限劳动合同】：用人单位自用工之日起满一年不与劳动者订立书面劳动合同的，视为用人单位与 劳动者已订立无固定期限劳动合同。

3. 类案参考：
类案材料中万某案与本案高度相关：
- 案情要点：万某2016年8月入职，公司一直未签订劳动合同，其主张2017年8月至2018年7月期间的双倍工资。
- 裁判结论：仲裁委员会仅支持2016年9月至2017年7月期间的双倍工资，驳回2017年8月后的双倍工资请求。
- 启示：
  （1）双倍工资计算期间严格限定为"用工满一个月后至满一年前"，最长11个月；
  （2）用工满一年后法律推定成立无固定期限劳动合同，但不再适用双倍工资罚则；
  （3）双倍工资请求需在法定仲裁时效内提出。
  ```
<img width="1341" height="867" alt="image" src="https://github.com/user-attachments/assets/ef502a92-3f61-4c74-95a6-cf7ffd1dac4c" />

