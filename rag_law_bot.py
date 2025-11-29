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
