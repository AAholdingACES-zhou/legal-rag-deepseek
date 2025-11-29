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

print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")


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
    similarity_top_k=3,          # 每次从知识库里找 3 条最相近的法条/案例
    response_mode="compact",     # 输出相对精简
)


def pretty_print_response(resp):
    """美化输出：正文 + 引用片段"""
    print("\n====== 模型回答 ======\n")
    print(str(resp))

    # 展示引用的法条片段，方便你核查
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