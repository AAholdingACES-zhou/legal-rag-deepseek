#DEBUG_LOG

记录我从零基础文科生，到成功搭建一个法律领域 RAG 系统的全过程
包含所有踩坑、解决方案、技术突破与反思

#Debug 日志：从 0 到 1 跑通一个 DeepSeek × LlamaIndex 的法律 RAG 系统

本项目是我作为商科生（经管法全沾了）探索 AI 产品工程的第一次完整实战。
目标是：在 没有 OpenAI Key 的情况下，用 DeepSeek API + LlamaIndex 自己手搓出一个可以“引用法条 + 分析 + 作答”的 RAG 系统。

整个过程包含了失败 → 重试 → 重新设计 → 手动 patch 框架 → 成功跑通的全过程。
这篇 Debug 日志既是技术排错记录，也是一个 AI 产品从 0 到 1 的还原。

# 为什么要做这个 RAG？

这个项目有三个目的：

🎯 展示我理解 RAG、向量数据库、LLM 的能力

🎯 证明我能从 0 完成 AI 原型（MVP）

🎯 作为 GitHub 作品集 & 简历项目

最终版本：
DeepSeek Chat（OpenAI-Compatible API） × LlamaIndex × BGE Embeddings 的法律问答 RAG

# 为什么不选择已有的云平台搭建？为什么避开OpenAI？

首先为什么不用OpenAI的API：首先因为主包境外卡一分钱没有了（悲痛）。恰逢周末，转钱周一才能到账（也不想找tb充钱），想要快点跑通项目的主包选择用人民币充钱的DeepSeek。

其次是：

Flowise：上面所有embeddings都是需要API的，只有本地部署才可以用免费BGE，对于个人用户来说本地部署太麻烦，遂放弃

Dify：吞金兽，遂放弃（文件一直排队，捣鼓了一晚上感觉被耍了，失败）

Siliconflow：没有商用版本，没有办法实现我的需求，放弃

# 为什么选择本地环境搭建？后附debug踩坑记录

GPT推荐以及看起来可行，开始环境搭建到最后跑通大约用了3小时，debug多亏了GPT

暂时顺利的过程：

# 在 PowerShell 里进入工程目录：

cd D:\law_rag_project # 创建RAG项目文件夹

# 创建虚拟环境

python -m venv .venv

# 激活Windows

.\.venv\Scripts\activate

# 安装依赖

pip install llama-index llama-index-llms-deepseek llama-index-embeddings-huggingface

# ❌ pip版本很旧 + Python 版本不兼容 + llama-index-llms-deepseek 包名是错误的（官方名称变了，gpt自己提供错的）

报错两次：

<img width="1324" height="405" alt="image" src="https://github.com/user-attachments/assets/a9084731-f9bc-4ffb-b58c-0992455b7a40" />

<img width="1325" height="451" alt="image" src="https://github.com/user-attachments/assets/131bc0cf-3493-434c-926c-f90fc8a586cc" />

升级到gpt推荐的 python 3.11.9 

DeepSeek 采用 OpenAI 格式 API，所以用这个包：

llama-index-llms-openai

升级 pip 后, 怕vpn太卡使用清华源， 用 DeepSeek 的 OpenAI 兼容接口来跑 RAG：

pip install "llama-index==0.11.10" llama-index-llms-openai llama-index-embeddings-openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple

# 准备你的法律文本数据

在 C:\law_rag_project 目录下，新建一个文件夹：

mkdir data

把清洗好的《劳动合同法》 txt 文件放进data文件夹

# 资源管理器手动创建 . env

选择用 DeepSeek 的真实 API key，这里我还不知道之后要对LlamaIndex进行欺骗

DEEPSEEK_API_KEY=你的深度求索_API_Key_填这里
OPENAI_API_KEY=你的深度求索_API_Key_填这里
OPENAI_BASE_URL=https://api.deepseek.com 

# 注意只写ds 的 api ； 以下解释来自gpt

大部分 Python 模型调用库（包括 LlamaIndex 的 OpenAI-compatible 驱动）默认使用：

OPENAI_API_KEY
OPENAI_BASE_URL

这是为了兼容 OpenAI 格式的 API ， 其实只有一个 key，但为了让所有代码都能找到它，必须写两遍

# 新建Python文件 

rag_law_bot.py

import os
from dotenv import load_dotenv

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 读 .env，拿到你的 DeepSeek Key
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")

if not DEEPSEEK_API_KEY:
    raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。")

# ⚠️ 如果后面下载 HuggingFace 模型太慢，可以试试镜像：
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 配置 LLM（用 DeepSeek 的 chat 接口，走 OpenAI 兼容协议）
llm = OpenAI(
    model="deepseek-chat",     # 你在 DeepSeek 控制台里看到的 chat 模型名
    api_key=DEEPSEEK_API_KEY,
    base_url=BASE_URL
)

# 3. 配置向量模型（用一个中文小模型，做检索用）
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"   # 中文向量模型，够用又不太大
)

# 统一设置到 LlamaIndex 的全局 Setting
Settings.llm = llm
Settings.embed_model = embed_model

# 4. 读取 ./data 目录里的所有 txt 文档
print("📚 正在加载本地文档 ./data ...")
documents = SimpleDirectoryReader("./data").load_data()
print(f"已加载文档数量: {len(documents)}")

# 5. 构建向量索引（第一次会稍慢一点）
print("🧠 正在构建向量索引（VectorStoreIndex）...")
index = VectorStoreIndex.from_documents(documents)
print("✅ 索引构建完成！可以开始提问了～")

# 6. 生成查询引擎
query_engine = index.as_query_engine(
    similarity_top_k=3,   # 检索 TopK，可以之后再调
)

# 7. 简单的命令行对话循环

def main():
    print("\n====== 劳动法 RAG 助手 ======")
    print("输入你的问题，输入 q / quit 退出。\n")

    while True:
        question = input("👩‍⚖️ 你问：").strip()
        if question.lower() in {"q", "quit", "exit"}:
            print("👋 Bye～")
            break

        if not question:
            continue

        try:
            response = query_engine.query(question)
        except Exception as e:
            print(f"❌ 调用出错: {e}")
            continue

        print("\n【回答】")
        print(response.response)

        # 展示一下引用到的法条片段，方便你核对
        print("\n【参考片段】")
        for i, sn in enumerate(response.source_nodes, start=1):
            content = sn.node.get_content().strip().replace("\n", " ")
            print(f"{i}. {content[:150]}...")
        print("\n---------------------------\n")


if __name__ == "__main__":
    main()


# ❌ LlamaIndex 主包缺少部分子模块，需要额外安装一个扩展包 （全程gpt写代码的原因，但换我我更不会写）

<img width="1332" height="275" alt="image" src="https://github.com/user-attachments/assets/c3fd27dd-fba0-4ae2-b343-30e693b96bac" />

ModuleNotFoundError: No module named 'llama_index.embeddings.huggingface'

# 保持现在这个虚拟环境，执行：

pip install llama-index-embeddings-huggingface -i https://pypi.tuna.tsinghua.edu.cn/simple

装了很长时间大概10分钟以上，因为llama-index-* 会连续装一堆依赖包

# 版本不兼容报错，可以忽略

<img width="1205" height="825" alt="image" src="https://github.com/user-attachments/assets/213cb297-fe60-4bb5-aec4-75595f2971f1" />

翻译：

llama-index-xxx 需要 llama-index-core <0.12.0, >=0.11.0, 但你现在有的是 0.14.8，版本不兼容

部分先前已经装过一批 llama-index-* 包（版本比较旧，要求 core 在 0.11.x 左右）

但是问题不大，安装成功

Successfully installed ... llama-index-embeddings-huggingface-0.6.1 ... torch-2.9.1 transformers-4.57.3

# 如果之后真的因为版本冲突挂了怎么办？ 赶时间没必要

# 1）先把以前装的 llama-index 系列都卸掉
pip uninstall -y "llama-index" "llama-index-*"

# 2）装一套相互兼容的版本（示例一套够用的组合）
pip install "llama-index==0.11.23" \
            "llama-index-llms-openai==0.1.16" \
            "llama-index-embeddings-huggingface==0.1.6"

# ❌ 运行后出现第一个BUG 

(.venv311) PS C:\law_rag_project> python rag_law_bot.py Traceback (most recent call last): File "C:\law_rag_project\rag_law_bot.py", line 19, in <module> raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。") ValueError: 没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。 (.venv311) PS C:\law_rag_project>

.env文件后缀错误，我一开始写的是 env.txt

# 强制重命名

ren env.txt .env

dir -Force

<img width="1207" height="557" alt="image" src="https://github.com/user-attachments/assets/f7216d92-2ef6-4f77-b754-74ffb90e719f" />

# ❌ 运行后出现第二个BUG 【重要】

<img width="679" height="258" alt="image" src="https://github.com/user-attachments/assets/cb42f92d-8db6-4c57-a5d7-79f3046cc61c" />

LlamaIndex 不认识 deepseek-chat 这个“OpenAI 模型名”，只认识官方的 GPT 模型名（gpt-4o、gpt-3.5-turbo 等）

#  LlamaIndex 打补丁，让它认识 deepseek-chat
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

# 持续报错

<img width="617" height="269" alt="image" src="https://github.com/user-attachments/assets/70126847-91c8-4fe9-b8e2-326bb6c78b5a" />

# 解释：LlamaIndex 这个包在别的文件里已经把函数“拷贝了一份引用”，所以我们在代码里 monkey-patch 了 utils.openai_modelname_to_contextsize 也没法覆盖那份旧引用

# LlamaIndex 报 Unknown model 'deepseek-chat'，就是因为：openai_modelname_to_contextsize() 只认这个 ALL_AVAILABLE_MODELS 里的 key

is_chat_model() 只认 CHAT_MODELS 里的 key

# 找到报错源文件，把 deepseek-chat 把它“骗”成一个已知模型就行（关键）

C:\law_rag_project\.venv311\Lib\site-packages\llama_index\llms\openai\utils.py

ALL_AVAILABLE_MODELS = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **GPT3_5_MODELS,
    **GPT3_MODELS,
    **AZURE_TURBO_MODELS,
    "deepseek-chat": 8192,   # 👈 新增这一行
}

CHAT_MODELS = {
    **O1_MODELS,
    **GPT4_MODELS,
    **TURBO_MODELS,
    **AZURE_TURBO_MODELS,
    "deepseek-chat": 8192,   # 👈 这里也加一行
}


# ❌ 运行后出现第三个BUG 【最重要】

查询出错： Error code: 401 - {'error': {'message': 'Incorrect API key provided: ************. 
You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None}}

必须手动告诉openai SDK，base_url 到 deepseek，key 是 deepseek key

# 加入🔧 DeepSeek 补丁

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = "https://api.deepseek.com/v1"

print("OpenAI SDK 已改用 DeepSeek API")
print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")

# 后续美化输出

<img width="1210" height="470" alt="image" src="https://github.com/user-attachments/assets/6bdf2074-6f3f-44b8-b1a9-e0b0112576a0" />

160个字符限制太短，直接改为：

def pretty_print_response(resp):
    """美化输出：正文 + 引用片段"""
    print("\n====== 模型回答 ======\n")
    print(str(resp))

    # 展示引用的法条片段，方便核查
    if getattr(resp, "source_nodes", None):
        print("\n====== 引用片段（Top 3）======")
        for i, sn in enumerate(resp.source_nodes[:3], 1):
            text = sn.node.get_content().strip()
            print(f"\n[{i}] score={sn.score:.3f}\n{text}\n")





