
# 纯小白从 0 到 1 跑通一个 DeepSeek × LlamaIndex 的法律 RAG 系统

记录我从零基础商科生（本科商科，研究生法学，此前编程基础薄弱），到成功搭建一个法律领域 RAG 系统的全过程

内容包含：环境搭建、所有踩坑、错误信息、解决方案、关键 patch、最终运行结果

---
## 1. 项目背景与目标

本项目是我作为 **商科 & 法律交叉背景学生** 的第一次 AI 工程实战。  
目标是在 **没有 OpenAI Key 的前提下**，利用：

- **DeepSeek Chat（OpenAI-Compatible API）**
- **LlamaIndex（RAG 框架）**
- **中文 BGE Embedding 模型**

搭建一个能够 **引用法条 + 分析问题 + 引用课堂案例 + 给出回答** 的法律领域 RAG 系统。

---

为什么要做这个 RAG？

这个项目有三个目的：

- 🎯 展示我理解 RAG、向量数据库、LLM 的能力

- 🎯 证明我能从 0 完成 AI 原型（MVP）

- 🎯 GitHub 作品集 & 简历项目

---

## 2. 为什么不选择已有的云平台搭建？为什么不用OpenAI？

### 2.1 不使用 OpenAI API 的原因
- 境外卡余额不足，恰逢周无内无法快速充值，急急急（真实客观原因）
- 既然 DeepSeek 也兼容 OpenAI 格式，那么完全可以使用 DeepSeek 来跑 RAG

### 2.2 为什么不用 Flowise / Dify / Siliconflow etc……

**Flowise**
- 所有embeddings需要API Key
- 本地部署流程复杂，对于个人用户来说太麻烦，遂放弃

**Dify**
- 吞金兽
- 文件一直排队，捣鼓了一晚上感觉被耍了，失败

**Siliconflow**
- 没有商用版本，没有办法实现我的需求，放弃

---

## 3.为什么选择本地环境搭建？后附debug踩坑记录

- GPT推荐 + 纯本地 + DeepSeek API（大陆充值友好），开始环境搭建到最后跑通大约用了3小时，debug多亏了GPT

### 3.1 工程目录（暂时顺利的过程）

```
cd D:\law_rag_project # 创建RAG项目文件夹
```

### 3.2 创建虚拟环境

```
python -m venv .venv
```

### 3.3 激活Windows

```
.\.venv\Scripts\activate
```

---

## 4. 安装依赖

### 4.1 初次尝试（❌ 报错）
```
pip install llama-index llama-index-llms-deepseek llama-index-embeddings-huggingface
```

<img width="1324" height="405" alt="image" src="https://github.com/user-attachments/assets/a9084731-f9bc-4ffb-b58c-0992455b7a40" />

<img width="1325" height="451" alt="image" src="https://github.com/user-attachments/assets/131bc0cf-3493-434c-926c-f90fc8a586cc" />

报错原因：

- pip 太旧
- llama-index-llms-deepseek（包名错误，GPT提供错）
- Python 版本不兼容
  
DeepSeek 采用 OpenAI 格式 API，所以用这个包：llama-index-llms-openai

---


## 5. 升级 pip 后, 怕vpn太卡使用清华源， 用 DeepSeek 的 OpenAI 兼容接口来跑 RAG：
```
pip install "llama-index==0.11.10" llama-index-llms-openai llama-index-embeddings-openai python-dotenv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 6. 准备你的法律文本数据

在工程目录创建 data 文件夹：

```
mkdir data
```
把清洗好的《劳动合同法》文件放入：

```
data/
│── labor_contract_law_1_98.txt  # 劳动合同法 1-98 条
│── cases_labor.txt              # 案例库
```
---

## 7. 创建 . env（环境变量）

手动创建 .env 文件（关键）：
选择用 DeepSeek 的真实 API key，这里我还不知道之后要对LlamaIndex进行欺骗

```
DEEPSEEK_API_KEY=你的深度求索_API_Key_填这里
OPENAI_API_KEY=你的深度求索_API_Key_填这里
OPENAI_BASE_URL=https://api.deepseek.com 
```

### 7.1. 为什么要写两遍？

大部分 Python 模型调用库（包括 LlamaIndex 的 OpenAI-compatible 驱动）默认使用：

因为：

- OpenAI SDK 默认读取 OPENAI_API_KEY
- LlamaIndex 也会用 OpenAI-compatible 接口
- 写两遍能同时被这两个系统识别
- DeepSeek 本质是 “冒充（兼容）OpenAI API 格式”

---

## 8. 主程序文件rag_law_bot.py

```
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
```

---

## 9. 踩坑记录

### 9.1 ❌ LlamaIndex 主包缺少部分子模块，需要额外安装一个扩展包

报错：

<img width="1332" height="275" alt="image" src="https://github.com/user-attachments/assets/c3fd27dd-fba0-4ae2-b343-30e693b96bac" />

```
ModuleNotFoundError: No module named 'llama_index.embeddings.huggingface'
```

解决：

```
# 保持现在这个虚拟环境，执行：

pip install llama-index-embeddings-huggingface -i https://pypi.tuna.tsinghua.edu.cn/simple
```

版本不兼容报错，可以忽略

<img width="1205" height="825" alt="image" src="https://github.com/user-attachments/assets/213cb297-fe60-4bb5-aec4-75595f2971f1" />

原因：

llama-index-xxx 需要 llama-index-core <0.12.0, >=0.11.0, 但你现在有的是 0.14.8，版本不兼容，但是问题不大，安装成功

### 9.2 如果之后真的因为版本冲突挂了怎么办？ 赶时间没必要
```
# 1）先把以前装的 llama-index 系列都卸掉
pip uninstall -y "llama-index" "llama-index-*"

# 2）装一套相互兼容的版本（示例一套够用的组合）
pip install "llama-index==0.11.23" \
            "llama-index-llms-openai==0.1.16" \
            "llama-index-embeddings-huggingface==0.1.6"
```
### 9.3 ❌ BUG1: .env 文件未生效
```
(.venv311) PS C:\law_rag_project> python rag_law_bot.py Traceback (most recent call last): File "C:\law_rag_project\rag_law_bot.py", line 19, in <module> raise ValueError("没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。") ValueError: 没有找到 OPENAI_API_KEY，请检查 .env 文件是否配置正确。 (.venv311) PS C:\law_rag_project>
```
原因：文件名写成 env.txt

解决：
```
# 强制重命名

ren env.txt .env

dir -Force
```
<img width="1207" height="557" alt="image" src="https://github.com/user-attachments/assets/f7216d92-2ef6-4f77-b754-74ffb90e719f" />

### 9.4 ❌ BUG2: LlamaIndex 不认识 deepseek-chat（模型名校验失败）


<img width="679" height="258" alt="image" src="https://github.com/user-attachments/assets/cb42f92d-8db6-4c57-a5d7-79f3046cc61c" />

<img width="617" height="269" alt="image" src="https://github.com/user-attachments/assets/70126847-91c8-4fe9-b8e2-326bb6c78b5a" />

报错：

```
Unknown model 'deepseek-chat'
```
原因：LlamaIndex 内部维护了一份 model whitelist，而 deepseek-chat 不在里面 → 直接报错
解决方式：手动 patch LlamaIndex 内部的模型列表
修改 llama_index/llms/openai/utils.py：

```
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
```
#### 效果：成功骗过 LlamaIndex，使其接受 deepseek-chat

### 9.5 ❌ OpenAI SDK 仍然在尝试走 openai.com

查询出错，openai SDK仍然认为key是OpenAI的key：
```
Error code: 401 - {'error': {'message': 'Incorrect API key provided: ************. 
You can find your API key at https://platform.openai.com/account/api-keys.', 'type': 'invalid_request_error', 'code': 'invalid_api_key', 'param': None}}
```
说明：
OpenAI SDK 没有使用 DeepSeek 的 base_url，而是默认访问 openai.com。

解决：手动覆盖 base_url：

```
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = "https://api.deepseek.com/v1"

print("OpenAI SDK 已改用 DeepSeek API")
print("✅ 已读取到 OPENAI_API_KEY，准备初始化 LLM 与向量模型...")

```
成功让 OpenAI SDK → 走 DeepSeek API

# 10. 后续美化输出

<img width="1210" height="470" alt="image" src="https://github.com/user-attachments/assets/6bdf2074-6f3f-44b8-b1a9-e0b0112576a0" />

原因：160个字符限制太短

解决：不截断法条内容
```
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

```



