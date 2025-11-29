# ⚖️ Chinese Legal RAG System — 基于 DeepSeek + LlamaIndex 的劳动合同法智能问答+期末考复习系统

本项目使用 **DeepSeek Chat + LlamaIndex + BGE 中文向量模型** 构建一个“可本地运行”的**法律检索增强生成（RAG）问答系统**，目前已支持对《劳动合同法》条文进行检索、引用与基础分析，后续会加入劳动法案例库，案例库材料来自复旦大学法学院劳动法课程，承担期末复习RAG的功能

---

## 📌 项目亮点（Overview）

- 利用 **DeepSeek Chat（OpenAI 兼容 API）** 作为大语言模型，避免直接依赖 OpenAI Key（适用于OpenAI Key暂时没有钱但是Deepseek有钱的情况）
- 使用 **BAAI/bge-small-zh-v1.5** 作为中文嵌入模型，构建法律文本向量索引
- 基于 **LlamaIndex** 搭建完整 RAG 流程：本地文档加载 → 向量化 → 索引 → 检索 → LLM 生成
- 支持在终端交互式提问：例如「试用期内能否随意辞退劳动者？」并返回结论 + 引用法条 + 引用类案
- 对 LlamaIndex 与 OpenAI SDK 进行了一系列“兼容 deepseek-chat”补丁，解决模型识别与 API 401 问题

---

## 🧱 项目结构

> 这是本地项目的大致结构（示例）：

```bash
law_rag_project/
│
├── data/                               # 法条/案例文件（本地知识库）
│   ├── labor_contract_law_1_98.txt     # 清洗后的《劳动合同法》1-98条
│   └── cases_labor.txt                 # 劳动法案例库
│
├── rag_law_bot.py                # RAG 主程序（终端问答）
├── .env                          # DeepSeek API Key（不上传github）
├── Pj_info.md                    # 项目说明文档（本文件）
└── requirements.txt              # 依赖（最简版本）
