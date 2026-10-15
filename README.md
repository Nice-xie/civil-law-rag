# 基于 RAG 的民事诉讼代理辅助系统

一个基于检索增强生成（RAG）的民事诉讼法律咨询系统。系统检索本地法律知识库（民法典、民事诉讼法、案例库与司法解释），结合本地大模型（Ollama + qwen:7b）生成带「法条溯源」的回答，并将对话历史持久化到 MySQL。

> ⚠️ 本系统仅为法律辅助工具，不能替代执业律师。

## 功能特性

- **多轮法律咨询**：支持连续追问，自动携带对话历史理解指代（如「那如果没领证呢」）。
- **法条溯源**：每个回答下方附上检索到的原文法条，证明答案来自知识库而非凭空推断。
- **结构化切分**：法律文本按法条边界切分（绝不跨条截断），分片头标注「法律名称 · 编/分编/章/节」。
- **对话历史管理**：按会话分组保存，支持分页查看、查看详情、删除单段、清空全部。
- **数据分析大屏**（`/stats`）：ECharts 大屏看板，规则关键词分类 + 时间趋势 + 高频词云，支持点击柱状图联动词云。
- **数据挖掘**（`/mining`）：TF-IDF + KMeans 聚类，肘部法则 / 轮廓系数评估，PCA 二维投影可视化。
- **交互式可视化**（`/pyecharts`）：Pyecharts 雷达图 / 仪表盘 / 漏斗图。

## 目录结构

```
civil_law_rag_system/
├── app.py               # Flask 主程序（切分 + 向量库 + RAG + 路由）
├── analysis.py          # 数据分析/数据挖掘/可视化模块（纯函数）
├── seed_data.py         # 灌入 40 条演示提问（供看板/聚类演示）
├── requirements.txt     # Python 依赖
├── law_docs/            # 法律知识库（UTF-8 文本）
│   ├── 民法典.txt
│   ├── 民事诉讼法.txt
│   ├── 劳动合同法.txt        # 2012修正
│   ├── 劳动争议调解仲裁法.txt
│   ├── 民间借贷规定.txt      # 法释〔2020〕17号（2021第二次修正）
│   ├── 涉彩礼规定.txt        # 法释〔2024〕1号
│   ├── 合同编通则解释.txt     # 法释〔2023〕13号
│   ├── 婚姻家庭编解释一.txt   # 法释〔2020〕22号
│   ├── 证据规定.txt          # 法释〔2019〕19号
│   └── civil.txt            # 民事案例库（50 个案例）
├── static/              # Matplotlib 生成的聚类图（运行时输出）
└── templates/           # 前端页面（Jinja2）
    ├── index.html       # 咨询首页
    ├── history.html     # 历史记录列表
    ├── history_detail.html  # 对话详情
    ├── stats.html       # ECharts 数据分析大屏
    ├── mining.html      # 数据挖掘（聚类）结果页
    └── pyecharts.html   # Pyecharts 交互式可视化
```

## 环境要求

- Python 3.10+（已在 3.12 验证）
- [Ollama](https://ollama.com/) 已安装并拉取 `qwen:7b` 模型
- MySQL 8.0
- 首次运行需联网下载向量模型 `BAAI/bge-base-zh-v1.5`（约 400 MB，已配置 HuggingFace 国内镜像）

## 安装步骤

### 1. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 2. 准备 Ollama 与模型

```bash
# 启动 Ollama 服务后拉取模型
ollama pull qwen:7b
```

### 3. 初始化数据库

数据库连接配置在 `app.py` 的 `DB_CONFIG` 中（默认 `localhost` / `root`）。建库建表：

```sql
CREATE DATABASE IF NOT EXISTS civil_law_rag_db DEFAULT CHARSET utf8mb4;
USE civil_law_rag_db;

CREATE TABLE IF NOT EXISTS chat_history (
    id INT NOT NULL AUTO_INCREMENT,
    session_id VARCHAR(64) DEFAULT NULL,
    user_query TEXT NOT NULL,
    ai_answer TEXT NOT NULL,
    create_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

## 运行

```bash
python app.py
```

启动成功后访问 http://127.0.0.1:5000

启动日志会依次输出知识库切分、向量模型加载、FAISS 构建、Ollama 加载进度；首次启动（含模型下载）较慢，请耐心等待。

## 使用说明

1. 在首页输入法律问题，点击「发送」或按 Enter 提交。
2. 回答下方「检索溯源」区域展示本轮命中的原文法条。
3. 点击「结束查询」保存当前会话并开始新一轮对话。
4. 点击右上角「查看问答历史记录」进入历史页，可按会话查看/删除/清空。
5. 数据分析与可视化页面（需先有提问数据，可运行 `python seed_data.py` 灌入演示数据）：
   - `/stats`：ECharts 大屏，KPI 指标 + 分类分布 + 时间趋势 + 高频词云，点击柱状图可联动切换某分类的高频词。
   - `/mining`：对全部提问做 TF-IDF + KMeans 聚类，展示肘部法则、轮廓系数、PCA 散点图与各簇关键词。
   - `/pyecharts`：Pyecharts 雷达图、仪表盘、漏斗图。

## 技术栈

- **Web**：Flask
- **RAG**：LangChain + FAISS
- **向量模型**：BAAI/bge-base-zh-v1.5（查询侧已加中文检索指令前缀）
- **大模型**：Ollama + qwen:7b
- **存储**：MySQL（pymysql）
- **数据处理**：Pandas、NumPy、jieba 中文分词
- **数据挖掘**：scikit-learn（TF-IDF + KMeans + PCA + 轮廓系数）
- **静态可视化**：Matplotlib（肘部法则 / 轮廓系数 / 聚类散点图）
- **交互式可视化**：ECharts（大屏看板 + 词云 + AJAX 联动）、Pyecharts（雷达图 / 仪表盘 / 漏斗图）

## 免责声明

本系统仅用于法律信息辅助查询与学习，回答基于本地知识库检索生成，可能存在滞后或遗漏，不构成正式法律意见。涉及具体案件请咨询执业律师。
