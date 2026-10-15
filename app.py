import warnings
import os
import uuid
import re
# HuggingFace国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request, redirect, url_for, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
import pymysql

app = Flask(__name__)

# ========= 1. 加载知识库，构建向量库 =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LAW_DIR = os.path.join(BASE_DIR, "law_docs")
CHUNK_SIZE = 500  # 每个检索分片的目标字符上限（按法条边界合并，绝不跨条截断）

ART_START = re.compile(r"^第[一二三四五六七八九十百千零〇]+条")
CITE_LINE = re.compile(r"^《[^》]+》\s*第[一二三四五六七八九十百千零〇]+条")
HDR_RE = re.compile(r"^第[一二三四五六七八九十百千零〇]+(?:分编|[编章节])")
LAW_NAME_RE = re.compile(r"^《[^》]+》$")
SECTION_RE = re.compile(r"^[一二三四五六七八九十]+、")


def split_law_text(text):
    """按法条边界切分法律文本为检索分片。

    - 一条法条（含其后续款/项）作为原子单位，绝不跨条截断；
    - 相邻短条文合并到接近 CHUNK_SIZE；
    - 分片开头标注法律名称与所属编/分编/章/节，便于模型正确引用与溯源。
    """
    chunks = []
    law_name = ""   # 当前法律名（独立《...》行，如《中华人民共和国民法典》）
    bian = fenbian = zhang = jie = section = ""
    buf = []
    buf_len = 0
    cur = None      # 当前条文累积文本（含款/项续行）

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        prefix = []
        if law_name:
            prefix.append(law_name)
        header = " · ".join(x for x in (bian, fenbian, zhang, jie, section) if x)
        if header:
            prefix.append(header)
        head = "\n".join(prefix) + "\n" if prefix else ""
        chunks.append(head + "\n".join(buf))
        buf = []
        buf_len = 0

    def add_unit(t):
        nonlocal buf, buf_len
        if buf and buf_len + len(t) + 1 > CHUNK_SIZE:
            flush()
        buf.append(t)
        buf_len += len(t) + 1

    def push_cur():
        nonlocal cur
        if cur is not None and cur.strip():
            add_unit(cur.strip())
        cur = None

    for raw in text.split("\n"):
        s = raw.strip()
        if not s:
            continue
        if re.match(r"^【[^】]+】$", s):   # 忽略语料中的【第X编 XXX】等重复标记行
            continue
        if SECTION_RE.match(s):          # 司法解释的“一、xxx”分部标题
            push_cur()
            flush()
            section = s
            bian = fenbian = zhang = jie = ""
            continue
        if HDR_RE.match(s) or s == "附则":
            push_cur()
            flush()
            section = ""
            level = re.match(r"^第[一二三四五六七八九十百千零〇]+(分编|[编章节])", s)
            if s == "附则":
                bian, fenbian, zhang, jie = s, "", "", ""
            elif level and level.group(1) == "编":
                bian, fenbian, zhang, jie = s, "", "", ""
            elif level and level.group(1) == "分编":
                fenbian, zhang, jie = s, "", ""
            elif level and level.group(1) == "章":
                zhang, jie = s, ""
            elif level and level.group(1) == "节":
                jie = s
            continue
        if s.startswith("#"):          # civil.txt 的案例库/法条库分隔标题
            push_cur()
            flush()
            bian = fenbian = zhang = jie = section = ""
            continue
        if LAW_NAME_RE.match(s):       # 独立法律名行（文件首行）
            push_cur()
            flush()
            law_name = s
            bian = fenbian = zhang = jie = section = ""
            continue
        if CITE_LINE.match(s):         # 《法律名》第X条（内容在后续行）
            push_cur()
            law_name = ""              # 法条自带完整引用，前缀不再加文件级法律名
            cur = s
            continue
        if ART_START.match(s):         # 第X条 内容（同行）
            push_cur()
            cur = s
            continue
        # 款/项/续行、案例、散段落
        if cur is not None:
            cur = cur + "\n" + s
        else:
            add_unit(s)
    push_cur()
    flush()
    return chunks


law_files = sorted(f for f in os.listdir(LAW_DIR) if f.endswith(".txt"))
print(f"开始加载知识库（共 {len(law_files)} 个文件）...")
all_chunks = []
for fname in law_files:
    path = os.path.join(LAW_DIR, fname)
    with open(path, encoding="utf-8") as f:
        text = f.read()
    chunks = split_law_text(text)
    all_chunks.extend(chunks)
    print(f"  {fname}: {len(text)} 字符 -> {len(chunks)} 个分片")
print(f"知识库切分完成，共 {len(all_chunks)} 个分片")
print("开始加载向量模型...")
from langchain_community.embeddings import HuggingFaceEmbeddings
model_name = "BAAI/bge-base-zh-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}


class BGEHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    """BGE 中文向量模型：查询需加指令前缀、文档不加，以获得最佳中文检索效果。"""
    query_instruction = "为这个句子生成表示以用于检索相关文章："

    def embed_query(self, text: str):
        return super().embed_query(self.query_instruction + text)


embedding = BGEHuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("向量模型加载完毕！")
print("构建FAISS向量库...")
vector_db = FAISS.from_texts(all_chunks, embedding)
print("FAISS向量库构建成功！")
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
llm = Ollama(model="qwen:7b")
print("Ollama qwen:7b加载完成！")

# ========= 自定义Prompt模板（含多轮对话历史） =========
PROMPT_TEMPLATE = """你是民事诉讼智能辅助助手，必须优先使用下面检索返回的法条和案例作答。
1. 回答必须写明引用的法律名称+具体条款；
2. 结合案情和检索到的法条/案例，给出具体、有针对性的法律分析；
3. 如果检索资料没有对应法条，直接回复：当前知识库暂无相关法条，本系统仅作参考，建议咨询执业律师。
4. 禁止脱离检索文档凭空推断，禁止编造法律条文。
5. 结尾标注：本系统仅为法律辅助工具，不能替代执业律师。

对话历史（用于理解上下文，例如“那如果没领证呢”等指代）：
{history}

参考资料（本轮检索到的法条/案例）：
{context}

用户当前问题：{question}
"""


def format_history(history):
    """把前端传来的对话历史格式化成文本，最多保留最近 12 条消息，避免 prompt 过长。"""
    if not history:
        return "（无，这是本轮对话的第一个问题）"
    recent = history[-12:]
    lines = []
    for item in recent:
        role = "用户" if item.get("role") == "user" else "助手"
        lines.append(f"{role}：{item.get('content', '')}")
    return "\n".join(lines)


def rag_ask(question, history):
    """多轮 RAG 问答：检索 + 拼历史 + 调 LLM，返回 (答案, 参考法条列表)。"""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = PROMPT_TEMPLATE.format(
        history=format_history(history),
        context=context,
        question=question,
    )
    answer = llm.invoke(prompt).strip()
    sources = [{"content": doc.page_content} for doc in docs]
    return answer, sources


# ========= 2. MySQL数据库保存对话历史（按会话分组） =========
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "civil_law_rag_db",
    "charset": "utf8mb4",
}


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def save_chat(question, answer, session_id):
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        sql = "INSERT INTO chat_history(user_query, ai_answer, session_id) VALUES(%s,%s,%s)"
        cursor.execute(sql, (question, answer, session_id))
        conn.commit()
    except Exception as e:
        print("保存历史记录出错：", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_conversations(page=1, per_page=10):
    """分页读取会话列表（按 session_id 分组），返回 (会话列表, 总会话数)。"""
    conv_list = []
    total = 0
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT COUNT(DISTINCT session_id) AS cnt FROM chat_history")
        total = cursor.fetchone()["cnt"]
        offset = (page - 1) * per_page
        sql = """
            SELECT c.session_id, c.turns, c.last_time, first.user_query AS title
            FROM (
                SELECT session_id, COUNT(*) AS turns, MAX(create_time) AS last_time, MIN(id) AS first_id
                FROM chat_history GROUP BY session_id
            ) c
            JOIN chat_history first ON first.id = c.first_id
            ORDER BY c.first_id DESC
            LIMIT %s OFFSET %s
        """
        cursor.execute(sql, (per_page, offset))
        conv_list = cursor.fetchall()
    except Exception as e:
        print("读取历史出错：", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return conv_list, total


def get_conversation(session_id):
    """读取某个会话的全部对话轮次（按时间正序）。"""
    turns = []
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute(
            "SELECT user_query, ai_answer, create_time FROM chat_history WHERE session_id=%s ORDER BY id ASC",
            (session_id,),
        )
        turns = cursor.fetchall()
    except Exception as e:
        print("读取会话详情出错：", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return turns


def delete_conversation(session_id):
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history WHERE session_id=%s", (session_id,))
        conn.commit()
    except Exception as e:
        print("删除会话出错：", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def clear_history():
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_history")
        conn.commit()
    except Exception as e:
        print("清空历史出错：", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def _fetch_all_questions():
    """读取全部提问及其时间，供分析模块使用。"""
    rows = []
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
        cursor.execute("SELECT user_query, create_time FROM chat_history ORDER BY id ASC")
        rows = cursor.fetchall()
    except Exception as e:
        print("读取提问数据出错：", e)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    return rows


# ========= 3. Flask路由 =========
def build_page_items(page, total_pages, window=2):
    """生成带省略号的分页页码序列。"""
    if total_pages <= 1:
        return [1]
    items = [1]
    start = max(2, page - window)
    end = min(total_pages - 1, page + window)
    if start > 2:
        items.append("...")
    for p in range(start, end + 1):
        items.append(p)
    if end < total_pages - 1:
        items.append("...")
    items.append(total_pages)
    return items


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    """多轮问答接口：前端以 JSON 传入 question / session_id / history。"""
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = (data.get("session_id") or "").strip() or str(uuid.uuid4())
    history = data.get("history") or []
    if not question:
        return jsonify({"error": "问题不能为空"}), 400
    try:
        answer, sources = rag_ask(question, history)
        save_chat(question, answer, session_id)
        return jsonify({"answer": answer, "sources": sources, "session_id": session_id})
    except Exception as e:
        print("问答出错：", e)
        return jsonify({
            "error": "问答服务暂时不可用，请确认 Ollama 已启动、且已安装 qwen:7b 模型后重试。"
        }), 500


@app.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 10, type=int)
    if page < 1:
        page = 1
    if per_page < 1 or per_page > 100:
        per_page = 10
    conversations, total = get_conversations(page, per_page)
    total_pages = max(1, (total + per_page - 1) // per_page)
    if page > total_pages:
        page = total_pages
        conversations, total = get_conversations(page, per_page)
    page_items = build_page_items(page, total_pages)
    return render_template(
        "history.html",
        conversations=conversations,
        page=page,
        per_page=per_page,
        total=total,
        total_pages=total_pages,
        page_items=page_items,
    )


@app.route("/history/<session_id>")
def history_detail(session_id):
    turns = get_conversation(session_id)
    if not turns:
        return redirect(url_for("history"))
    title = turns[0]["user_query"]
    return render_template("history_detail.html", turns=turns, session_id=session_id, title=title)


@app.route("/delete/<session_id>", methods=["POST"])
def delete_record(session_id):
    delete_conversation(session_id)
    return redirect(url_for("history"))


@app.route("/clear_history", methods=["POST"])
def clear_all():
    clear_history()
    return redirect(url_for("history"))


# ========= 4. 数据分析 / 数据挖掘 / 可视化 =========
def _analysis_dep_error_html(exc):
    """分析依赖缺失时的友好提示（HTML 页面用）。"""
    return (
        "<h2 style='font-family:sans-serif'>缺少数据分析依赖</h2>"
        "<p style='font-family:sans-serif'>请先安装："
        "<code>pip install pandas numpy scikit-learn matplotlib jieba pyecharts</code></p>"
        f"<pre>{exc}</pre>"
    ), 500


@app.route("/stats")
def stats_page():
    return render_template("stats.html")


@app.route("/stats/api")
def stats_api():
    try:
        from analysis import aggregate
    except ImportError as e:
        return jsonify({
            "error": "缺少数据分析依赖，请先安装 pandas numpy scikit-learn matplotlib jieba pyecharts",
            "detail": str(e),
        }), 500
    rows = _fetch_all_questions()
    return jsonify(aggregate(rows))


@app.route("/mining")
def mining_page():
    try:
        from analysis import run_clustering
    except ImportError as e:
        return _analysis_dep_error_html(e)
    rows = _fetch_all_questions()
    questions = [r["user_query"] for r in rows]
    result = run_clustering(questions)
    return render_template("mining.html", result=result)


@app.route("/pyecharts")
def pyecharts_page():
    try:
        from analysis import aggregate, build_pyecharts
    except ImportError as e:
        return _analysis_dep_error_html(e)
    rows = _fetch_all_questions()
    stats = aggregate(rows)
    charts = build_pyecharts(stats)
    return render_template("pyecharts.html", charts=charts, stats=stats)


if __name__ == "__main__":
    print("Flask服务启动：http://127.0.0.1:5000")
    # debug=True 会启用重载器导致向量模型/FAISS 加载两次，故关闭
    app.run(debug=False)
