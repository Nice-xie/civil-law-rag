# -*- coding: utf-8 -*-
"""数据分析 / 数据挖掘 / 可视化模块。

本模块只提供纯函数，不直接访问数据库：
app.py 从 MySQL 读取 chat_history 后，把数据传入这里的函数。
重依赖（pandas / sklearn / matplotlib / jieba / pyecharts）均采用函数内惰性导入，
避免影响 app.py 启动与核心问答功能。
"""
import os
import re
from collections import Counter

# ============ 1. 规则关键词分类器 ============
CATEGORIES = ["民间借贷", "婚约彩礼", "劳动纠纷", "合同纠纷", "民事诉讼程序"]

KEYWORDS = {
    "民间借贷": [
        "借条", "欠条", "借款", "借贷", "借钱", "利息", "利率", "本金", "逾期",
        "出借", "放贷", "砍头息", "转贷", "高利贷", "还款", "贷款",
    ],
    "婚约彩礼": [
        "彩礼", "婚约", "订婚", "悔婚", "恋爱", "结婚", "离婚", "共同生活",
        "婚姻", "返还",
    ],
    "劳动纠纷": [
        "工资", "辞退", "离职", "加班", "社保", "试用期", "经济补偿", "赔偿金",
        "竞业", "工伤", "双倍工资", "劳动合同", "无固定期限", "仲裁", "劳动",
    ],
    "合同纠纷": [
        "合同", "违约", "定金", "买卖", "租赁", "退款", "履行",
        "解约", "发货", "交付", "卖房", "买房", "购房", "房屋", "房产",
    ],
    "民事诉讼程序": [
        "管辖", "法院", "时效", "举证", "证据", "起诉", "判决", "保全", "执行",
        "缺席", "上诉", "支付令", "调解", "自认", "一审", "二审",
    ],
}

STOPWORDS = set("""
的 了 我 你 他 她 它 吗 呢 啊 吧 是 在 有 和 与 及 或 对 把 被 让 请 问
怎么 什么 如何 可以 能 要 会 还 就 都 这 那 一个 没有 多少 是否 已经 如果
因为 所以 但是 然后 以及 等等 相关 规定 法律 法院 民事 诉讼 应该 需要 时候
""".split())


def classify(text):
    """把一句法律咨询归入一个类别（按关键词命中数计分，平局取列表靠前的类别）。"""
    if not text:
        return "其他"
    t = text.lower()
    scores = {c: 0 for c in CATEGORIES}
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            if kw in t:
                scores[cat] += 1
    best = max(scores, key=lambda c: (scores[c], -CATEGORIES.index(c)))
    return best if scores[best] > 0 else "其他"


def _top_keywords(texts, topn=60):
    """jieba 分词 + 停用词过滤 + 词频统计，返回 [{name, value}, ...]。"""
    import jieba
    counter = Counter()
    for t in texts:
        for w in jieba.lcut(t or ""):
            w = w.strip()
            if (len(w) >= 2 and w not in STOPWORDS
                    and not re.fullmatch(r"[0-9a-zA-Z]+", w)):
                counter[w] += 1
    return [{"name": w, "value": c} for w, c in counter.most_common(topn)]


# ============ 2. 统计聚合（供看板） ============
def aggregate(rows):
    """rows: [{user_query, create_time}, ...] -> 看板数据字典。"""
    if not rows:
        return {"total": 0, "categories": [], "trend": [], "keywords": [],
                "category_keywords": {}}

    import pandas as pd

    df = pd.DataFrame(rows)
    df["category"] = df["user_query"].map(classify)

    # 分类分布（保持固定顺序，含 0 值类别）
    all_cats = CATEGORIES + ["其他"]
    counts = df["category"].value_counts().reindex(all_cats).fillna(0).astype(int)
    categories = [{"name": k, "value": int(v)} for k, v in counts.items()]

    # 时间趋势：按“日期 × 分类”统计
    df["date"] = pd.to_datetime(df["create_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    trend = (
        df.dropna(subset=["date"])
        .groupby(["date", "category"]).size()
        .reset_index(name="value")
        .to_dict("records")
    )

    # 全局高频词
    keywords = _top_keywords(df["user_query"])

    # 每类高频词（供前端多图表联动）
    category_keywords = {}
    for cat in all_cats:
        sub = df[df["category"] == cat]["user_query"]
        category_keywords[cat] = _top_keywords(sub, topn=30) if len(sub) else []

    return {
        "total": int(len(df)),
        "categories": categories,
        "trend": trend,
        "keywords": keywords,
        "category_keywords": category_keywords,
    }


# ============ 3. 数据挖掘：TF-IDF + KMeans 聚类 ============
def run_clustering(questions, out_dir=None):
    """对提问文本做 TF-IDF + KMeans 聚类，返回评估指标并保存 Matplotlib 图。"""
    import numpy as np
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    n = len(questions)
    if n < 3:
        return {"error": "数据量不足：至少需要 3 条提问才能聚类，请先运行 seed_data.py 灌入测试数据"}

    corpus = [" ".join(jieba.lcut(q)) for q in questions]
    vec = TfidfVectorizer(max_features=2000)
    X = vec.fit_transform(corpus)

    # 肘部法 + 轮廓系数，遍历 k 选择较优聚类数
    kmax = min(8, n)
    ks = list(range(2, kmax + 1))
    inertias, sil = [], []
    for kk in ks:
        km = KMeans(n_clusters=kk, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_) if kk < n else 0.0)
    best_k = ks[int(np.argmax(sil))]

    final = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(X)
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X.toarray())

    # 图1：肘部曲线
    plt.figure(figsize=(6, 4))
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("聚类数 k")
    plt.ylabel("惯性（inertia）")
    plt.title("肘部法则")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "elbow.png"), dpi=110)
    plt.close()

    # 图2：轮廓系数曲线
    plt.figure(figsize=(6, 4))
    plt.plot(ks, sil, marker="s", color="crimson")
    plt.xlabel("聚类数 k")
    plt.ylabel("轮廓系数")
    plt.title("轮廓系数曲线")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "silhouette.png"), dpi=110)
    plt.close()

    # 图3：PCA 二维投影聚类散点
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=final.labels_, cmap="tab10", s=70, alpha=0.85)
    for i, q in enumerate(questions):
        plt.annotate(q[:5], (X2[i, 0], X2[i, 1]), fontsize=7, alpha=0.7)
    plt.colorbar(sc, label="簇")
    plt.title(f"提问文本聚类（k={best_k}，PCA 二维投影）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "clusters.png"), dpi=110)
    plt.close()

    # 每个簇的代表词（取聚类中心的 TF-IDF 高权重词）
    terms = vec.get_feature_names_out()
    cluster_terms = {}
    for ci in range(best_k):
        idx = np.argsort(final.cluster_centers_[ci])[::-1][:10]
        cluster_terms[f"簇{ci + 1}"] = [str(terms[i]) for i in idx]

    return {
        "n": n,
        "k": best_k,
        "silhouette": round(float(sil[best_k - 2]), 4),
        "sizes": [int((final.labels_ == ci).sum()) for ci in range(best_k)],
        "cluster_terms": cluster_terms,
        "elbow": "elbow.png",
        "silhouette_img": "silhouette.png",
        "scatter": "clusters.png",
    }


# ============ 4. Pyecharts 交互式图表 ============
def _embed_fragment(chart):
    """render_embed() 返回的是完整 HTML 文档，这里只取 <body> 内的图表片段，
    并去掉自带的 echarts CDN <script>（前端页面已在 <head> 引入 echarts）。"""
    html = chart.render_embed()
    m = re.search(r"<body[^>]*>(.*)</body>", html, re.S)
    frag = m.group(1) if m else html
    return re.sub(r'<script[^>]*echarts[^>]*></script>', '', frag)


def build_pyecharts(stats):
    """根据统计结果生成 Pyecharts 图表 HTML 片段（雷达图 / 仪表盘 / 漏斗图）。"""
    total = stats.get("total", 0)
    items = [(c["name"], c["value"]) for c in stats.get("categories", []) if c["value"] > 0]
    if not items:
        return {"empty": True, "radar": "", "gauge": "", "funnel": ""}

    from pyecharts.charts import Radar, Gauge, Funnel
    from pyecharts import options as opts

    names = [n for n, _ in items]
    values = [v for _, v in items]
    maxv = max(values) if values else 1

    radar = (
        Radar(init_opts=opts.InitOpts(width="100%", height="420px"))
        .add_schema(schema=[opts.RadarIndicatorItem(name=n, max_=maxv) for n in names])
        .add("咨询分类分布", [values], color="#2f6fed",
             areastyle_opts=opts.AreaStyleOpts(opacity=0.3))
    )

    gauge = (
        Gauge(init_opts=opts.InitOpts(width="100%", height="340px"))
        .add("", [("提问总量", total)], min_=0, max_=max(total * 2, 10))
    )

    funnel = (
        Funnel(init_opts=opts.InitOpts(width="100%", height="340px"))
        .add("分类占比", items, sort_="descending")
        .set_global_opts(legend_opts=opts.LegendOpts(pos_left="10%"))
    )

    return {
        "empty": False,
        "radar": _embed_fragment(radar),
        "gauge": _embed_fragment(gauge),
        "funnel": _embed_fragment(funnel),
    }
