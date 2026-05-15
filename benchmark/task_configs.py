"""
C-MTEB task configurations for Chinese text embedding evaluation.

Defines task lists, presets, and metadata for running MTEB benchmarks.
"""

# ============================================================
# 完整任务列表（按类别）
# ============================================================

RETRIEVAL_TASKS = [
    "T2Retrieval",          # 大规模中文段落检索
    "MMarcoRetrieval",      # 中文 MS MARCO
    "DuRetrieval",          # 百度搜索引擎查询
    "CmedqaRetrieval",      # 中文医学问答检索
    "EcomRetrieval",        # 电商商品检索
    "MedicalRetrieval",     # 医学文献检索
    "CovidRetrieval",       # COVID 文献检索
    "VideoRetrieval",       # 视频检索
]

CLASSIFICATION_TASKS = [
    "TNews",                # 新闻分类 (15类)
    "IFlyTek",              # 应用描述分类 (119类)
    "MultilingualSentiment",# 多语言情感分类
    "JDReview",             # 京东评论情感
    "OnlineShopping",       # 在线购物评论
]

STS_TASKS = [
    "ATEC",                 # 蚂蚁金服语义相似度
    "BQ",                   # 银行问答相似度
    "LCQMC",                # 问题复述识别
    "PAWSX",                # 句子相似度
    "STS22",                # 多语言 STS
]

CLUSTERING_TASKS = [
    "CLSClusteringS2S",     # 学术论文聚类
    "CLSClusteringS2S.v2",  # 学术论文聚类 v2
    "ThuNewsClusteringS2S", # 新闻聚类
]

RERANKING_TASKS = [
    "T2Reranking",          # T2 重排序
    "CMedQAv2-reranking",   # 中文医学问答重排序
]

# ============================================================
# 预设配置
# ============================================================

PRESETS = {
    "quick": {
        "description": "5个核心任务，快速验证",
        "tasks": [
            "T2Retrieval",
            "MMarcoRetrieval",
            "TNews",
            "ATEC",
            "LCQMC",
        ],
    },
    "retrieval": {
        "description": "仅检索类任务",
        "tasks": RETRIEVAL_TASKS,
    },
    "full": {
        "description": "全部 C-MTEB 任务",
        "tasks": (
            RETRIEVAL_TASKS
            + CLASSIFICATION_TASKS
            + STS_TASKS
            + CLUSTERING_TASKS
            + RERANKING_TASKS
        ),
    },
}


def get_tasks(preset_or_tasks: str) -> list[str]:
    """
    根据 preset 名称或逗号分隔的任务名列表获取任务列表。

    Args:
        preset_or_tasks: "quick" / "full" / "retrieval"，或逗号分隔的任务名

    Returns:
        任务名列表
    """
    if preset_or_tasks in PRESETS:
        return PRESETS[preset_or_tasks]["tasks"]
    # 视为逗号分隔的任务名
    return [t.strip() for t in preset_or_tasks.split(",") if t.strip()]


def get_task_info(task_name: str) -> dict:
    """获取单个任务的类别和描述。"""
    all_tasks = {}
    for name in RETRIEVAL_TASKS:
        all_tasks[name] = {"category": "retrieval"}
    for name in CLASSIFICATION_TASKS:
        all_tasks[name] = {"category": "classification"}
    for name in STS_TASKS:
        all_tasks[name] = {"category": "sts"}
    for name in CLUSTERING_TASKS:
        all_tasks[name] = {"category": "clustering"}
    for name in RERANKING_TASKS:
        all_tasks[name] = {"category": "reranking"}
    return all_tasks.get(task_name, {"category": "unknown"})


def list_all_tasks() -> None:
    """打印所有可用的任务和预设。"""
    print("=== 预设配置 ===")
    for name, preset in PRESETS.items():
        print(f"  {name}: {preset['description']} ({len(preset['tasks'])} tasks)")
    print()
    print("=== 按类别 ===")
    for cat_name, tasks in [
        ("Retrieval", RETRIEVAL_TASKS),
        ("Classification", CLASSIFICATION_TASKS),
        ("STS", STS_TASKS),
        ("Clustering", CLUSTERING_TASKS),
        ("Reranking", RERANKING_TASKS),
    ]:
        print(f"\n  [{cat_name}]")
        for t in tasks:
            print(f"    - {t}")
