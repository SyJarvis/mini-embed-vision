"""
C-MTEB task configurations for mini-embed-vision benchmark.

Defines task lists, presets, and metadata for running MTEB benchmarks
on both text (Chinese C-MTEB) and image / multi-modal tasks.
"""

# ============================================================
# Text task lists (Chinese C-MTEB)
# ============================================================

RETRIEVAL_TASKS = [
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "CovidRetrieval",
    "VideoRetrieval",
]

CLASSIFICATION_TASKS = [
    "TNews",
    "IFlyTek",
    "MultilingualSentiment",
    "JDReview",
    "OnlineShopping",
]

STS_TASKS = [
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STS22",
]

CLUSTERING_TASKS = [
    "CLSClusteringS2S",
    "CLSClusteringS2S.v2",
    "ThuNewsClusteringS2S",
]

RERANKING_TASKS = [
    "T2Reranking",
    "CMedQAv2-reranking",
]

# ============================================================
# Image / multi-modal task lists
# ============================================================

IMAGE_CLASSIFICATION_TASKS = [
    "Imagenet1k",               # Imagenet1kClassification (50k val, 1000 classes)
]

IMAGE_RETRIEVAL_TASKS = [
    "Flickr30kT2IRetrieval",    # text → image, 31k corpus
    "Flickr30kI2TRetrieval",    # image → text (optional)
]

# ============================================================
# Presets
# ============================================================

PRESETS = {
    "quick": {
        "description": "5 core Chinese text tasks, quick smoke test",
        "tasks": [
            "T2Retrieval",
            "MMarcoRetrieval",
            "TNews",
            "ATEC",
            "LCQMC",
        ],
    },
    "retrieval": {
        "description": "All Chinese retrieval tasks",
        "tasks": RETRIEVAL_TASKS,
    },
    "full": {
        "description": "All C-MTEB tasks (Chinese text)",
        "tasks": (
            RETRIEVAL_TASKS
            + CLASSIFICATION_TASKS
            + STS_TASKS
            + CLUSTERING_TASKS
            + RERANKING_TASKS
        ),
    },
    # ── Image presets ──
    "imagenet": {
        "description": "ImageNet-1k classification (5 experiments)",
        "tasks": IMAGE_CLASSIFICATION_TASKS,
    },
    "flickr": {
        "description": "Flickr30k text→image retrieval",
        "tasks": IMAGE_RETRIEVAL_TASKS,
    },
    "quick_image": {
        "description": "Quick image tasks: ImageNet + Flickr30k",
        "tasks": IMAGE_CLASSIFICATION_TASKS + IMAGE_RETRIEVAL_TASKS,
    },
}


def get_tasks(preset_or_tasks: str) -> list[str]:
    """
    Return a list of task names from a preset name or comma-separated list.
    """
    if preset_or_tasks in PRESETS:
        return PRESETS[preset_or_tasks]["tasks"]
    return [t.strip() for t in preset_or_tasks.split(",") if t.strip()]


def get_task_info(task_name: str) -> dict:
    """Get the category and description for a single task."""
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
    for name in IMAGE_CLASSIFICATION_TASKS:
        all_tasks[name] = {"category": "image_classification"}
    for name in IMAGE_RETRIEVAL_TASKS:
        all_tasks[name] = {"category": "image_retrieval"}
    return all_tasks.get(task_name, {"category": "unknown"})


def list_all_tasks() -> None:
    """Print all available tasks and presets."""
    print("=== Presets ===")
    for name, preset in PRESETS.items():
        print(f"  {name}: {preset['description']} ({len(preset['tasks'])} tasks)")
    print()
    print("=== By category ===")
    for cat_name, tasks in [
        ("Retrieval (text)", RETRIEVAL_TASKS),
        ("Classification (text)", CLASSIFICATION_TASKS),
        ("STS", STS_TASKS),
        ("Clustering", CLUSTERING_TASKS),
        ("Reranking", RERANKING_TASKS),
        ("Image Classification", IMAGE_CLASSIFICATION_TASKS),
        ("Image Retrieval", IMAGE_RETRIEVAL_TASKS),
    ]:
        print(f"\n  [{cat_name}]")
        for t in tasks:
            print(f"    - {t}")
