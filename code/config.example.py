"""实验配置文件。使用前复制为 config.py 并填入 API Key。"""

import os

# DeepSeek API（在此填入你的 Key）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 模型选择
MODEL_FLASH = "deepseek-v4-flash"
MODEL_PRO = "deepseek-v4-pro"

# 推理参数
TEMPERATURE = 0.0       # 确定性输出
MAX_TOKENS = 1024
TOP_P = 1.0

# 各数据集的加载路径和采样量
DATASET_CONFIGS = {
    "medqa": {
        "hf_path": "GBaker/MedQA-USMLE-4-options",
        "split": "test",
        "max_samples": None,        # 全量 1273 题
    },
    "pubmedqa": {
        "hf_path": "pubmed_qa",
        "split": "train",
        "config": "pqa_labeled",
        "max_samples": 500,
    },
    "medmcqa": {
        "hf_path": "medmcqa",
        "split": "validation",
        "max_samples": 1000,        # 从 4183 中抽样
    },
}

# 四种提示策略
PROMPT_STRATEGIES = ["zero_shot", "cot", "few_shot", "structured"]

# 输出路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

SEED = 42
