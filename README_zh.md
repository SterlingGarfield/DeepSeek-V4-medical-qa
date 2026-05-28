# DeepSeek-V4 医学问答评测

论文代码：*DeepSeek-V4 in Clinical Reasoning: Flash vs. Pro and the Scale Reversal in Medical Question Answering*

> [English README](README.md)

## 环境配置

```bash
pip install -r requirements.txt
cp config.example.py config.py   # 编辑 config.py，填入 DeepSeek API Key
```

## 运行实验

```bash
# 快速测试（每配置 10 题，约 ¥0.13）
python run_experiments.py --max-samples 10

# Flash 完整评测（3 基准 × 4 策略，约 ¥16）
python run_experiments.py --cost-limit 35

# Pro 对比（仅 MedQA，约 ¥27）
python run_experiments.py --datasets medqa --model deepseek-v4-pro --cost-limit 35

# 续跑中断的实验
python run_experiments.py --resume --cost-limit 35
```

## 分析结果

```bash
python analyze_results.py    # Flash：准确率 + 错误分布 + 图表
python analyze_pro.py        # Pro：准确率 + 错误分布
```

## 数据集

所有基准通过 HuggingFace 自动加载：

| 数据集 | 来源 | 题目数 | 题型 |
|---|---|---|---|
| MedQA | `GBaker/MedQA-USMLE-4-options` (test) | 1,273 | 四选一 |
| PubMedQA | `pubmed_qa` (pqa_labeled) | 500 | 是/否/可能 |
| MedMCQA | `medmcqa` (validation) | 1,000 | 四选一 |

## 核心结果

| 基准 | 最优策略 | Flash 准确率 | Pro 准确率 |
|---|---|---|---|
| MedQA | Few-shot | **71.4%** | 41.3% |
| PubMedQA | CoT | **70.2%** | — |
| MedMCQA | Few-shot | **63.0%** | — |

Flash 错误分布：推理错误占 **56.9%**，知识缺失仅 6.2%。

## 成本

| 模型 | API 调用 | Token（输入/输出） | 成本 |
|---|---|---|---|
| V4-Flash | 11,092 | 3.56M / 6.42M | ¥16.39 |
| V4-Pro（仅 MedQA） | 5,092 | 1.68M / 3.72M | ¥27.35 |
| **合计** | **16,184** | **5.24M / 10.14M** | **¥43.74** |

## 文件说明

```
code/
├── run_experiments.py      # 主评测脚本
├── prompts.py               # 四种提示策略模板
├── utils.py                 # 答案提取与文件读写
├── config.example.py        # 复制为 config.py 并填入 API Key
├── analyze_results.py       # Flash 分析与图表生成
├── analyze_pro.py           # Pro 分析
└── requirements.txt
```

## 引用

```bibtex
@article{lui2026deepseekv4,
  title={DeepSeek-V4 in Clinical Reasoning: Flash vs. Pro and the Scale Reversal in Medical Question Answering},
  author={LUI, Chun},
  year={2026}
}
```
