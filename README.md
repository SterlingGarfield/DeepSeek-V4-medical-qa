# DeepSeek-V4 Medical QA Evaluation

Code for the paper: *DeepSeek-V4 in Clinical Reasoning: Flash vs. Pro and the Scale Reversal in Medical Question Answering*

> [中文 README](README_zh.md)

## Setup

```bash
pip install -r requirements.txt
cp config.example.py config.py   # edit config.py to add your DeepSeek API key
```

## Run Experiments

```bash
# Quick test (10 questions per config, ~¥0.13)
python run_experiments.py --max-samples 10

# Full Flash evaluation (3 benchmarks × 4 strategies, ~¥16)
python run_experiments.py --cost-limit 35

# Pro comparison (MedQA only, ~¥27)
python run_experiments.py --datasets medqa --model deepseek-v4-pro --cost-limit 35

# Resume interrupted run
python run_experiments.py --resume --cost-limit 35
```

## Analyze Results

```bash
python analyze_results.py    # Flash: accuracy + error distribution + figures
python analyze_pro.py        # Pro: accuracy + error distribution
```

## Datasets

All benchmarks load from HuggingFace automatically at runtime:

| Dataset | Source | Questions | Format |
|---|---|---|---|
| MedQA | `GBaker/MedQA-USMLE-4-options` (test) | 1,273 | 4-choice |
| PubMedQA | `pubmed_qa` (pqa_labeled) | 500 | yes/no/maybe |
| MedMCQA | `medmcqa` (validation) | 1,000 | 4-choice |

## Key Results

| Benchmark | Best Strategy | Flash Accuracy | Pro Accuracy |
|---|---|---|---|
| MedQA | Few-shot | **71.4%** | 41.3% |
| PubMedQA | CoT | **70.2%** | — |
| MedMCQA | Few-shot | **63.0%** | — |

Flash error distribution: reasoning errors **56.9%**, knowledge gaps 6.2%.

## Cost

| Model | Queries | Tokens (in/out) | Cost |
|---|---|---|---|
| V4-Flash | 11,092 | 3.56M / 6.42M | ¥16.39 |
| V4-Pro (MedQA) | 5,092 | 1.68M / 3.72M | ¥27.35 |
| **Total** | **16,184** | **5.24M / 10.14M** | **¥43.74** |

## Files

```
code/
├── run_experiments.py      # main evaluation script
├── prompts.py               # four prompt templates
├── utils.py                 # answer extraction and file I/O
├── config.example.py        # copy to config.py and fill in API key
├── analyze_results.py       # Flash analysis and figures
├── analyze_pro.py           # Pro analysis
└── requirements.txt
```

## Citation

```bibtex
@article{lui2026deepseekv4,
  title={DeepSeek-V4 in Clinical Reasoning: Flash vs. Pro and the Scale Reversal in Medical Question Answering},
  author={LUI, Chun},
  year={2026}
}
```
