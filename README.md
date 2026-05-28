# DeepSeek-V4 Medical QA Evaluation

Code for: *DeepSeek-V4 in Clinical Reasoning: Flash vs. Pro and the Scale Reversal in Medical Question Answering*

## Quick Start

```bash
pip install -r requirements.txt
cp code/config.example.py code/config.py   # then fill in your API key
python code/run_experiments.py --max-samples 10   # quick test
python code/run_experiments.py                     # full run
python code/analyze_results.py                     # generate figures
```

## Datasets

All benchmarks load automatically from HuggingFace at runtime:

| Dataset | Path | Questions |
|---|---|---|
| MedQA | `GBaker/MedQA-USMLE-4-options` | 1,273 |
| PubMedQA | `pubmed_qa` (pqa_labeled) | 500 |
| MedMCQA | `medmcqa` (validation) | 4,183 |

## Cost (DeepSeek V4-Flash)

| Benchmark | Queries | Cost |
|---|---|---|
| MedQA × 4 strategies | 5,092 | ~¥6 |
| PubMedQA × 4 strategies | 2,000 | ~¥3 |
| MedMCQA × 4 strategies (1,000 sampled) | 4,000 | ~¥5 |
| **Total** | **11,092** | **~¥16** |

## Files

```
code/
├── run_experiments.py      # main evaluation script
├── analyze_results.py       # analysis & figures
├── prompts.py               # 4 prompt templates
├── utils.py                 # answer extraction
├── config.example.py        # copy to config.py and fill in API key
├── analyze_pro.py           # Pro model analysis
└── requirements.txt
paper/                        # LaTeX source for the paper
```

## Citation

```bibtex
@article{lui2026deepseekv4,
  title={DeepSeek-V4 in Clinical Reasoning: Flash vs. Pro and the Scale Reversal in Medical Question Answering},
  author={LUI, Chun},
  year={2026}
}
```
