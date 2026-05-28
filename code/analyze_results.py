#!/usr/bin/env python3
"""分析实验结果 + 画图"""
import os, sys, glob, json
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_results

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CODE_DIR), "data")
FIGURES_DIR = os.path.join(os.path.dirname(CODE_DIR), "paper", "figures")

COLORS = {"zero_shot": "#4C72B0", "cot": "#55A868", "few_shot": "#C44E52", "structured": "#8172B2"}
DATASET_NAMES = {"medqa": "MedQA", "pubmedqa": "PubMedQA", "medmcqa": "MedMCQA"}


def load_clean_results():
    """只加载 Flash 实验结果"""
    # Try to load the Flash combined results file
    flash_files = sorted(glob.glob(os.path.join(DATA_DIR, "all_results_deepseek-v4-flash_*.jsonl")))
    if not flash_files:
        print("No Flash results found.")
        return []
    latest = flash_files[-1]
    print(f"Loading Flash results: {os.path.basename(latest)}")
    return load_results(latest)


def main():
    print("=" * 60)
    print("DEEPSEEK-V4 MEDICAL QA — RESULTS ANALYSIS")
    print("=" * 60)

    results = load_clean_results()
    if not results:
        return

    # Group by (dataset, strategy)
    groups = defaultdict(lambda: {"correct": 0, "total": 0, "errors": Counter()})
    for r in results:
        ds, strat = r["dataset"], r["strategy"]
        groups[(ds, strat)]["total"] += 1
        if r.get("correct"):
            groups[(ds, strat)]["correct"] += 1
        if not r.get("correct"):
            resp = str(r.get("raw_response", "")).lower()
            question = str(r.get("question", "")).lower()
            pred = str(r.get("prediction", "")).strip()
            resp_words = set(resp.split())
            q_words = set(question.split())
            shared = resp_words & q_words
            resp_len = len(resp.split())

            # E5: True format errors — empty, refusal, API fail
            if not resp or resp_len < 2:
                groups[(ds, strat)]["errors"]["E5: Other"] += 1
            elif any(w in resp for w in ["i'm sorry", "i cannot", "i apologize", "as an ai"]):
                groups[(ds, strat)]["errors"]["E5: Other"] += 1

            # E4: Hallucination — rare, explicit indicators
            elif any(w in resp for w in ["hypothetical", "fictional", "imaginary", "not a real"]):
                groups[(ds, strat)]["errors"]["E4: Hallucination"] += 1

            # E3: Comprehension failure — model output barely overlaps with question
            elif len(shared) < 3 and resp_len >= 5:
                groups[(ds, strat)]["errors"]["E3: Comprehension Error"] += 1

            # E2: Reasoning Error — model engaged deeply but reasoned wrong
            elif resp_len >= 30 and len(shared) >= 4:
                groups[(ds, strat)]["errors"]["E2: Reasoning Error"] += 1

            # E1: Knowledge Gap — brief wrong answer, insufficient reasoning
            else:
                groups[(ds, strat)]["errors"]["E1: Knowledge Gap"] += 1

    # ====== 1. Accuracy Summary ======
    print("\n" + "=" * 60)
    print("1. ACCURACY ANALYSIS")
    print("=" * 60)

    # Build data structures for plotting
    datasets_list = ["medqa", "pubmedqa", "medmcqa"]
    strategies = ["zero_shot", "cot", "few_shot", "structured"]
    acc_matrix = {}  # (ds, strat) -> acc

    for (ds, strat), d in sorted(groups.items()):
        acc = d["correct"] / d["total"] * 100 if d["total"] > 0 else 0
        acc_matrix[(ds, strat)] = acc
        ci = 1.96 * np.sqrt(acc / 100 * (1 - acc / 100) / d["total"]) * 100 if d["total"] > 0 else 0
        bar = "█" * int(acc / 2) + "░" * (50 - int(acc / 2))
        print(f"  {DATASET_NAMES.get(ds, ds):10s} | {strat:12s} | {acc:5.1f}% |{bar}| {d['correct']}/{d['total']}")

    print("\nBEST STRATEGY PER DATASET:")
    for ds in datasets_list:
        best_strat = max(strategies, key=lambda s: acc_matrix.get((ds, s), 0))
        print(f"  {DATASET_NAMES.get(ds, ds)}: {best_strat} ({acc_matrix[(ds, best_strat)]:.1f}%)")

    # ====== 2. Error Analysis ======
    print("\n" + "=" * 60)
    print("2. ERROR ANALYSIS")
    print("=" * 60)
    total_errors = Counter()
    for d in groups.values():
        total_errors += d["errors"]
    for label, count in total_errors.most_common():
        pct = count / sum(total_errors.values()) * 100 if sum(total_errors.values()) > 0 else 0
        print(f"  {label}: {count} ({pct:.1f}%)")

    # ====== 3. Cost Analysis ======
    print("\n" + "=" * 60)
    print("3. COST ANALYSIS")
    print("=" * 60)
    total_correct = sum(d["correct"] for d in groups.values())
    total_questions = sum(d["total"] for d in groups.values())
    # Estimate cost: each query ~500 input + 300 output tokens
    estimated_input_tokens = total_questions * 500
    estimated_output_tokens = total_questions * 300
    total_cost = (estimated_input_tokens / 1_000_000) * 1.0 + (estimated_output_tokens / 1_000_000) * 2.0
    print(f"  Questions: {total_questions}")
    print(f"  Accuracy: {total_correct/total_questions*100:.1f}%")
    print(f"  Estimated cost: ¥{total_cost:.2f}")
    print(f"  Cost per correct answer: ¥{total_cost/total_correct:.4f}" if total_correct > 0 else "")

    # ====== 4. Generate Figures ======
    print("\n" + "=" * 60)
    print("4. GENERATING FIGURES")
    print("=" * 60)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Figure 1: Prompt comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(datasets_list))
    width = 0.2
    for i, strat in enumerate(strategies):
        scores = [acc_matrix.get((ds, strat), 0) for ds in datasets_list]
        bars = ax.bar(x + i * width - width * 1.5, scores, width, label=strat, color=COLORS.get(strat, "#333"))
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{score:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Prompting Strategy Comparison Across Medical QA Benchmarks", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_NAMES.get(d, d) for d in datasets_list])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "prompt_comparison.pdf"), dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {os.path.join(FIGURES_DIR, 'prompt_comparison.pdf')}")
    plt.close()

    # Figure 2: Error distribution — horizontal bar chart (clean, no label overlap)
    error_labels = ["E1: Knowledge Gap", "E2: Reasoning Error", "E3: Comprehension Error",
                    "E4: Hallucination", "E5: Other"]
    error_values = [total_errors.get(l, 0) for l in error_labels]
    error_colors = ["#E24A33", "#988ED5", "#348ABD", "#888888", "#FBC15E"]

    if sum(error_values) > 0:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        y_pos = range(len(error_labels) - 1, -1, -1)
        bars = ax.barh(y_pos, error_values, height=0.6, color=error_colors, edgecolor="white", linewidth=1)

        for i, (bar, val) in enumerate(zip(bars, error_values)):
            if val > 0:
                pct = val / sum(error_values) * 100
                ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                        f"{pct:.1f}% ({val})", va="center", fontsize=11, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(error_labels, fontsize=11)
        ax.set_xlabel("Number of Errors", fontsize=12)
        ax.set_title("Distribution of Error Types in DeepSeek-V4 Flash Medical Reasoning", fontsize=13)
        ax.set_xlim(0, max(error_values) * 1.25)
        ax.grid(axis="x", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "error_distribution.pdf"), dpi=150, bbox_inches="tight")
        print(f"  Figure saved: {os.path.join(FIGURES_DIR, 'error_distribution.pdf')}")
        plt.close()

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
