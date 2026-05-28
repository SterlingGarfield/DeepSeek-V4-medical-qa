#!/usr/bin/env python3
"""批量调用 DeepSeek-V4 跑医学 QA 评测。

用法举例：
    python run_experiments.py                              # 全量跑
    python run_experiments.py --datasets medqa --strategies zero_shot  # 指定数据集+策略
    python run_experiments.py --max-samples 50              # 先跑 50 题试水
    python run_experiments.py --cost-limit 35 --resume      # 设成本上限 + 续跑
"""
import os
import sys
import time
import json
import glob
import argparse
from datetime import datetime
from collections import defaultdict

from openai import OpenAI, APIStatusError, APITimeoutError, RateLimitError
from datasets import load_dataset
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL,
    MODEL_FLASH, MODEL_PRO,
    TEMPERATURE, MAX_TOKENS, TOP_P,
    DATASET_CONFIGS, PROMPT_STRATEGIES,
    DATA_DIR, SEED
)
from prompts import PROMPT_REGISTRY
from utils import extract_multiple_choice_answer, extract_pubmedqa_answer, find_answer_letter, save_results

# ============================================================
# Cost tracking (熔断机制)
# ============================================================
# DeepSeek V4-Flash pricing (元/百万tokens)
COST_PER_M_INPUT = 1.0
COST_PER_M_OUTPUT = 2.0
COST_LIMIT_YUAN = 15.0  # default cost limit (safety)

class CostTracker:
    """跟踪 API 消耗，超限就停"""

    def __init__(self, limit_yuan=COST_LIMIT_YUAN):
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.limit_yuan = limit_yuan
        self.estimate_per_query_input = 500  # conservative estimate
        self.estimate_per_query_output = 300  # conservative estimate
        self.stopped = False

    def estimate_cost(self, n_queries):
        """Estimate cost for n queries before running."""
        tokens_in = n_queries * self.estimate_per_query_input
        tokens_out = n_queries * self.estimate_per_query_output
        cost = (tokens_in / 1_000_000) * COST_PER_M_INPUT + \
               (tokens_out / 1_000_000) * COST_PER_M_OUTPUT
        return cost

    def check_before_run(self, n_queries, dataset_name, strategy_name):
        """Check if estimated cost is within limit before running."""
        est_cost = self.estimate_cost(n_queries)
        if est_cost > self.limit_yuan * 1.5:  # 1.5x buffer
            print(f"\n  ⚠️  Estimated cost for {dataset_name}/{strategy_name}: ¥{est_cost:.2f}")
            print(f"  ⚠️  Exceeds safety threshold (¥{self.limit_yuan}). Skipping.")
            print(f"  💡  Use --cost-limit X to increase limit, or --max-samples N to reduce.")
            return False
        return True

    def add_usage(self, input_tokens, output_tokens):
        """Record actual token usage from API response."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        current_cost = self.current_cost()
        if current_cost > self.limit_yuan:
            print(f"\n  🛑 COST LIMIT REACHED: ¥{current_cost:.2f} > ¥{self.limit_yuan:.2f}")
            print(f"  🛑 Stopping further API calls to prevent excessive spending.")
            self.stopped = True

    def current_cost(self):
        """Calculate current cost in yuan."""
        cost = (self.total_input_tokens / 1_000_000) * COST_PER_M_INPUT + \
               (self.total_output_tokens / 1_000_000) * COST_PER_M_OUTPUT
        return cost

    def summary(self):
        """Print cost summary."""
        print(f"\n{'='*50}")
        print(f"COST SUMMARY")
        print(f"{'='*50}")
        print(f"  Input tokens:  {self.total_input_tokens:,}")
        print(f"  Output tokens: {self.total_output_tokens:,}")
        print(f"  Total cost:    ¥{self.current_cost():.2f}")
        print(f"  Cost limit:    ¥{self.limit_yuan:.2f}")
        if self.stopped:
            print(f"  Status:        🛑 Stopped early (limit reached)")
        else:
            print(f"  Status:        ✅ Completed within limit")


# ============================================================
# API Client
# ============================================================
def init_client():
    """返回 DeepSeek API 客户端"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
        print("=" * 50)
        print("ERROR: DeepSeek API Key not configured!")
        print("=" * 50)
        print("Please set your API key in one of these ways:")
        print("  1. Edit config.py and set DEEPSEEK_API_KEY")
        print("  2. Set environment variable: set DEEPSEEK_API_KEY=sk-...")
        print("  3. Pass via command line: --api-key sk-...")
        sys.exit(1)

    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )
    return client


def query_model(client, prompt, cost_tracker, model=MODEL_FLASH, max_retries=5):
    """发一次 API 请求，失败自动退避重试"""
    last_error = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                timeout=120,  # 2-minute timeout per request
            )

            # Track token usage from API response
            if hasattr(response, 'usage') and response.usage:
                input_tokens = response.usage.prompt_tokens or 0
                output_tokens = response.usage.completion_tokens or 0
            else:
                input_tokens = len(prompt) // 2
                output_tokens = len(response.choices[0].message.content or "") // 2

            cost_tracker.add_usage(input_tokens, output_tokens)
            return response.choices[0].message.content

        except APITimeoutError:
            backoff = (2 ** attempt) * 10  # 10s, 20s, 40s, 80s, 160s
            print(f"\n  ⏱️  Timeout (attempt {attempt+1}/{max_retries}), waiting {backoff}s...")
            time.sleep(backoff)
            last_error = "timeout"

        except RateLimitError:
            backoff = 30
            print(f"\n  🐢  Rate limited, waiting {backoff}s...")
            time.sleep(backoff)
            last_error = "rate_limit"

        except APIStatusError as e:
            if e.status_code == 529:  # Service overloaded
                print(f"\n  🔄  Service overloaded, waiting 30s...")
                time.sleep(30)
            else:
                print(f"\n  ⚠️  API status error {e.status_code}: {e}")
                time.sleep(10)
            last_error = f"status_{e.status_code}"

        except Exception as e:
            print(f"\n  ⚠️  Unexpected error: {e}")
            time.sleep(15)
            last_error = str(e)

    print(f"\n  ❌  Failed after {max_retries} retries (last: {last_error})")
    return None


# ============================================================
# Dataset Loading
# ============================================================
def load_dataset_by_config(config):
    """从 HuggingFace 按配置加载数据集"""
    hf_path = config["hf_path"]
    split = config.get("split", "test")
    max_samples = config.get("max_samples", None)
    config_name = config.get("config", None)

    print(f"  Loading {hf_path} (split={split}, config={config_name})...")
    try:
        if config_name:
            dataset = load_dataset(hf_path, config_name, split=split)
        else:
            dataset = load_dataset(hf_path, split=split)
    except Exception as e:
        print(f"  ❌ Failed to load dataset: {e}")
        return None

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    print(f"  ✅ Loaded {len(dataset)} samples")
    return dataset


# ============================================================
# Dataset Formatting
# ============================================================
def format_medqa(item):
    """MedQA 数据格式化：答案文本 → 选项字母"""
    question = item["question"]
    options = {}
    opts = item["options"]
    if isinstance(opts, dict):
        for k in ["A", "B", "C", "D"]:
            options[k] = opts.get(k, "")
    elif isinstance(opts, list):
        labels = ["A", "B", "C", "D"]
        for i, v in enumerate(opts[:4]):
            options[labels[i]] = v
    else:
        options = {"A": "", "B": "", "C": "", "D": ""}

    # Convert gold answer text to letter
    gold_text = item.get("answer", "")
    gold_letter = find_answer_letter(options, gold_text)

    return question, options, gold_letter


def format_pubmedqa(item):
    """PubMedQA 格式化，带摘要上下文"""
    question = item["question"]
    context = item.get("context", "")
    contexts = item.get("contexts", {})
    # Add PubMed abstract as context if available
    if context:
        question = f"PubMed Abstract: {context}\n\nQuestion: {question}"
    elif contexts:
        # contexts can be a dict with "contexts" key
        ctx_text = " ".join(contexts.values()) if isinstance(contexts, dict) else str(contexts)
        question = f"PubMed Abstract: {ctx_text}\n\nQuestion: {question}"

    options = {"yes": "yes", "no": "no", "maybe": "maybe"}
    answer = item.get("final_decision", item.get("answer", ""))
    return question, options, answer


def format_medmcqa(item):
    """MedMCQA 格式化，答案索引 → 选项字母"""
    question = item["question"]
    options = {}
    labels = ["A", "B", "C", "D"]
    # MedMCQA uses keys 'opa', 'opb', 'opc', 'opd' for options
    key_map = {"opa": "A", "opb": "B", "opc": "C", "opd": "D"}
    for op_key, label in key_map.items():
        if op_key in item:
            options[label] = item[op_key]
    # Fill from lowercase a, b, c, d if opa/opb/etc not found
    if not options:
        for i, lbl in enumerate(labels):
            lower_key = lbl.lower()
            if lower_key in item:
                options[lbl] = item[lower_key]

    # MedMCQA stores answer as index (0, 1, 2, 3)
    correct_idx = item.get("correct", item.get("cop", -1))
    try:
        correct_idx = int(correct_idx)
        idx_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
        answer = idx_to_label.get(correct_idx, str(correct_idx))
    except (ValueError, TypeError):
        answer = str(correct_idx)

    return question, options, answer


# ============================================================
# Evaluation Runner
# ============================================================
def run_evaluation(client, dataset, dataset_name, strategy_name,
                   prompt_fn, cost_tracker, model=MODEL_FLASH):
    """跑一组 (数据集 + 策略) 的完整评测"""
    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name} | Strategy: {strategy_name} | Model: {model}")
    print(f"{'='*60}")

    if dataset is None or len(dataset) == 0:
        print("  ❌ No data to evaluate. Skipping.")
        return []

    # Select format and extract functions
    format_fn = {
        "medqa": format_medqa,
        "pubmedqa": format_pubmedqa,
        "medmcqa": format_medmcqa,
    }.get(dataset_name, format_medqa)

    extract_fn = {
        "medqa": extract_multiple_choice_answer,
        "pubmedqa": extract_pubmedqa_answer,
        "medmcqa": extract_multiple_choice_answer,
    }.get(dataset_name, extract_multiple_choice_answer)

    # Cost check before running
    if not cost_tracker.check_before_run(len(dataset), dataset_name, strategy_name):
        return []

    results = []
    start_time = time.time()

    for idx, item in enumerate(tqdm(dataset, desc=f"{dataset_name}-{strategy_name}")):
        # Check cost limit periodically
        if cost_tracker.stopped:
            print(f"\n  🛑 Cost limit reached. Stopping evaluation.")
            break

        try:
            question, options, gold_answer = format_fn(item)
            prompt = prompt_fn(question, options)
            response = query_model(client, prompt, cost_tracker, model)

            if response:
                prediction = extract_fn(response)
                correct = (str(prediction).strip().lower() == str(gold_answer).strip().lower()
                          if prediction else False)
                results.append({
                    "id": idx,
                    "dataset": dataset_name,
                    "strategy": strategy_name,
                    "model": model,
                    "question": question,
                    "options": options,
                    "gold_answer": str(gold_answer),
                    "prediction": str(prediction) if prediction else None,
                    "raw_response": response,
                    "correct": correct,
                })
            else:
                results.append({
                    "id": idx,
                    "dataset": dataset_name,
                    "strategy": strategy_name,
                    "model": model,
                    "question": question,
                    "options": options,
                    "gold_answer": str(gold_answer),
                    "prediction": None,
                    "raw_response": None,
                    "correct": False,
                    "error": "API call failed"
                })

            # Small delay every 50 queries
            if (idx + 1) % 50 == 0:
                time.sleep(0.5)

        except Exception as e:
            print(f"\n❌ Error at index {idx}: {e}")
            continue

    elapsed = time.time() - start_time
    n_correct = sum(1 for r in results if r.get("correct"))
    print(f"\n  Results: {n_correct}/{len(results)} correct "
          f"({n_correct/len(results)*100:.1f}%) in {elapsed:.0f}s")

    return results


def is_completed(dataset_name, strategy_name, model, data_dir=DATA_DIR):
    """查一下这组实验是不是已经跑过了"""
    pattern = f"{dataset_name}_{strategy_name}_{model}_*.jsonl"
    full_pattern = os.path.join(data_dir, pattern)
    matching = glob.glob(full_pattern)

    # Check if there's a completed run (ignore test runs with max-samples=10)
    for f in matching:
        try:
            results = load_results(f)
            # A run is "complete" if it has >50 results (not just a test run)
            if len(results) > 50:
                return True, len(results)
        except Exception:
            continue
    return False, 0


class IncrementalSaver:
    """分批存结果，崩了也不全丢"""

    def __init__(self, filepath, save_interval=100):
        self.filepath = filepath
        self.save_interval = save_interval
        self.buffer = []
        self.total_saved = 0

    def add(self, result):
        self.buffer.append(result)
        if len(self.buffer) >= self.save_interval:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        import jsonlines
        mode = 'a' if os.path.exists(self.filepath) else 'w'
        with jsonlines.open(self.filepath, mode=mode) as writer:
            for r in self.buffer:
                writer.write(r)
        self.total_saved += len(self.buffer)
        self.buffer = []

    def finish(self):
        self.flush()
        return self.total_saved


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4 医学 QA 评测")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIGS.keys()),
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Datasets to evaluate")
    parser.add_argument("--strategies", nargs="+", default=PROMPT_STRATEGIES,
                        choices=PROMPT_STRATEGIES,
                        help="Prompting strategies to evaluate")
    parser.add_argument("--model", default=MODEL_FLASH, choices=[MODEL_FLASH, MODEL_PRO],
                        help="Model version")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples per dataset (for debugging)")
    parser.add_argument("--cost-limit", type=float, default=15.0,
                        help="Maximum API cost in yuan (default: 15.0)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="DeepSeek API key (overrides config.py)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed dataset+strategy combinations")
    parser.add_argument("--save-interval", type=int, default=50,
                        help="Save intermediate results every N questions (default: 50)")
    args = parser.parse_args()

    # Override API key if provided
    if args.api_key:
        global DEEPSEEK_API_KEY
        import config as cfg
        cfg.DEEPSEEK_API_KEY = args.api_key

    # Create data directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Initialize cost tracker with user-specified limit
    cost_tracker = CostTracker(limit_yuan=args.cost_limit)

    # Initialize API client
    client = init_client()

    # Run timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Override max_samples if specified
    if args.max_samples:
        for d in args.datasets:
            if d in DATASET_CONFIGS:
                DATASET_CONFIGS[d]["max_samples"] = args.max_samples

    # Pre-run cost estimation
    print(f"\n{'='*60}")
    print("COST ESTIMATION")
    print(f"{'='*60}")
    total_est = 0
    for dataset_name in args.datasets:
        est_config = DATASET_CONFIGS[dataset_name]
        n_q = est_config.get("max_samples")
        # Resolve None to actual dataset size
        if n_q is None:
            if dataset_name == "medqa":
                n_q = 1273
            elif dataset_name == "pubmedqa":
                n_q = 500
            elif dataset_name == "medmcqa":
                n_q = 4183
            else:
                n_q = 1000

        for strategy_name in args.strategies:
            est = cost_tracker.estimate_cost(n_q)
            total_est += est
            print(f"  {dataset_name:12s} | {strategy_name:12s} | {n_q:5d} questions | ~¥{est:.2f}")

    print(f"  {'─'*50}")
    print(f"  {'Total estimated cost':>30s} | ~¥{total_est:.2f}")
    print(f"  {'Cost limit':>30s} | ¥{args.cost_limit:.2f}")

    if total_est > args.cost_limit * 1.2:
        print(f"\n  ⚠️  Estimated cost exceeds limit!")
        print(f"  💡  Use --max-samples N to reduce, or --cost-limit X to increase")
        proceed = input(f"  Continue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("  Aborted.")
            return

    print(f"\n{'='*60}")
    print("STARTING EXPERIMENTS")
    print(f"{'='*60}")

    all_results = []

    # Iterate over datasets and strategies
    for dataset_name in args.datasets:
        config = DATASET_CONFIGS[dataset_name]
        dataset = load_dataset_by_config(config)
        if dataset is None:
            continue

        for strategy_name in args.strategies:
            if cost_tracker.stopped:
                print(f"\n  🛑 Cost limit reached. Stopping all evaluations.")
                break
            if strategy_name not in PROMPT_REGISTRY:
                print(f"Warning: Unknown strategy '{strategy_name}', skipping")
                continue

            # Resume check: skip if already completed
            if args.resume:
                completed, n_done = is_completed(dataset_name, strategy_name, args.model)
                if completed:
                    print(f"\n  ⏭️  Skipping {dataset_name}/{strategy_name} — already completed ({n_done} results)")
                    continue

            prompt_fn = PROMPT_REGISTRY[strategy_name]
            output_file = os.path.join(
                DATA_DIR, f"{dataset_name}_{strategy_name}_{args.model}_{run_id}.jsonl"
            )

            # Use incremental saver to prevent data loss on crash
            saver = IncrementalSaver(output_file, save_interval=args.save_interval)
            results = run_evaluation(client, dataset, dataset_name,
                                      strategy_name, prompt_fn,
                                      cost_tracker, args.model)

            if results:
                n_saved = saver.finish()
                print(f"  💾 Saved {n_saved} results to: {output_file}")
                all_results.extend(results)

        if cost_tracker.stopped:
            break

        if cost_tracker.stopped:
            break

    # Save combined results
    if all_results:
        combined_file = os.path.join(DATA_DIR, f"all_results_{args.model}_{run_id}.jsonl")
        save_results(all_results, combined_file)
        print(f"\n💾 All results saved to: {combined_file}")

    # Print cost summary
    cost_tracker.summary()

    # Print accuracy summary
    if all_results:
        print(f"\n{'='*60}")
        print("ACCURACY SUMMARY")
        print(f"{'='*60}")
        summary = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in all_results:
            key = (r["dataset"], r["strategy"])
            summary[key]["total"] += 1
            if r.get("correct"):
                summary[key]["correct"] += 1

        for (ds, strat), counts in sorted(summary.items()):
            acc = counts["correct"] / counts["total"] * 100 if counts["total"] > 0 else 0
            bar_len = int(acc / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  {ds:12s} | {strat:12s} | {acc:5.1f}% |{bar}| {counts['correct']}/{counts['total']}")

    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
