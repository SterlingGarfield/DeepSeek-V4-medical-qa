from utils import load_results

r = load_results("../data/all_results_deepseek-v4-pro_20260525_010733.jsonl")

for strat in ["zero_shot", "cot", "few_shot", "structured"]:
    subset = [x for x in r if x["strategy"] == strat]
    total = len(subset)
    correct = sum(1 for x in subset if x.get("correct"))
    none_pred = [x for x in subset if x.get("prediction") == "None"]
    wrong_format = [x for x in subset if x.get("prediction") and not x.get("correct")]

    print(f"\n{'='*50}")
    print(f"{strat}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"  None predictions: {len(none_pred)}/{total}")
    print(f"  Wrong answers:    {len(wrong_format)}/{total}")

    if none_pred:
        print(f"\n  --- None prediction examples ---")
        for c in none_pred[:2]:
            raw = str(c.get("raw_response", ""))[:250]
            print(f"  Gold: {c.get('gold_answer', '?')}")
            print(f"  Raw:  {raw}")
            print()

    if wrong_format:
        print(f"  --- Wrong answer examples ---")
        for c in wrong_format[:2]:
            raw = str(c.get("raw_response", ""))[:250]
            print(f"  Gold: {c.get('gold_answer', '?')}")
            print(f"  Pred: {c.get('prediction', '?')}")
            print(f"  Raw:  {raw}")
            print()
