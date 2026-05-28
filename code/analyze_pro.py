"""Analyze V4-Pro results."""
from utils import load_results
from collections import Counter

r = load_results("../data/all_results_deepseek-v4-pro_20260525_010733.jsonl")

groups = {}
for x in r:
    k = x['strategy']
    if k not in groups: groups[k] = {'total': 0, 'correct': 0}
    groups[k]['total'] += 1
    if x.get('correct'): groups[k]['correct'] += 1

errors = Counter()
for x in r:
    if not x.get('correct'):
        resp = str(x.get('raw_response', '')).lower()
        q = str(x.get('question', '')).lower()
        resp_words = set(resp.split())
        q_words = set(q.split())
        shared = resp_words & q_words
        resp_len = len(resp.split())

        # E5: True format errors
        if not resp or resp_len < 2:
            errors['E5'] += 1
        elif any(w in resp for w in ['i am sorry', 'i cannot', 'i apologize', 'as an ai']):
            errors['E5'] += 1
        # E4: Hallucination
        elif any(w in resp for w in ['hypothetical', 'fictional', 'imaginary']):
            errors['E4'] += 1
        # E3: Comprehension
        elif len(shared) < 3 and resp_len >= 5:
            errors['E3'] += 1
        # E2: Reasoning Error
        elif resp_len >= 30 and len(shared) >= 4:
            errors['E2'] += 1
        # E1: Knowledge Gap
        else:
            errors['E1'] += 1

print('=== Pro Results ===')
for k, v in sorted(groups.items()):
    print(f'  {k}: {v["correct"]}/{v["total"]} = {v["correct"]/v["total"]*100:.1f}%')

total_err = sum(errors.values())
print(f'\nPro error distribution (total={total_err}):')
for k in ['E1', 'E2', 'E3', 'E4', 'E5']:
    pct = errors[k] / total_err * 100 if total_err else 0
    print(f'  {k}: {errors[k]} ({pct:.1f}%)')
