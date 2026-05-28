"""工具函数：答案提取、文件读写"""

import re
import json


def extract_multiple_choice_answer(text):
    """从模型输出中抓取 A/B/C/D 选项"""
    if not text:
        return None

    text = text.strip()

    # \boxed{A} 格式
    match = re.search(r'\\boxed\s*\{?\s*([A-Da-d])\s*\}?', text)
    if match:
        return match.group(1).upper()

    # \textbf{A} 格式
    match = re.search(r'\\textbf\s*\{?\s*([A-Da-d])\s*\}?', text)
    if match:
        return match.group(1).upper()

    # "Answer: A"
    match = re.search(r'(?i)(?:answer|conclusion|所以|答案是?)\s*[:\-]?\s*([A-Da-d])(?:\.|\)|\s|$)', text)
    if match:
        return match.group(1).upper()

    # "A. 某答案" 在尾部
    match = re.search(r'(?<!\w)([A-Da-d])(?:\.|\))\s+(?:is|text)', text)
    if match:
        return match.group(1).upper()

    # 末尾独立字母
    match = re.search(r'(?<!\w)([A-Da-d])(?!\w)\s*$', text)
    if match:
        return match.group(1).upper()

    # 某一行就是单个字母
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if re.match(r'^[A-Da-d]$', line):
            return line.upper()

    # "A)" 或 "A." 开头的行
    for line in reversed(lines):
        line = line.strip()
        if line and re.match(r'^[A-Da-d][\)\.]', line):
            return line[0].upper()

    return None


def extract_pubmedqa_answer(text):
    """从模型输出中抓取 yes/no/maybe"""
    if not text:
        return None

    text = text.strip().lower()

    # \boxed{yes}
    match = re.search(r'\\boxed\s*\{?\s*(yes|no|maybe)\s*\}?', text)
    if match:
        return match.group(1).lower()

    # "Answer: yes"
    match = re.search(r'(?i)(?:answer|conclusion|所以|结论)\s*[:\-]?\s*(yes|no|maybe)', text)
    if match:
        return match.group(1).lower()

    first_word = text.split()[0] if text.split() else ""
    if first_word in ["yes", "no", "maybe"]:
        return first_word

    if re.search(r'\byes\b', text):
        return "yes"
    if re.search(r'\bno\b', text):
        return "no"
    if re.search(r'\bmaybe\b', text):
        return "maybe"

    return None


def find_answer_letter(options, answer_text):
    """根据答案文本匹配选项字母"""

    if isinstance(options, list):
        labels = ["A", "B", "C", "D"]
        options = {labels[i]: str(opt) for i, opt in enumerate(options) if i < len(labels)}

    if not isinstance(options, dict):
        return str(answer_text)

    answer_text = str(answer_text).strip().lower()

    # 精确匹配
    for letter, text in options.items():
        if str(text).strip().lower() == answer_text:
            return letter

    # 模糊匹配：答案含在选项内，或选项含在答案内
    for letter, text in options.items():
        opt_text = str(text).strip().lower()
        if answer_text in opt_text or opt_text in answer_text:
            return letter

    if answer_text.upper() in ["A", "B", "C", "D"]:
        return answer_text.upper()

    # 索引匹配
    try:
        idx = int(answer_text)
        return ["A", "B", "C", "D"][idx]
    except (ValueError, IndexError):
        pass

    return str(answer_text)


def save_results(results, filepath):
    """存结果到 JSONL"""
    import jsonlines
    with jsonlines.open(filepath, mode='w') as writer:
        for r in results:
            writer.write(r)


def load_results(filepath):
    """读 JSONL 结果文件"""
    import jsonlines
    results = []
    with jsonlines.open(filepath) as reader:
        for item in reader:
            results.append(item)
    return results
