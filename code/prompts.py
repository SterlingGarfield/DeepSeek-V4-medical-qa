"""四种提示策略的 prompt 模板"""


def format_options(options):
    """把选项转成 A/B/C/D 格式字符串"""
    if isinstance(options, dict):
        lines = []
        for key in ["A", "B", "C", "D"]:
            if key in options:
                lines.append(f"{key}. {options[key]}")
        return "\n".join(lines)
    elif isinstance(options, list):
        labels = ["A", "B", "C", "D"]
        return "\n".join(f"{lbl}. {opt}" for lbl, opt in zip(labels, options))
    return str(options)


def zero_shot_prompt(question, options):
    """P1: 零样本，直接选答"""
    options_str = format_options(options)
    prompt = f"""You are a medical expert answering a clinical question. Choose the single best answer from the options provided.

Question: {question}
Options:
{options_str}

Answer:"""
    return prompt


def cot_prompt(question, options):
    """P2: 思维链，先推再答"""
    options_str = format_options(options)
    prompt = f"""You are a medical expert answering a clinical question. First, reason step by step, then give your final answer.

Question: {question}
Options:
{options_str}

Reasoning:"""
    return prompt


def few_shot_prompt(question, options):
    """P3: 3 个示例后作答"""
    examples = """
Example 1:
Question: A 65-year-old man with hypertension and diabetes presents with sudden-onset severe headache and confusion. Blood pressure is 220/120 mmHg. What is the most appropriate immediate management?
A. Oral nifedipine
B. Intravenous labetalol
C. Sublingual captopril
D. Intravenous sodium nitroprusside
Reasoning: This is a hypertensive emergency with encephalopathy. Oral medications act too slowly. Intravenous labetalol is preferred as it can be carefully titrated. Sodium nitroprusside can cause cyanide toxicity.
Answer: B

Example 2:
Question: A 35-year-old woman presents with fatigue, weight gain, and cold intolerance. Labs show elevated TSH and low free T4. What is the most likely diagnosis?
A. Graves' disease
B. Hashimoto's thyroiditis
C. Subacute thyroiditis
D. Pituitary adenoma
Reasoning: Elevated TSH with low free T4 indicates primary hypothyroidism. Hashimoto's thyroiditis is the most common cause of primary hypothyroidism in this age group. Graves' disease causes hyperthyroidism (low TSH, high T4). Subacute thyroiditis typically presents with painful thyroid.
Answer: B

Example 3:
Question: A patient with atrial fibrillation is started on warfarin. Which lab value is most important to monitor?
A. Platelet count
B. INR
C. aPTT
D. Bleeding time
Reasoning: Warfarin inhibits vitamin K-dependent clotting factors (II, VII, IX, X). The INR is specifically designed to monitor warfarin therapy. aPTT monitors heparin. Platelet count and bleeding time assess platelet function.
Answer: B

"""
    options_str = format_options(options)
    prompt = f"""You are a medical expert answering clinical questions. Here are three examples:

{examples}
Now answer the following question:
Question: {question}
Options:
{options_str}

Answer:"""
    return prompt


def structured_prompt(question, options):
    """P4: SOAP 结构化推理"""
    options_str = format_options(options)
    prompt = f"""You are a medical expert answering a clinical question.
Follow this structured reasoning process:

[Analysis] Identify key clinical findings and their implications.
[Evidence] List relevant medical knowledge that applies.
[Differential] Consider possible diagnoses and rule them in/out.
[Conclusion] Select the best answer based on your analysis.

Question: {question}
Options:
{options_str}

[Analysis]"""
    return prompt


PROMPT_REGISTRY = {
    "zero_shot": zero_shot_prompt,
    "cot": cot_prompt,
    "few_shot": few_shot_prompt,
    "structured": structured_prompt,
}
