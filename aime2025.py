import re
from typing import Optional

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

DATASET_PATH = "math-ai/aime25"

USER_PROMPT_TEMPLATE = """
Please solve the following math problems step by step.

{prompt}

The final answer is a non-negative integer. Write your final answer on a separate last line in the format "Answer: N" where N is a plain integer (e.g. "Answer: 42"). Do NOT use LaTeX, \\boxed, or any formatting - just the plain number.
""".strip()

ANSWER_PATTERN = re.compile(
    r"\*{0,2}Answer[:：]?\*{0,2}[:：\s]+(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def _extract_boxed_content(text: str) -> list[str]:
    answers = []
    for piece in text.split("boxed{")[1:]:
        depth = 0
        for i, char in enumerate(piece):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    content = piece[:i] if piece[i : i + 2] != "}%" else piece[: i + 1]
                    if content.strip():
                        answers.append(content.strip())
                    break
    return answers


def _extract_number_from_text(text: str) -> Optional[str]:
    if not text:
        return None

    text = text.strip()
    if text.isdigit():
        return text.lstrip("0") or "0"

    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", "", cleaned)
    cleaned = re.sub(r"[^\d]", "", cleaned)

    if cleaned:
        return cleaned.lstrip("0") or "0"
    return None


def _extract_from_equals(expr: str) -> Optional[str]:
    for part in reversed(expr.split("=")[1:]):
        num = _extract_number_from_text(part)
        if num:
            return num
    return None


def extract_answer(completion: str) -> Optional[str]:
    """Extract answer from model completion. Priority: Answer: pattern > \\boxed{} > last line number"""
    text = completion.replace("mathbf{", "boxed{")

    matches = ANSWER_PATTERN.findall(text)
    if matches:
        answer_text = matches[-1].strip()
        result = _extract_number_from_text(answer_text)
        if result:
            return result

    boxed = _extract_boxed_content(text)
    if boxed:
        content = boxed[-1]
        if content.isdigit():
            return content
        if "=" in content:
            result = _extract_from_equals(content)
            if result:
                return result
        result = _extract_number_from_text(content)
        if result:
            return result

    last_line = text.strip().split("\n")[-1].strip()
    cleaned = re.sub(r"^[\$\\\[\]\(\)]+|[\$\\\[\]\(\)]+$", "", last_line).strip()
    if cleaned.isdigit():
        return cleaned

    return None


def verify_answer(prediction: str, target: str) -> bool:
    try:
        pred_int = int(prediction)
        target_int = int(target)
        # AIME answers are integers in range [0, 999]
        if not (0 <= pred_int <= 999):
            return False
        return pred_int == target_int
    except ValueError:
        pass

    try:
        from math_verify import parse, verify

        return verify(
            parse(f"\\boxed{{{prediction}}}"),
            parse(f"\\boxed{{{target}}}"),
        )
    except Exception:
        return False


@solver
def aime2025_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = USER_PROMPT_TEMPLATE.format(
                prompt=state.user_prompt.text
            )
        return await generate(state)

    return solve


@scorer(metrics=[accuracy(), stderr()])
def aime2025_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        extracted = extract_answer(state.output.completion)
        target_val = target.text.strip()

        if extracted is None:
            return Score(
                value=INCORRECT,
                answer=None,
                explanation="Failed to extract answer",
            )

        correct = verify_answer(extracted, target_val)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=extracted,
            explanation=f"Extracted={extracted}, Target={target_val}",
        )

    return score


@task
def aime2025() -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="test",
            sample_fields=lambda r: Sample(
                id=r["id"],
                input=r["problem"],
                target=str(r["answer"]),
            ),
        ),
        solver=aime2025_solver(),
        scorer=aime2025_scorer(),
    )
