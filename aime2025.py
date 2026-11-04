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
from math_verify import parse, verify

from utils import with_request_context

DATASET_PATH = "math-ai/aime25"

def verify_answer(prediction: str, target: str) -> tuple[int | None, bool]:
    extracted_answer = None
    try:
        parsed_pred = parse(prediction)
        extracted_answer = parsed_pred[0] if parsed_pred else prediction

        if not target.startswith('\\boxed{'):
            target = f'\\boxed{{{target}}}'
        parsed_gold = parse(f"\\boxed{{{target}}}")
        gold_value = parsed_gold[0] if parsed_gold else target

        is_correct = verify(gold_value, extracted_answer, strict=False)
        return extracted_answer, is_correct
    except Exception as e:
        return extracted_answer, False


@solver
def aime2025_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if state.user_prompt:
            state.user_prompt.text = f"{state.user_prompt.text}\n\n Please reason step by step, and put your final answer within \\boxed{{}}."
        return await generate(state)

    return solve


@scorer(metrics=[accuracy(), stderr()])
def aime2025_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        target_val = target.text.strip()
        extracted_answer, correct = verify_answer(state.output.completion, target_val)
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=str(extracted_answer),
            explanation=f"Extracted={extracted_answer}, Target={target_val}",
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
        solver=with_request_context(aime2025_solver()),
        scorer=aime2025_scorer(),
    )
