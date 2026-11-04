import ast
import base64
import random
import re
import string
from io import BytesIO
from typing import Optional

import numpy as np
from datasets import load_dataset
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
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
from inspect_ai.solver import TaskState, generate

from utils import with_request_context

MMMU_PRO_10c_DATASET = "MMMU/MMMU_Pro"
MMMU_PRO_10c_SUBSET = "standard (10 options)"

MMMU_PRO_10c_PROMPT = (
    "Answer the following multiple-choice question. The last line of your response should be of "
    "the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of the options. "
    "Think step by step before answering."
)


def _image_to_base64(img) -> Optional[str]:
    if hasattr(img, "convert"):
        buffered = BytesIO()
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()
    elif isinstance(img, bytes):
        return base64.b64encode(img).decode()
    elif isinstance(img, str):
        return img
    return None


def _parse_image(row: dict) -> list[str]:
    imgs = []
    for i in range(1, 8):
        img = row[f"image_{i}"]
        if img is not None:
            img_base64 = _image_to_base64(img)
            if not img_base64:
                raise ValueError("Failed to convert image to base64")
            imgs.append(img_base64)
    return imgs


def _parse_choices(row: dict) -> tuple[list[str], dict[str, str]]:
    options_str = row["options"]
    options_list = ast.literal_eval(options_str)

    all_choices = []
    index2ans = {}
    for i, opt in enumerate(options_list):
        letter = string.ascii_uppercase[i]
        all_choices.append(letter)
        index2ans[letter] = str(opt)

    return all_choices, index2ans


def _row_to_sample(row: dict, idx: int) -> Sample:
    imgs = _parse_image(row)
    all_choices, index2ans = _parse_choices(row)
    answer = row["answer"].strip().upper()

    options_prompt = "\n".join([f"{letter}. {index2ans[letter]}" for letter in all_choices])

    prompt = f"{row['question']}\n{options_prompt}\n\n"
    prompt += MMMU_PRO_10c_PROMPT

    content = []
    for image_base64 in imgs:
        content.append(ContentImage(image=f"data:image/jpeg;base64,{image_base64}"))
    content.append(ContentText(text=prompt))

    return Sample(
        id=str(row.get("id", idx)),
        input=[ChatMessageUser(content=content)],
        target=answer,
        metadata={
            "all_choices": all_choices,
            "index2ans": index2ans,
            "subject": row.get("subject", ""),
        },
    )


def load_mmmu_pro_dataset(
    dataset_name: str = MMMU_PRO_10c_DATASET,
    subset: str = MMMU_PRO_10c_SUBSET,
    limit: Optional[int] = None,
) -> MemoryDataset:
    ds = load_dataset(dataset_name, subset, split="test")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    samples = [_row_to_sample(row, idx) for idx, row in enumerate(ds)]
    return MemoryDataset(samples=samples, name="MMMU_Pro_10c")


def parse_multi_choice_response(
    response: str, all_choices: list[str], index2ans: dict[str, str]
    ) -> str:
    """Parse the prediction from the generated response. Return the predicted index (A, B, C, D, etc.)."""
    if not all_choices:
        raise ValueError("all_choices is empty — dataset error")

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []

    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if not candidates:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)

    if not candidates:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)

    if not candidates and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False

    if not candidates:
        pred_index = random.choice(all_choices)
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    start_indexes.append(response.rfind(f"({can})"))
            else:
                for can in candidates:
                    start_indexes.append(response.rfind(f" {can} "))
        else:
            for can in candidates:
                start_indexes.append(response.lower().rfind(index2ans[can].lower()))
        pred_index = candidates[int(np.argmax(start_indexes))]
    else:
        pred_index = candidates[0]

    return pred_index


@scorer(metrics=[accuracy(), stderr()])
def mmmu_pro_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        target_answer = target.text.strip().upper()

        metadata = state.metadata or {}
        all_choices = metadata.get("all_choices", [])
        index2ans = metadata.get("index2ans", {})

        predicted = parse_multi_choice_response(completion, all_choices, index2ans)

        correct = predicted == target_answer
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=predicted,
            explanation=f"Predicted={predicted}, Target={target_answer}",
            metadata={
                "predicted": predicted,
                "target": target_answer,
                "raw_completion_tail": completion[-500:] if not correct else None,
            },
        )

    return score


@task
def mmmu_pro_10c(
    dataset_name: str = MMMU_PRO_10c_DATASET,
    subset: str = MMMU_PRO_10c_SUBSET,
    limit: Optional[int] = None,
) -> Task:
    return Task(
        dataset=load_mmmu_pro_dataset(dataset_name, subset, limit),
        solver=[with_request_context(generate())],
        scorer=mmmu_pro_scorer(),
    )
