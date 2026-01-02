import ast
import base64
import json
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

MMMU_PRO_V_DATASET = "moonshotai/mmmu-pro-vision"

MMMU_PRO_V_PROMPT = (
    "Write out the multiple-choice question in the image and then solve it. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options. "
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


def _parse_images(row: dict) -> list[str]:
    images = []

    if "image" in row and row["image"] is not None:
        image_data = row["image"]
        if isinstance(image_data, list):
            raw_images = image_data
        elif isinstance(image_data, str):
            try:
                raw_images = ast.literal_eval(image_data)
            except Exception:
                try:
                    raw_images = json.loads(image_data)
                except Exception:
                    raw_images = [image_data]
        else:
            raw_images = [image_data]

        if not isinstance(raw_images, list):
            raw_images = [raw_images]

        for img in raw_images:
            img_base64 = _image_to_base64(img)
            if img_base64:
                images.append(img_base64)

    if not images:
        for i in range(1, 8):
            img_key = f"image_{i}"
            if img_key in row and row[img_key] is not None:
                img_base64 = _image_to_base64(row[img_key])
                if img_base64:
                    images.append(img_base64)

    return images


def _parse_choices(row: dict) -> tuple[list[str], dict[str, str]]:
    all_choices = []
    index2ans = {}

    options_str = row.get("options", "")
    if options_str:
        try:
            options_list = ast.literal_eval(options_str)
            for i, opt in enumerate(options_list):
                letter = string.ascii_uppercase[i]
                all_choices.append(letter)
                index2ans[letter] = str(opt)
        except Exception:
            pass

    if not all_choices:
        for key in string.ascii_uppercase:
            if key in row and row[key] is not None:
                all_choices.append(key)
                index2ans[key] = str(row[key])

    return all_choices, index2ans


def _row_to_sample(row: dict, idx: int) -> Optional[Sample]:
    try:
        images = _parse_images(row)
        all_choices, index2ans = _parse_choices(row)
        answer = str(row.get("answer", "")).strip().upper()

        content = [ContentImage(image=f"data:image/jpeg;base64,{img}") for img in images]
        content.append(ContentText(text=MMMU_PRO_V_PROMPT))

        return Sample(
            id=str(row.get("id", row.get("index", idx))),
            input=[ChatMessageUser(content=content)],
            target=answer,
            metadata={
                "all_choices": all_choices,
                "index2ans": index2ans,
                "category": row.get("category", ""),
                "subject": row.get("subject", ""),
            },
        )
    except Exception as e:
        print(f"[Row parse error] {e}")
        return None


def load_mmmu_pro_dataset(
    dataset_name: str = MMMU_PRO_V_DATASET,
    limit: Optional[int] = None,
) -> MemoryDataset:
    ds = load_dataset(dataset_name, split="test")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    samples = []
    for idx, row in enumerate(ds):
        sample = _row_to_sample(row, idx)
        if sample is not None:
            samples.append(sample)

    return MemoryDataset(samples=samples, name="MMMU_Pro_Vision")


def parse_multi_choice_response(
    response: str, all_choices: list[str], index2ans: dict[str, str]
) -> str:
    """Parse the prediction from the generated response. Return the predicted index (A, B, C, D, etc.)."""
    if not all_choices:
        raise ValueError("all_choices is empty — dataset error")

    choices_pattern = "|".join(re.escape(c) for c in all_choices)

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        match = re.match(
            rf"^[\s\*\$\(\[\:]*({choices_pattern})[\s\*\$\)\]\.\,\:]*",
            answer_str,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).upper()

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
def mmmu_pro_v(
    dataset_name: str = MMMU_PRO_V_DATASET,
    limit: Optional[int] = None,
) -> Task:
    return Task(
        dataset=load_mmmu_pro_dataset(dataset_name, limit),
        solver=[generate()],
        scorer=mmmu_pro_scorer(),
    )
