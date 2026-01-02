import base64
import io
from typing import Any, Optional

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
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

DATASET_PATH = "echo840/OCRBench"


def pil_to_data_url(image) -> str:
    """Convert PIL Image to data URL for inspect-ai multimodal input."""
    from PIL import Image
    
    if isinstance(image, dict):
        if "bytes" in image:
            image = Image.open(io.BytesIO(image["bytes"]))
        elif "path" in image:
            image = Image.open(image["path"])
        else:
            raise ValueError(f"Unknown image dict format: {image.keys()}")
    
    buffered = io.BytesIO()
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"


def match_answer(prediction: str, answers: list[str], category: str) -> bool:
    """
    Check if prediction contains any ground truth answer.
    - Math expressions: whitespace-insensitive exact matching
    - Other categories: case-insensitive substring matching
    """
    if category == "Handwritten Mathematical Expression Recognition":
        pred_norm = prediction.strip().replace("\n", "").replace(" ", "")
        for ans in answers:
            if ans.strip().replace("\n", "").replace(" ", "") in pred_norm:
                return True
    else:
        pred_norm = prediction.lower().strip().replace("\n", " ")
        for ans in answers:
            if ans.lower().strip().replace("\n", " ") in pred_norm:
                return True
    return False


@scorer(metrics=[accuracy(), stderr()])
def ocrbench_scorer() -> Scorer:
    """Score by checking if prediction contains ground truth answer."""
    
    async def score(state: TaskState, target: Target) -> Score:
        completion = state.output.completion
        metadata = state.metadata or {}
        category = metadata.get("category", "")
        answers = metadata.get("answers", [])
        
        if not answers and target.text:
            try:
                import ast
                parsed = ast.literal_eval(target.text)
                answers = parsed if isinstance(parsed, list) else [target.text]
            except (ValueError, SyntaxError):
                answers = [target.text]
        
        matched = match_answer(completion, answers, category)
        
        return Score(
            value=CORRECT if matched else INCORRECT,
            answer=completion[:500] if completion else None,
            explanation=f"Matched={matched}, Answers={answers}",
        )
    
    return score


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert HuggingFace record to inspect-ai Sample."""
    return Sample(
        input=[
            ChatMessageUser(content=[
                ContentImage(image=pil_to_data_url(record["image"])),
                ContentText(text=record["question"]),
            ])
        ],
        target=str(record["answer"]),
        metadata={
            "category": record["question_type"],
            "answers": record["answer"],
        },
    )


@task
def ocrbench(limit: Optional[int] = None) -> Task:
    """OCRBench evaluation task."""
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="test",
            sample_fields=record_to_sample,
            limit=limit,
        ),
        solver=[generate()],
        scorer=ocrbench_scorer(),
    )
