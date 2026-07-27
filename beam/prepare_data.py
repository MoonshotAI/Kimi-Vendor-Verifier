#!/usr/bin/env python3
"""Prepare BEAM 1M data files from an official BEAM chat directory.

Converts the official BEAM repo layout

    <src>/<chat_id>/chat.json
    <src>/<chat_id>/probing_questions/probing_questions.json

into the two compact files consumed by beam_generate.py / beam_judge.py:

    data/beam_1m_chats.jsonl.gz     one line per chat:  {"chat_id", "messages"}
    data/beam_1m_questions.jsonl    one line per probing question

Usage:
    python prepare_data.py --src /path/to/BEAM/chats/1M --out-dir data

The official BEAM dataset can be obtained from
https://github.com/mohammadtavakoli78/BEAM (chats/1M) or
https://huggingface.co/datasets/Mohammadta/BEAM
"""

import argparse
import gzip
import json
import os

QUESTION_TYPES = [
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
]

# question_type -> field holding the gold answer (reference only; the judge
# scores against rubric items, not the gold answer)
_GOLD_ANSWER_FIELD = {
    "abstention": "ideal_response",
    "contradiction_resolution": "ideal_answer",
    "event_ordering": "answer",
    "information_extraction": "answer",
    "instruction_following": None,
    "knowledge_update": "answer",
    "multi_session_reasoning": "answer",
    "preference_following": None,
    "summarization": "ideal_summary",
    "temporal_reasoning": "answer",
}


def load_chat_messages(chat_json_path: str) -> list:
    """Flatten an official BEAM chat.json into a standard messages array.

    chat.json structure: list of batches, each batch has 'turns' (list of
    turn-groups, each turn-group is a list of messages with role/content).
    """
    with open(chat_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    messages = []
    for batch in data:
        for turn_group in batch.get("turns", []):
            for msg in turn_group:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
    return messages


def get_gold_answer(question_type: str, question_obj: dict) -> str:
    field = _GOLD_ANSWER_FIELD.get(question_type)
    if field and field in question_obj:
        return str(question_obj[field])
    rubric = question_obj.get("rubric", [])
    if isinstance(rubric, list):
        return "\n".join(rubric)
    return str(rubric)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True,
                        help="Official BEAM chat size directory, e.g. chats/1M")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "data"),
                        help="Output directory (default: ./data next to this script)")
    parser.add_argument("--tag", default="1m", help="Output file tag (default: 1m)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    chats_path = os.path.join(args.out_dir, f"beam_{args.tag}_chats.jsonl.gz")
    questions_path = os.path.join(args.out_dir, f"beam_{args.tag}_questions.jsonl")

    chat_ids = sorted(
        [d for d in os.listdir(args.src) if d.isdigit()],
        key=int,
    )

    n_chats = 0
    n_questions = 0
    with gzip.open(chats_path, "wt", encoding="utf-8") as fc, \
            open(questions_path, "w", encoding="utf-8") as fq:
        for chat_id in chat_ids:
            chat_dir = os.path.join(args.src, chat_id)
            chat_json_path = os.path.join(chat_dir, "chat.json")
            pq_path = os.path.join(chat_dir, "probing_questions", "probing_questions.json")
            if not os.path.exists(chat_json_path) or not os.path.exists(pq_path):
                print(f"skip chat {chat_id}: missing chat.json or probing_questions.json")
                continue

            messages = load_chat_messages(chat_json_path)
            fc.write(json.dumps({"chat_id": chat_id, "messages": messages},
                                ensure_ascii=False) + "\n")
            n_chats += 1

            with open(pq_path, "r", encoding="utf-8") as f:
                pq_data = json.load(f)
            chat_q = 0
            for question_type, questions in pq_data.items():
                for q_idx, q_obj in enumerate(questions):
                    rubric = q_obj.get("rubric", [])
                    if not isinstance(rubric, list):
                        rubric = [str(rubric)]
                    fq.write(json.dumps({
                        "chat_id": chat_id,
                        "question_type": question_type,
                        "question_index": q_idx,
                        "question": q_obj["question"],
                        "gold_answer": get_gold_answer(question_type, q_obj),
                        "rubric": rubric,
                        "difficulty": q_obj.get("difficulty", "unknown"),
                    }, ensure_ascii=False) + "\n")
                    n_questions += 1
                    chat_q += 1
            if chat_q != 20:
                print(f"warning: chat {chat_id} has {chat_q} questions (expected 20)")

    print(f"wrote {n_chats} chats -> {chats_path}")
    print(f"wrote {n_questions} questions -> {questions_path}")

    # ------------------------------------------------------------------
    # Validation: rebuild and re-read both files.
    # ------------------------------------------------------------------
    with gzip.open(chats_path, "rt", encoding="utf-8") as f:
        chats_check = [json.loads(line) for line in f]
    with open(questions_path, "r", encoding="utf-8") as f:
        questions_check = [json.loads(line) for line in f]
    assert len(chats_check) == n_chats, "chat count mismatch after rewrite"
    assert len(questions_check) == n_questions, "question count mismatch after rewrite"
    per_chat = {}
    for q in questions_check:
        per_chat[q["chat_id"]] = per_chat.get(q["chat_id"], 0) + 1
    bad = {c: n for c, n in per_chat.items() if n != 20}
    assert not bad, f"chats with != 20 questions: {bad}"
    print(f"validation OK: {n_chats} chats x 20 questions = {n_questions}")


if __name__ == "__main__":
    main()
