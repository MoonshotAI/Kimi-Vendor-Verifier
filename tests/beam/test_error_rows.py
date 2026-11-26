"""Offline tests: failed requests and failed judge calls must not pass for scores.

No endpoint is contacted; the generator and judge run against stub clients.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from types import SimpleNamespace

import pytest

from beam import beam_generate, beam_judge


def _answer(index: int, **fields) -> dict:
    return {
        "chat_id": 1,
        "question_type": "information_extraction",
        "question_index": index,
        "question": "q",
        "gold_answer": "a",
        "rubric": ["mentions a", "mentions b"],
        "response": "a and b",
        "finish_reason": "stop",
        "completion_id": f"cmpl-{index}",
        "error": None,
        **fields,
    }


def _write(path, rows) -> str:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return str(path)


class _Judge:
    """Stub judge client: replies in order; an Exception reply is raised."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **_):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _judge_args(answers: str, output: str, **overrides) -> argparse.Namespace:
    defaults = dict(
        answers=answers, output=output, judge_model="judge", judge_base_url="http://x",
        judge_api_key="k", judge_reasoning_effort="", judge_max_tokens=16,
        judge_temperature=0.3, concurrency=1, max_retries=1, timeout=1, limit=0,
        fail_on_errors=False,
    )
    return argparse.Namespace(**{**defaults, **overrides})


def _run_judge(monkeypatch, args, replies) -> _Judge:
    judge = _Judge(replies)
    monkeypatch.setattr(beam_judge, "AsyncOpenAI", lambda **_: judge)
    asyncio.run(beam_judge.run(args))
    return judge


def test_generate_resume_retries_failed_requests(tmp_path):
    """A row with `error` is not done; a later good row for the same key is."""
    path = _write(tmp_path / "answers.jsonl", [
        _answer(0),
        _answer(1, response="", error="APIConnectionError('Connection error.')"),
        _answer(2, response="", error="APIConnectionError('Connection error.')"),
        _answer(2),
    ])

    assert beam_generate.load_done_keys(path) == {
        "1|information_extraction|0",
        "1|information_extraction|2",
    }


def test_judge_scores_failed_requests_zero_and_warns(tmp_path, monkeypatch, capsys):
    """An endpoint outage must not be turned into a score silently."""
    answers = _write(tmp_path / "answers.jsonl", [
        _answer(0),
        _answer(1, response="", error="APIConnectionError('Connection error.')"),
    ])
    args = _judge_args(answers, str(tmp_path / "scores.jsonl"))

    judge = _run_judge(monkeypatch, args, ['{"score": 1.0}', '{"score": 1.0}'])

    assert judge.calls == 2  # only the real answer reaches the judge
    summary = capsys.readouterr().out
    assert "overall: 0.5000" in summary
    assert "WARNING: 1 of 2 answers are failed requests" in summary
    assert "1 failed requests (scored 0), 0 empty responses" in summary


def test_judge_fail_on_errors_stops_before_judging(tmp_path, monkeypatch):
    answers = _write(tmp_path / "answers.jsonl", [
        _answer(0),
        _answer(1, response="", error="APIConnectionError('Connection error.')"),
    ])
    output = tmp_path / "scores.jsonl"
    args = _judge_args(answers, str(output), fail_on_errors=True)

    with pytest.raises(SystemExit, match="1 of 2 answers are failed requests"):
        _run_judge(monkeypatch, args, [])
    assert not output.exists()


def test_judge_rejudges_failed_judge_calls_and_regenerated_answers(
    tmp_path, monkeypatch, capsys
):
    answers_path = tmp_path / "answers.jsonl"
    output = tmp_path / "scores.jsonl"
    args = _judge_args(_write(answers_path, [_answer(0), _answer(1)]), str(output))

    # Question 0: one judge call fails, one returns no content. Question 1: fine.
    _run_judge(monkeypatch, args, [
        RuntimeError("judge down"), None, '{"score": 1.0}', '{"score": 1.0}',
    ])
    first = capsys.readouterr().out
    assert "overall: 0.5000" in first
    assert "the judge failed: 1 questions" in first

    # Next run retries question 0 only.
    judge = _run_judge(monkeypatch, args, ['{"score": 1.0}', '{"score": 1.0}'])
    assert judge.calls == 2
    second = capsys.readouterr().out
    assert "overall: 1.0000" in second
    assert "the judge failed: 0 questions" in second

    # Question 1 is regenerated (resume appends): its old score is stale.
    with answers_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_answer(1, completion_id="cmpl-new")) + "\n")
    judge = _run_judge(monkeypatch, args, ['{"score": 0.0}', '{"score": 0.0}'])
    assert judge.calls == 2
    assert "overall: 0.5000" in capsys.readouterr().out
