"""Exercise both BEAM resume drivers against interrupted JSONL files."""

import asyncio
import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from beam import beam_generate, beam_judge


@pytest.fixture
def endpoint():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(
                {
                    "id": "local-completion",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "local-test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"score": 1.0}',
                            },
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("driver", ["generate", "judge"])
@pytest.mark.parametrize(
    "tail",
    [
        "partial-json",
        "partial-utf8",
        "complete-row",
        "complete-line",
        "long-partial",
        "long-row",
        "new",
    ],
)
def test_resume_preserves_completed_rows(tmp_path, endpoint, driver, tail):
    base_url, requests = endpoint
    rows = [
        {
            "chat_id": "chat",
            "question_type": "information_extraction",
            "question_index": index,
            "question": "What was remembered?",
            "gold_answer": "回答",
            "rubric": ["The answer is correct."],
            "response": "回答",
            "score": 1.0,
        }
        for index in range(3)
    ]
    output = tmp_path / "output.jsonl"
    prefix = json.dumps(rows[0], ensure_ascii=False).encode() + b"\n"
    if tail == "long-row":
        rows[1]["response"] *= 5000
    second = json.dumps(rows[1], ensure_ascii=False).encode()
    if tail in ("complete-row", "long-row"):
        damaged = second
    elif tail == "complete-line":
        damaged = second + b"\n"
    elif tail == "partial-utf8":
        damaged = second[: second.index("回".encode()) + 1]
    elif tail == "long-partial":
        damaged = b'{"chat_id":' + b" " * 9000
    else:
        damaged = b'{"chat_id": "chat", "question_index":'
    if tail != "new":
        output.write_bytes(prefix + damaged)
    args = SimpleNamespace(
        output=str(output),
        limit=0,
        concurrency=1,
        max_retries=1,
        timeout=5,
        model="local-test",
        base_url=base_url,
        api_key="local-test",
        dry_run=False,
        thinking_json="",
        temperature=1.0,
        top_p=0.95,
        max_tokens=16,
        _tokenizer=None,
        _max_content_tokens=100,
        judge_model="local-test",
        judge_base_url=base_url,
        judge_api_key="local-test",
        judge_reasoning_effort="",
        judge_max_tokens=16,
        judge_temperature=0.3,
    )
    if driver == "generate":
        chats = tmp_path / "chats.gz"
        with gzip.open(chats, "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"chat_id": "chat", "messages": []}) + "\n")
        questions = tmp_path / "questions.jsonl"
        questions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        args.chats_file, args.questions_file = str(chats), str(questions)
        run = beam_generate.run
    else:
        answers = tmp_path / "answers.jsonl"
        answers.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        args.answers = str(answers)
        run = beam_judge.run

    asyncio.run(run(args))
    saved = [
        json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["question_index"] for row in saved] == [0, 1, 2]
    if tail != "new":
        assert output.read_bytes().startswith(prefix)
    assert output.read_bytes().endswith(b"\n")
    expected_requests = (
        3
        if tail == "new"
        else (1 if tail in ("complete-row", "complete-line", "long-row") else 2)
    )
    assert len(requests) == expected_requests
    asyncio.run(run(args))
    assert len(requests) == expected_requests
