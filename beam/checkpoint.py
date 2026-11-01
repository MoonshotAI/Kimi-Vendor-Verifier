"""Keep interrupted JSONL checkpoints safe to append on restart."""

import json
import os


def open_append(path: str):
    """Discard an incomplete final row, preserving complete rows byte for byte."""
    with open(path, "a+b") as stream:
        stream.seek(0, os.SEEK_END)
        end = stream.tell()
        if end:
            stream.seek(end - 1)
            if stream.read(1) != b"\n":
                position = end
                start = 0
                while position:
                    start = max(0, position - 8192)
                    stream.seek(start)
                    chunk = stream.read(position - start)
                    boundary = chunk.rfind(b"\n")
                    if boundary >= 0:
                        start += boundary + 1
                        break
                    position = start
                stream.seek(start)
                try:
                    json.loads(stream.read())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    stream.truncate(start)
                else:
                    stream.seek(0, os.SEEK_END)
                    stream.write(b"\n")
    return open(path, "a", encoding="utf-8")
