import openai

from tests.helper import token_kwarg


def test_stop_matches_content_not_reasoning(client: openai.Client, model: str):
    """A stop sequence in reasoning_content does not stop content generation."""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly <block>no</block>."}],
        stop=["</block>"],
        **token_kwarg(256),
    )

    choice = completion.choices[0]
    reasoning = getattr(choice.message, "reasoning_content", None) or ""
    assert "</block>" in reasoning, (
        f"expected the stop sequence in reasoning_content, got: {reasoning!r}"
    )
    assert choice.message.content == "<block>no", (
        f"stop should truncate content at </block>, got: {choice.message.content!r}"
    )
    assert choice.finish_reason == "stop", choice
