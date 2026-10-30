from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.openai_model import OpenAIModel


class Answer(BaseModel):
    count: int = Field(gt=0)

    @model_validator(mode="after")
    def odd(self):
        if self.count % 2 == 0:
            raise ValueError("must be odd")
        return self


def completion(content=None, calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=calls, refusal=None)
            )
        ]
    )


def model(mode="auto"):
    result = OpenAIModel.model_construct(model="test", structured_output_mode=mode)
    result._client = cast(
        OpenAI,
        SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=Mock()))
        ),
    )
    return result


def run(m, retries=2):
    return m.chat(
        None,
        0,
        {},
        [BHMessage(role=MessageRole.USER, content="Count")],
        Answer,
        retries,
    )


def test_native_validates_custom_validators_and_repairs():
    m = model()
    m._client.chat.completions.create.side_effect = [
        completion('{"count":2}'),
        completion('{"count":3}'),
    ]
    messages = run(m)
    assert isinstance(messages[-1], BHMessage)
    assert isinstance(messages[-1].parsed, Answer)
    assert messages[-1].parsed.count == 3
    kwargs = m._client.chat.completions.create.call_args_list[0].kwargs
    assert kwargs["response_format"]["json_schema"]["strict"] is True
    assert (
        kwargs["response_format"]["json_schema"]["schema"]["additionalProperties"]
        is False
    )
    assert (
        "must be odd"
        in m._client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    )


@pytest.mark.parametrize("payload", ["", "not json", '{"count":0}', "{}"])
def test_validation_retries_are_bounded(payload):
    m = model()
    m._client.chat.completions.create.return_value = completion(payload)
    with pytest.raises(ValidationError):
        run(m)
    assert m._client.chat.completions.create.call_count == 3


def test_tool_fallback_does_not_execute_fake_tool():
    m = model("tool")
    call = SimpleNamespace(
        function=SimpleNamespace(
            name="beehive_structured_output", arguments='{"count":1}'
        )
    )
    m._client.chat.completions.create.return_value = completion(calls=[call])
    assert run(m)[-1].parsed.count == 1
    kwargs = m._client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs
    assert kwargs["tools"][0]["function"]["parameters"] == Answer.model_json_schema()


def test_auto_falls_back_only_on_unsupported_format():
    import httpx
    from openai import BadRequestError

    m = model()
    error = BadRequestError(
        "unsupported",
        response=httpx.Response(
            400, request=httpx.Request("POST", "https://example.com")
        ),
        body={"param": "response_format", "code": "unsupported_value"},
    )
    call = SimpleNamespace(
        function=SimpleNamespace(
            name="beehive_structured_output", arguments='{"count":1}'
        )
    )
    m._client.chat.completions.create.side_effect = [error, completion(calls=[call])]
    assert run(m)[-1].parsed.count == 1
    assert "response_format" not in m._client.chat.completions.create.call_args.kwargs


def test_transport_errors_are_not_validation_retries():
    m = model()
    m._client.chat.completions.create.side_effect = RuntimeError("offline")
    with pytest.raises(RuntimeError, match="offline"):
        run(m)
    assert m._client.chat.completions.create.call_count == 1


def test_agent_response_model_needs_no_termination_condition():
    from beehive.invokable.agent import BeehiveAgent

    m = model()
    m._client.chat.completions.create.return_value = completion('{"count":1}')
    with patch.object(OpenAIModel, "_create_client", return_value=m._client):
        agent = BeehiveAgent(name="structured", model=m, response_model=Answer)
    messages = agent._invoke("Count", verbose=False)
    assert isinstance(messages[-1], BHMessage)
    assert isinstance(messages[-1].parsed, Answer)


def test_agent_does_not_multiply_validation_retry_budget():
    from beehive.invokable.agent import BeehiveAgent

    m = model()
    m._client.chat.completions.create.return_value = completion('{"count":2}')
    with patch.object(OpenAIModel, "_create_client", return_value=m._client):
        agent = BeehiveAgent(
            name="structured",
            model=m,
            response_model=Answer,
            structured_output_retries=1,
        )
    with pytest.raises(ValidationError):
        agent._invoke("Count", verbose=False, pass_back_model_errors=True)
    assert m._client.chat.completions.create.call_count == 2


def test_chat_without_schema_preserves_plain_text_and_task_history():
    m = model()
    m._client.chat.completions.create.return_value = completion("Plain answer")
    task = BHMessage(role=MessageRole.USER, content="Question")
    conversation: list[BHMessage | BHToolMessage] = []
    messages = m.chat(task, 0, {}, conversation)
    assert messages[0].content == "Plain answer"
    assert messages[0].parsed is None
    assert conversation == [task]
    kwargs = m._client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.parametrize("structured", [False, True])
def test_chat_rejects_invalid_task_role(structured):
    m = model()
    task = BHMessage(role=MessageRole.ASSISTANT, content="Wrong role")
    with pytest.raises(ValueError, match="role"):
        m.chat(task, 0, {}, [], response_model=Answer if structured else None)
    m._client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("mode", ["native", "tool"])
def test_structured_chat_runs_ordinary_tool_before_final_output(mode):
    from beehive.tools.base import BHTool
    from beehive.tools.types import DocstringFormat

    def get_count(label: str) -> str:
        """Get the current count.

        Args:
            label: Name of the counter.

        Returns:
            The current count.
        """
        return "1"

    m = model(mode)
    tool = BHTool(func=get_count, docstring_format=DocstringFormat.GOOGLE)
    ordinary_call = SimpleNamespace(
        id="call_count",
        function=SimpleNamespace(name="get_count", arguments='{"label":"counter"}'),
    )
    final_call = SimpleNamespace(
        function=SimpleNamespace(
            name="beehive_structured_output", arguments='{"count":1}'
        )
    )
    m._client.chat.completions.create.side_effect = [
        completion(calls=[ordinary_call]),
        completion('{"count":1}')
        if mode == "native"
        else completion(calls=[final_call]),
    ]
    conversation = [BHMessage(role=MessageRole.USER, content="Count")]
    messages = m.chat(None, 0, {"get_count": tool}, conversation, response_model=Answer)
    assert messages[0].tool_calls[0].tool_name == "get_count"
    assert messages[1].content == "1"
    assert messages[-1].parsed.count == 1
    assert len(conversation) == 1
    kwargs = m._client.chat.completions.create.call_args.kwargs
    assert kwargs["messages"][-1]["role"] == "tool"
    assert kwargs["messages"][-1]["content"] == "1"


def test_repair_prompts_do_not_leak_into_conversation():
    m = model()
    m._client.chat.completions.create.side_effect = [
        completion('{"count":2}'),
        completion('{"count":1}'),
    ]
    task = BHMessage(role=MessageRole.USER, content="Count")
    conversation: list[BHMessage | BHToolMessage] = []
    messages = m.chat(task, 0, {}, conversation, response_model=Answer)
    assert conversation == []
    assert len(messages) == 1
    assert messages[0].parsed.count == 1
