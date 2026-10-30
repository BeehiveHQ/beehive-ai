import json
from collections import defaultdict
from typing import Any, Iterable, Literal, Optional, Tuple

from pydantic import BaseModel, PrivateAttr

from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.base import BHChatModel
from beehive.tools.base import BHTool, BHToolCall

# OpenAI imports
try:
    from openai import OpenAI
    from openai.types.chat.chat_completion import ChatCompletion
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

# do nothing — if the user tries to instantiate an OpenAI model, they'll receive an
# error indicating that they need to pip install additional packages.
except ImportError:
    pass


# Logger
import logging

logger = logging.getLogger(__file__)


class OpenAIModel(BHChatModel):
    """A wrapper class for interacting with OpenAI's chat models, inheriting from `BHChatModel`.

    This class facilitates the instantiation and usage of OpenAI's language models,
    such as `gpt-3.5-turbo`, with additional keyword arguments that may be required by the OpenAI API.

    args:
    - `model` (str): the name of the OpenAI language model (LLM) to use, e.g., `gpt-3.5-turbo`.
    - **`model_kwargs` (dict[str, Any]): keyword arguments passed to the `openai.OpenAI` client for model instantiation. These might include parameters such as `temperature`, `max_tokens`, etc.

    examples:
    - check out the documentation here: https://beehivehq.github.io/beehive-ai/
    """

    _client: OpenAI = PrivateAttr()
    structured_output_mode: Literal["auto", "native", "tool"] = "auto"

    def _structured_completion(
        self,
        response_model: type[BaseModel],
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
    ) -> str:
        # The SDK normalizes nested objects, references, and required fields for
        # OpenAI's strict JSON Schema subset without changing the Pydantic model.
        from openai import BadRequestError
        from openai.lib._pydantic import to_strict_json_schema

        # Reserve a name to distinguish final output from executable tools.
        name = "beehive_structured_output"
        if name in tools:
            raise ValueError(f"Tool name {name!r} is reserved for structured outputs")

        # Auto mode tries native schema enforcement before falling back to tools.
        mode = self.structured_output_mode
        native = mode != "tool"
        schema = (
            to_strict_json_schema(response_model)
            if native
            else response_model.model_json_schema()
        )

        # Bound tool execution independently from validation retries.
        for _ in range(10):
            response_format = None
            output_schema = None
            if native:
                response_format = {
                    "type": "json_schema",
                    "json_schema": {"name": name, "strict": True, "schema": schema},
                }
            else:
                # Deliver final JSON through a synthetic tool instead of text.
                output_schema = response_model.model_json_schema()
            try:
                messages = self.call_completions_api(
                    temperature,
                    tools,
                    conversation,
                    response_format=response_format,
                    output_schema=output_schema,
                )
            except BadRequestError as exc:
                body = exc.body if isinstance(exc.body, dict) else {}
                error = body.get("error", body)

                # If we are in "auto" mode and the response format is unsupported, then
                # try again with the explicit tool fallback.
                if (
                    mode == "auto"
                    and native
                    and error.get("param")
                    in ("response_format", "response_format.type")
                    and (
                        error.get("code") == "unsupported_value"
                        or "not supported" in str(error.get("message", "")).lower()
                    )
                ):
                    native = False
                    continue

                # All other errors (invalid schemas, authentication, and transport
                # errors) should be raised.
                raise

            response = messages[0]
            if not isinstance(response, BHMessage):
                raise TypeError("Expected an assistant message from the completion")
            if response.tool_calls:
                # Feed ordinary tool results into the next turn before final output.
                conversation.extend(messages)
                continue
            return response.content
        raise RuntimeError("Exceeded structured output tool turn limit (10)")

    def _create_client(self, **client_kwargs) -> OpenAI:
        return OpenAI(**client_kwargs)

    def convert_tool_calls_to_messages(
        self,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: dict[str, BHTool],
    ) -> Tuple[list[BHToolCall], list[BHMessage | BHToolMessage]]:
        tool_messages: list[BHMessage | BHToolMessage] = []

        # Create tool calls
        tc_for_msgs: list[BHToolCall] = []
        for tc in tool_calls:
            if tc.function.name not in tools:
                logger.warning(
                    f"{tc.function.name} not found in agent's tools! Available tools are {', '.join(list(tools.keys()))}."
                )
            else:
                tool_call_obj = BHToolCall.from_openai_tool_call(
                    tools[tc.function.name], tc
                )
                tc_for_msgs.append(tool_call_obj)

        # If the completion had tool calls, we need to create tool messages for each
        # tool call.
        for bh_tc in tc_for_msgs:
            tool_msg = BHToolMessage(
                tool_call_id=bh_tc.tool_call_id,
                name=bh_tc.tool_name,
                content=bh_tc.output,
            )
            tool_messages.append(tool_msg)

        return tc_for_msgs, tool_messages

    def convert_completion_to_messages(
        self,
        content: str | None,
        tool_calls: list[ChatCompletionMessageToolCall],
        tools: dict[str, BHTool],
    ) -> list[BHMessage | BHToolMessage]:
        final_messages: list[BHMessage | BHToolMessage] = []

        # Create messages
        tool_call_objs, tool_messages = self.convert_tool_calls_to_messages(
            tool_calls, tools
        )

        # Create a message that contains the completion's content
        message = BHMessage(
            role=MessageRole.ASSISTANT,
            content=content if content else "",
            tool_calls=tool_call_objs,
        )
        final_messages.append(message)
        final_messages.extend(tool_messages)
        return final_messages

    def call_completions_api(
        self,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
        *,
        response_format: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> list[BHMessage | BHToolMessage]:
        """Send one completion, extracting synthetic output before tool execution."""
        specs = [tool.derive_json_specification() for tool in tools.values()]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": temperature,
            "messages": [message.msg for message in conversation],
        }
        name = "beehive_structured_output"
        if response_format is not None:
            kwargs["response_format"] = response_format
        if output_schema is not None:
            # This tool carries the final response; it is never executed.
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": "Return the final structured response.",
                        "parameters": output_schema,
                    },
                }
            )
            kwargs["tool_choice"] = "required"
        kwargs["tools"] = specs or None
        completion: ChatCompletion = self._client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        calls = message.tool_calls or []
        structured = response_format is not None or output_schema is not None
        if structured:
            # Refusals and unknown tools are provider errors, not validation errors.
            if getattr(message, "refusal", None):
                raise ValueError(f"Model refused structured output: {message.refusal}")
            if output_schema is not None:
                output_calls = [call for call in calls if call.function.name == name]
                if output_calls:
                    if len(calls) != 1:
                        raise ValueError(
                            "Expected exactly one structured output tool call"
                        )
                    return [
                        BHMessage(
                            role=MessageRole.ASSISTANT,
                            content=output_calls[0].function.arguments,
                        )
                    ]
            unknown = [
                call.function.name for call in calls if call.function.name not in tools
            ]
            if unknown:
                raise ValueError(f"Unknown tool calls: {unknown}")
        # Missing fallback tool arguments must fail Pydantic validation, even if
        # the provider returned valid JSON as plain text.
        content = "" if output_schema is not None and not calls else message.content
        return self.convert_completion_to_messages(content, calls, tools)

    def stream_completions_api(
        self,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
        printer: Optional["Printer"] = None,  # type: ignore # noqa: F821
    ):
        # Call the Completions API. For some reason, chat.completions.create doesn't
        # like `None` in `tools`.
        completion: Iterable[
            ChatCompletionChunk
        ] = self._client.chat.completions.create(  # type: ignore
            model=self.model,
            temperature=temperature,
            tools=[v.derive_json_specification() for _, v in tools.items()]  # type: ignore
            if tools
            else None,
            messages=[c.msg for c in conversation],
            stream=True,
        )

        # Here, we just print the content as it appears
        role: str | None = None
        curr_tools: dict[int, str] = defaultdict(str)
        curr_tool_call_ids: dict[int, str] = defaultdict(str)
        curr_tool_arguments: dict[int, str] = defaultdict(str)
        final_content = ""
        final_tool_calls: list[BHToolCall] = []
        final_tool_messages: list[BHToolMessage] = []
        for chunk in completion:
            delta = chunk.choices[0].delta
            delta_tool_calls = delta.tool_calls
            delta_content = delta.content

            # Stream tool call arguments. These arguments stream in chunk-by-chunk.
            if delta_tool_calls:
                for tc in delta_tool_calls:
                    if tc.function:
                        # The `name` and `arguments`` attributes for a specific function
                        # are optional.
                        if tc.function.name:
                            curr_tools[tc.index] = tc.function.name
                        if tc.function.arguments:
                            curr_tool_arguments[tc.index] += tc.function.arguments
                    if tc.id:
                        curr_tool_call_ids[tc.index] = tc.id

            if not role and delta.role:
                role = delta.role

            # Print message content
            if delta_content:
                if printer:
                    printer._console.print(delta_content)
                final_content += delta_content

            # Tool calls have finished streaming
            if not delta_tool_calls and curr_tools:
                for idx, name in curr_tools.items():
                    bh_tool_call = BHToolCall(
                        tool=tools[name],
                        tool_name=name,
                        tool_arguments=json.loads(curr_tool_arguments[idx]),
                        tool_call_id=curr_tool_call_ids[idx],
                    )
                    bh_tool_message = BHToolMessage(
                        tool_call_id=curr_tool_call_ids[idx],
                        name=name,
                        content=bh_tool_call.output,
                    )
                    final_tool_calls.append(bh_tool_call)
                    final_tool_messages.append(bh_tool_message)
                    if printer:
                        printer.update_invokable_panel_with_content(bh_tool_call.output)

        # Completion messages
        completion_messages = [
            BHMessage(
                role=MessageRole(role) if role else MessageRole.ASSISTANT,
                content=final_content,
                tool_calls=final_tool_calls,
            )
        ]

        # mypy thinks `extend` should only accept an iterable, not a list. This is
        # wrong, and it seems like a mypy issue:
        # https://github.com/python/mypy/pull/17310
        completion_messages.extend(final_tool_messages)  # type: ignore
        return completion_messages

    def stream(
        self,
        task_message: BHMessage | None,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
        printer: Optional["Printer"] = None,  # type: ignore # noqa: F821
    ):
        if task_message:
            if task_message.msg["role"] != MessageRole.USER:
                raise ValueError(
                    f"Invalid `task_message` role. Expected 'user', found '{task_message.msg["role"]}'."
                )
            if task_message not in conversation:
                conversation.append(task_message)
        return self.stream_completions_api(temperature, tools, conversation, printer)
