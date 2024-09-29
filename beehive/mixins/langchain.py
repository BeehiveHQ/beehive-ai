from typing import Any
from uuid import uuid4

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import StructuredTool

from beehive.invokable.types import AnyMessage, AnyToolCall
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.tools.base import BHTool, BHToolCall


class LangchainMixin:
    def _convert_langchain_tool_call_to_beehive_tool_call(self, tool_call: AnyToolCall):
        if isinstance(tool_call, BHToolCall):
            return tool_call
        else:
            tool_call_id = tool_call.get("id", None)
            if not tool_call_id:
                tool_call_id = f"tool_call_{uuid4()}"

            # The parent class needs to have a `_tools_map` for this function to work
            # properly.
            if not hasattr(self, "_tools_map"):
                raise ValueError(
                    f"Cannot access tool `{tool_call['name']}` from `_tools_map` attribute!"
                )

            # Langchain tool must be a StructuredTool
            lc_tool = self._tools_map.get(tool_call["name"], None)
            if not isinstance(lc_tool, StructuredTool):
                raise ValueError("Langchain tool is not a StructuredTool instance!")

            return BHToolCall(
                tool=BHTool.from_langchain_structured_tool(lc_tool),
                tool_name=tool_call["name"],
                tool_arguments=tool_call["args"],
                tool_call_id=tool_call_id,
            )

    def _convert_langchain_message_to_beehive_message(
        self, message: AnyMessage
    ) -> BHMessage | BHToolMessage:
        if isinstance(message, BHMessage) or isinstance(message, BHToolMessage):
            return message
        elif isinstance(message, HumanMessage):
            return BHMessage(
                role=MessageRole.USER,
                content=str(message.content),
            )
        elif isinstance(message, SystemMessage):
            return BHMessage(
                role=MessageRole.SYSTEM,
                content=str(message.content),
            )
        elif isinstance(message, AIMessage):
            return BHMessage(
                role=MessageRole.ASSISTANT,
                content=str(message.content),
                tool_calls=[
                    self._convert_langchain_tool_call_to_beehive_tool_call(tc)
                    for tc in message.tool_calls
                ],
            )
        elif isinstance(message, ToolMessage):
            return BHToolMessage(
                tool_call_id=message.tool_call_id,
                name=message.name,
                content=str(message.content),
            )
        else:
            raise ValueError(
                f"Beehive cannot convert message class `{message.__class__.__name__}`"
            )

    def _convert_beehive_tool_call_to_langchain_tool_call(self, tool_call: AnyToolCall):
        if isinstance(tool_call, BHToolCall):
            return ToolCall(
                name=tool_call.tool_name,
                args=tool_call.tool_arguments,
                id=tool_call.tool_call_id,
            )
        else:
            return tool_call

    def _convert_beehive_message_to_langchain_message(
        self, message: AnyMessage
    ) -> BaseMessage:
        if isinstance(message, BaseMessage):
            return message
        elif isinstance(message, BHMessage):
            match message.role:
                case MessageRole.USER:
                    return HumanMessage(content=message.content)
                case MessageRole.SYSTEM:
                    return SystemMessage(content=message.content)
                case MessageRole.ASSISTANT:
                    return AIMessage(
                        content=message.content,
                        tool_calls=[
                            self._convert_beehive_tool_call_to_langchain_tool_call(tc)
                            for tc in message.tool_calls
                        ],
                    )
                case MessageRole.CONTEXT:
                    return AIMessage(
                        content=message.content,
                        tool_calls=[
                            self._convert_beehive_tool_call_to_langchain_tool_call(tc)
                            for tc in message.tool_calls
                        ],
                    )
                case _:
                    raise ValueError("Unknown role! Cannot instantiate message class.")

        elif isinstance(message, BHToolMessage):
            return ToolMessage(
                tool_call_id=message.tool_call_id,
                name=message.name,
                content=str(message.content),
            )

    def _convert_state_to_langchain_base_message(
        self, state: list[Any]
    ) -> list[BaseMessage]:
        # TODO iterates through the state every single time... figure out a more
        # efficient way to handle this.
        new_state: list[BaseMessage] = []
        for msg in state:
            if (
                isinstance(msg, BHMessage)
                or isinstance(msg, BHToolMessage)
                or isinstance(msg, BaseMessage)
            ):
                new_state.append(
                    self._convert_beehive_message_to_langchain_message(msg)
                )
        return new_state
