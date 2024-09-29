import json
from enum import Enum, StrEnum
from typing import Final, Optional, TypedDict, Union

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import BaseMessage, ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from beehive.message import BHMessage, BHToolMessage
from beehive.models.base import BHChatModel
from beehive.tools.base import BHTool, BHToolCall

# Types
AnyBHMessage = Union[BHMessage, BHToolMessage]
AnyMessage = Union[BaseMessage, BHMessage, BHToolMessage]
AnyBHMessageSequence = list[AnyBHMessage]
AnyMessageSequence = AnyBHMessageSequence | list[BaseMessage]

AnyChatModel = Union[BHChatModel, BaseChatModel]
AnyLCTool = Union[BaseTool, StructuredTool]
AnyTool = Union[BHTool, BaseTool, StructuredTool]
AnyToolCall = Union[ToolCall, BHToolCall]


class ColorFormat(StrEnum):
    RICH: Final = "rich"
    HEX: Final = "hex"
    RGB: Final = "rgb"


class InvokableQuestion(BaseModel):
    question: str = Field(
        description=(
            "Question to pose to a a different agent. Note that these agents"
            " like you, are powered by language models."
        )
    )
    reason: str = Field(
        description="Reason this question is necessary for you to complete your task."
    )
    invokable: str = Field(description="Agent to whom the question will be asked.")


class BHStateElt(BaseModel):
    """Special output class for Beehives"""

    index: int = Field()
    task_id: str | None = Field()
    task: str = Field()
    invokable: Optional["Invokable"] = Field()  # type: ignore # noqa: F821
    completion_messages: AnyBHMessageSequence = Field()

    def __str__(self) -> str:
        obj = {
            "index": self.index,
            "task": self.task,
            "invokable": self.invokable.name if self.invokable else "",
            "messages": [x.content for x in self.completion_messages],
        }
        return json.dumps(obj)


class EmbeddingDistance(str, Enum):
    """
    See documentation here:
    https://docs.trychroma.com/guides#changing-the-distance-function
    """

    L2: Final = "l2"
    IP: Final = "ip"
    COSINE: Final = "cosine"


class ExecutorOutput(TypedDict):
    task_id: str | None
    messages: AnyMessageSequence | list[BHStateElt]
    printer: Optional["Printer"]  # type: ignore # noqa: F821
