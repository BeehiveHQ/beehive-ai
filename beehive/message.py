import json
from enum import StrEnum

from pydantic import BaseModel, Field

from beehive.tools.base import BHToolCall


class MessageRole(StrEnum):
    SYSTEM: str = "system"
    USER: str = "user"
    ASSISTANT: str = "assistant"

    # This type enables us to distinguish context messages from other assistant
    # messages.
    CONTEXT: str = "context"

    # This type enables us to distinguish question messages from other assistant
    # messages.
    QUESTION: str = "question"


class BHMessage(BaseModel):
    role: MessageRole = Field(
        description="The message's role. The role can take one of three values: `system`, `user`, or `assistant`."
    )
    content: str = Field(description="The message's content.")
    tool_calls: list[BHToolCall] = Field(
        default_factory=list,
        description="List of tool calls made by the agent. Default is [].",
    )

    def add_to_content(self, new_content: str, newline: bool = True):
        newline_char = "\n" if newline else ""
        self.content += newline_char + new_content

    @property
    def processed_role(self) -> MessageRole:
        if self.role == MessageRole.CONTEXT:
            return MessageRole.ASSISTANT
        elif self.role == MessageRole.QUESTION:
            return MessageRole.USER
        else:
            return self.role

    @property
    def msg(self):
        if self.tool_calls:
            return {
                "role": self.processed_role,
                "content": self.content,
                "tool_calls": [tc.serialize() for tc in self.tool_calls],
            }
        else:
            return {
                "role": self.processed_role,
                "content": self.content,
            }

    def __str__(self):
        return f"{self.__class__.__name__}(role='{self.role}', content='{self.content}', tool_calls={[x.serialize for x in self.tool_calls]})"

    def pprint(self):
        if self.content:
            print(self.content)


class BHToolMessage(BaseModel):
    tool_call_id: str = Field(description="Tool call ID assigned by the model.")
    name: str | None = Field(description="Name of the tool.")
    content: str | None = Field(
        default=None, description="Output of the tool call. Default is None."
    )

    @property
    def msg(self):
        base = {
            "role": "tool",
            "tool_call_id": self.tool_call_id,
            "content": str(self.content),
        }
        if self.name:
            base["name"] = self.name
        return base

    def __str__(self):
        return json.dumps(self.msg)

    def pprint(self):
        if self.content:
            print(
                f"Tool {self.name} was called, and the result was: {str(self.content)}."
            )
