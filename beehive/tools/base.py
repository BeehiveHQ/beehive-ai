import json
import re
import uuid
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, PrivateAttr, model_validator

# OpenAI imports
try:
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
    )

# do nothing — if the user tries to instantiate an OpenAI model, they'll receive an
# error indicating that they need to pip install additional packages.
except ImportError:
    pass

from beehive.tools.google import GoogleParser
from beehive.tools.parser import SchemaParser
from beehive.tools.sphinx import SphinxParser
from beehive.tools.types import DocstringFormat


def create_parser(
    function: Callable[..., Any], docstring_format: DocstringFormat | None = None
) -> SchemaParser:
    function_name = function.__name__
    docstring = function.__doc__
    if docstring is None:
        raise ValueError(f"Function `{function_name}` is missing docstring!")

    if docstring_format == DocstringFormat.SPHINX:
        return SphinxParser(
            function=function, function_name=function_name, docstring=docstring
        )
    elif docstring_format == DocstringFormat.GOOGLE:
        return GoogleParser(
            function=function, function_name=function_name, docstring=docstring
        )
    elif docstring_format:
        raise ValueError(f"Unrecognized docstring format `{docstring_format}`.")
    else:
        # Detect docstring format using. This is pretty basic, but it'll do for now.
        if re.findall(r":param\s[A-Za-z0-9]*:", docstring):
            return SphinxParser(
                function=function, function_name=function_name, docstring=docstring
            )
        elif re.findall(r"Args:", docstring):
            return GoogleParser(
                function=function, function_name=function_name, docstring=docstring
            )
        else:
            raise ValueError("Could not detect docstring format!")


class BHTool(BaseModel):
    func: Callable[..., Any] = Field(
        description="Function that can be invoked by the agent."
    )
    docstring_format: DocstringFormat | None = Field(
        default=None, description="Docstring format."
    )
    _name: str = PrivateAttr()

    @model_validator(mode="after")
    def set_private_attrs(self) -> "BHTool":
        self._name = self.func.__name__
        return self

    @classmethod
    def from_langchain_structured_tool(cls, structured_tool: StructuredTool):
        if not structured_tool.func:
            raise ValueError("StructuredTool function not defined!")
        return cls(
            func=structured_tool.func,
            docstring_format=None,
        )

    def derive_json_specification(self) -> dict[str, str | dict[str, Any]]:
        parser = create_parser(self.func, self.docstring_format)
        spec = parser.parse()
        return spec.serialize()


class BHToolCall(BaseModel):
    tool: BHTool | None = Field(description="Tool that was called.")
    tool_name: str = Field(description="Name of the tool that was called.")
    tool_arguments: dict[str, Any] = Field(
        description="Keyword arguments for the tool's function."
    )
    tool_call_id: str = Field(
        default_factory=uuid.uuid4,
        description="ID for the tool call. If not specified, defaults to a UUID4.",
    )
    execute: bool = Field(
        default=True,
        description=(
            "Whether to execute the tool and generate its output upon creation."
            " Default is True."
        ),
    )
    _output: Any = PrivateAttr()

    @model_validator(mode="after")
    def set_private_attrs(self) -> "BHToolCall":
        if self.execute:
            if not self.tool:
                raise ValueError("`tool` is None!")
            self._output = self.tool.func(**self.tool_arguments)
        return self

    @classmethod
    def from_openai_tool_call(
        cls,
        tool: BHTool,
        completion_tool_call: ChatCompletionMessageToolCall,
    ):
        tool_call_id = completion_tool_call.id
        tool_arguments = json.loads(completion_tool_call.function.arguments)
        tool_name = completion_tool_call.function.name
        return cls(
            tool=tool,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            tool_call_id=tool_call_id,
        )

    def serialize(self):
        function_response = {
            "name": self.tool_name,
            "arguments": json.dumps(self.tool_arguments),
        }
        return {
            "id": self.tool_call_id,
            "type": "function",
            "function": function_response,
        }

    @property
    def output(self):
        return self._output
