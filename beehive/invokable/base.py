import datetime
import json
import logging
import re
from typing import Any, Callable
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    InstanceOf,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import ValidationError
from rich.logging import RichHandler
from sqlalchemy import func, select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.functions import Function
from sqlalchemy.sql.sqltypes import Boolean
from sqlalchemy.types import NullType

from beehive.invokable.types import (
    AnyChatModel,
    AnyMessage,
    BHStateElt,
    ColorFormat,
    EmbeddingDistance,
    ExecutorOutput,
)
from beehive.memory.db_storage import (
    DbStorage,
    MessageModel,
    TaskModel,
    ToolCallModel,
    ToolMessageModel,
)
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.base import BHChatModel, BHEmbeddingModel
from beehive.models.openai_embedder import OpenAIEmbedder
from beehive.prompts import BHPrompt, FullContextPrompt
from beehive.tools.base import BHTool, BHToolCall
from beehive.tools.types import DocstringFormat
from beehive.utilities.printer import Printer

# Logging
logging.basicConfig(
    level="WARN",
    format="%(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__file__)


class Feedback(BaseModel):
    confidence: int
    suggestions: list[str]


class Context(BaseModel):
    agent_backstory: str
    agent_messages: list[str]


class Route(BaseModel):
    this: Any = Field(description="Invokable or Route that acts first.")
    other: Any = Field(description="Invokable or Route that acts second.")
    id: str = Field(default_factory=uuid4)

    _invokable_order: list["Invokable"] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def validate_input_types(self) -> "Route":
        if not (isinstance(self.this, Invokable) or isinstance(self.this, Route)):
            raise ValueError(
                "Route can only be constructed with invokables or other routes!"
            )

        if not (isinstance(self.other, Invokable) or isinstance(self.other, Route)):
            raise ValueError(
                "Route can only be constructed with invokables or other routes!"
            )
        if self.this == self.other:
            raise ValueError(
                "Route can only be defined between two different invokables or routes!"
            )

        # Construct the invokable order
        self._invokable_order = self._construct_invokable_order()
        return self

    def _construct_invokable_order(self) -> list["Invokable"]:
        invokable_order = (
            [self.this]
            if isinstance(self.this, Invokable)
            else self.this._construct_invokable_order()
        )
        invokable_order.extend(
            [self.other]
            if isinstance(self.other, Invokable)
            else self.other._construct_invokable_order()
        )
        # Because of some import stuff, self.this and self.other are technically `Any`,
        # not `Invokable` instances. We validate our Pydantic model, but mypy still gets
        # confused.
        return invokable_order  # type: ignore

    def __str__(self):
        this_name = (
            self.this.name if isinstance(self.this, Invokable) else self.this.__str__()
        )
        other_name = (
            self.other.name
            if isinstance(self.other, Invokable)
            else self.other.__str__()
        )
        return f"{this_name} >> {other_name}"

    def __rshift__(self, other: Any) -> "Route":
        return self.__class__(
            this=self,
            other=other,
        )


class Invokable(BaseModel):
    """Base actor class. Invokables invoke an LLM to answer some user task. You will
    never need to instantiate this class directly — you should always use one of the
    more complete child classes, e.g., BeehiveAgent, BeehiveLangchainAgent,
    BeehiveDebateTeam, etc.

    args:
    - `name` (str): the invokable name.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel` | `BaseChatModel`): chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`.
    - `state` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `history` (bool): whether to use previous interactions / messages when responding to the current task. Default is `False`.
    - `history_lookback` (int): number of days worth of previous messages to use for answering the current task.
    - `feedback` (bool): whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used.
    - `feedback_embedder` (`BHEmbeddingModel` | None): embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`.
    - `feedback_model` (`BHChatModel` | `BaseChatModel`): language model used to generate feedback for the invokable. If `None`, then default to the `model` attribute.
    - `feedback_embedding_distance` (`EmbeddingDistance`): distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function.
    - `n_feedback_results` (int): amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`.
    - `color` (str): color used to represent the invokable in verbose printing. This can be a HEX code, an RGB code, or a standard color supported by the Rich API. See https://rich.readthedocs.io/en/stable/appendix/colors.html for more details. Default is `chartreuse2`.

    raises:
    - `pydantic_core.ValidationError`
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Name of the AI actor.")
    backstory: str = Field(
        default="You are a helpful AI assistant.",
        description=(
            "Backstory for the AI actor. This is used to prompt the AI actor and direct"
            " tasks towards it. Default is: 'You are a helpful AI assistant.'"
        ),
    )
    model: Any = Field(
        description=(
            "ChatModel used by the invokable to execute its function. This can be a "
            "Beehive ChatModel or a Langchain ChatModel."
        )
    )
    state: list[Any] = Field(
        default_factory=list,
        description=(
            "List of messages that this actor has seen. This enables the actor to build"
            " off of previous conversations / outputs."
        ),
    )
    history: bool = Field(
        default=False,
        description=(
            "Whether to use previous interactions / messages when responding to the"
            " current task. Default is False."
        ),
    )
    history_lookback: int = Field(
        default=1,
        description=(
            "Number of days worth of previous messages to use for answering the current"
            " task."
        ),
    )
    feedback: bool = Field(
        default=False,
        description=(
            "Whether to use feedback from the invokable's previous interactions. Feedback"
            " enables the LLM to improve their responses over time. Note that only feedback"
            " from tasks with a similar embedding are used."
        ),
    )
    feedback_embedder: BHEmbeddingModel | None = Field(
        default=None,
        description=(
            "Embedding model used to calculate embeddings of tasks. These embeddings are"
            " stored in a vector database. When a user asks task, Beehive searches against"
            " this vector database using the task embedding. If then takes the suggestions"
            " generated for similar, previous tasks and concatenates them to the task prompt."
            " Default is None."
        ),
    )
    feedback_model: AnyChatModel | None = Field(
        default=None,
        description=(
            "Language model used to generate feedback for the invokable. If `None`,"
            " then default to the `model` attribute."
        ),
    )
    feedback_embedding_distance: EmbeddingDistance = Field(
        default=EmbeddingDistance.L2,
        description=(
            "Distance method of the embedding space. See the ChromaDB documentation"
            " for more information: https://docs.trychroma.com/guides#changing-the-distance-function."
        ),
    )
    n_feedback_results: int = Field(
        default=1,
        description=(
            "Amount of feedback to incorporate into answering the current task. This"
            " takes `n` tasks with the most similar embedding to the current one and"
            " incorporates their feedback into the Invokable's model. Default is 1."
        ),
    )
    color: str = Field(
        default="chartreuse2",
        description=(
            "Color used to represent the invokable in verbose printing. This can be"
            " a HEX code, an RGB code, or a standard color supported by the Rich API."
            " See https://rich.readthedocs.io/en/stable/appendix/colors.html for more"
            " details."
        ),
    )
    _flag_has_seen_question_schema: bool = PrivateAttr(default=False)
    _context_messages: dict[str, Context] = PrivateAttr(default_factory=dict)
    _history_messages: list[BHMessage | BHToolMessage] = PrivateAttr()
    _compatible_with_memory: bool = PrivateAttr(default=True)
    _db_storage: InstanceOf[DbStorage] = PrivateAttr()
    _color_format: ColorFormat = PrivateAttr()
    _logged: bool = PrivateAttr(default=False)

    @field_validator("model")
    @classmethod
    def validate_model_class(cls, v):
        """Validate the model class. We need to do this via a custom validator rather
        than type-hinting, because Langchain uses Pydantic V1. If type-hint the `model`
        and instantiate an Invokable with a BaseChatModel, we get an error that reads:

        `TypeError: BaseModel.validate() takes 2 positional arguments but 3 were given`
        """
        if not (isinstance(v, BHChatModel) or isinstance(v, BaseChatModel)):
            raise ValidationError(f"Unsupported model class {v.__class__.__name__}")
        return v

    @model_validator(mode="after")
    def define_color_format(self) -> "Invokable":
        # Minor string processing
        self.color = self.color.strip()
        self.color = self.color.lower()

        if re.findall(r"^#[0-9A-Za-z]{6}$", self.color):
            self._color_format = ColorFormat.HEX
        elif re.findall(r"^rgb\([0-9]{1,3}\,[0-9]{1,3}\,[0-9]{1,3}\)$", self.color):
            self._color_format = ColorFormat.RGB
        else:
            self._color_format = ColorFormat.RICH
        return self

    @model_validator(mode="after")
    def validate_feedback_fields(self) -> "Invokable":
        # Feedback stuff
        if self.feedback:
            # Define the embedding model. This should default to an OpenAI embedder.
            # Note that if the user does not have the OPENAI_API_KEY specified, then
            # this will fail.
            if not self.feedback_embedder:
                self.feedback_embedder = OpenAIEmbedder()

            # Feedback model
            if self.feedback_model is None:
                self.feedback_model = self.model
        else:
            if self.feedback_embedder is not None:
                # Use the _logged variable to ensure that warning's are not repeated.
                if not self._logged:
                    logger.warning(
                        "Specified an embedding model, but `feedback=False`! Set `feedback=True` to store model outputs and recommendations."
                    )
                    self._logged = True
        return self

    @model_validator(mode="after")
    def initiate_db_storage(self) -> "Invokable":
        self._db_storage = DbStorage()
        return self

    def __rshift__(self, other: Any):
        return Route(
            this=self,
            other=other,
        )

    def __hash__(self) -> int:
        return hash((self.name, self.backstory))

    def grab_history_for_invokable_execution(self) -> list[Any]:
        history: list[BHMessage | BHToolMessage] = []

        # Compute WHERE expression for number of days. This exact function will depend
        # on the type of database used.
        days_where_clause: Function[NullType] | ColumnElement[Boolean]
        if "sqlite" in self._db_storage.db_uri:
            days_where_clause = (
                func.julianday(datetime.datetime.now(datetime.timezone.utc))
                - func.julianday(MessageModel.created_at)
                <= self.history_lookback
            )
        elif "postgres" in self._db_storage.db_uri:
            days_where_clause = func.extract(
                "day",
                func.age(
                    datetime.datetime.now(datetime.timezone.utc),
                    MessageModel.created_at,
                ),
            )

        # TODO store supported databases in a enum or something
        else:
            raise ValueError("Unsupported database type!")

        # Messages
        messages = self._db_storage.get_model_objects(
            (
                select(MessageModel)
                .join(TaskModel, TaskModel.id == MessageModel.task)
                .where(
                    TaskModel.invokable == self.name,
                    days_where_clause,
                )
                .order_by(MessageModel.created_at)
            )
        )
        for m in messages:
            # Tool calls
            tcs = self._db_storage.get_model_objects(
                (select(ToolCallModel).where(ToolCallModel.message == m.id))
            )

            # Create the ToolCall objects
            message_tool_calls: list[BHToolCall] = []
            for tc in tcs:
                tool = None
                if hasattr(self, "_tools_map"):
                    if self._tools_map:
                        # Make sure it's a BHTool
                        tool = self._tools_map[tc.name]
                        if isinstance(tool, StructuredTool):
                            tool = BHTool.from_langchain_structured_tool(
                                self._tools_map[tc.name]
                            )
                bh_tool_call = BHToolCall(
                    tool=tool,
                    tool_name=tc.name,
                    tool_arguments=tc.args,
                    tool_call_id=tc.tool_call_id,
                    execute=False,  # we don't want to execute historical tools
                )
                message_tool_calls.append(bh_tool_call)

            # Create the Message
            bh_message = BHMessage(
                role=m.role,
                content=m.content,
                tool_calls=message_tool_calls,
            )
            history.append(bh_message)

            # If the message has tool calls, then grab the associated tool messages
            tool_messages: list[BHToolMessage] = []
            for tc in tcs:
                tool_message_objects = self._db_storage.get_model_objects(
                    (
                        select(ToolMessageModel).where(
                            ToolMessageModel.tool_call_id == tc.tool_call_id
                        )
                    )
                )
                for tm in tool_message_objects:
                    tool_message = BHToolMessage(
                        tool_call_id=tc.tool_call_id,
                        name=tc.name,
                        content=tm.content,
                    )
                    tool_messages.append(tool_message)
            history.extend(tool_messages)

        return history

    def _add_message_contents_to_list(
        self, msg: AnyMessage, inv_message_contents: list[str]
    ) -> list[str]:
        if isinstance(msg, BHToolMessage):
            if msg.content:
                inv_message_contents.append(msg.content)
        elif isinstance(msg, BHMessage) and msg.role == MessageRole.ASSISTANT:
            # Skip messages with context
            if msg.content:
                inv_message_contents.append(msg.content)

        # Assuming the content of these messages are strings. According to
        # Langchain type-hinting, they could be a list of strings or dicts.
        # Ignore that for now.
        elif isinstance(msg, AIMessage) or isinstance(msg, ToolMessage):
            if msg.content:
                # With Langchain, we can't easily check if the message is a context
                # message. For now, we'll just search the content for
                # <context></context> and some other Beehive-specific stuff.
                flag_is_context = (
                    "<context>" in str(msg.content)
                    and "</context>" in str(msg.content)
                    and (
                        "Here is some context provided by previous LLM agents in the conversation. The format for this context is the same as before."
                        in str(msg.content)
                        or '{"type": "object", "properties": {"agent_backstory": {"type": "string"}, "agent_messages": {"type": "array", "items": [{"type": "string"}]}, "agent_name": {"type": "string"}}, "required": ["agent_backstory", "agent_messages", "agent_name"]}'
                        in str(msg.content)
                    )
                )
                if not flag_is_context:
                    inv_message_contents.append(str(msg.content))
        return inv_message_contents

    def construct_context_dictionary(
        self,
        invokables: list["Invokable"] | None,
        context_messages: dict[str, Context] | None = None,
    ) -> dict[str, Context]:
        if not invokables:
            return {}

        # Initialize empty dictionary
        all_context_messages: dict[str, Context] = (
            {} if context_messages is None else context_messages
        )

        # Iterate through the invokables and gather their messages. Only focus on
        # message with role `Assistant`, since these were produced by the LLM itself.
        for inv in invokables:
            # Don't include invokables own messages in context
            if inv.name == self.name and inv.backstory == self.backstory:
                continue
            inv_state_elts = inv.state

            # Because of import hell, we can't directly check isinstance(inv, Beehive).
            # Use the state as a proxy.
            flag_is_beehive = (
                False
                if not inv_state_elts
                else isinstance(inv_state_elts[0], BHStateElt)
            )
            if flag_is_beehive:
                if not hasattr(inv, "_invokables"):
                    raise ValueError("Beehive does not have `_invokables` attribute!")
                all_context_messages = self.construct_context_dictionary(
                    inv._invokables, all_context_messages
                )

            # Grab content
            else:
                inv_message_contents: list[str] = []
                for elt in inv_state_elts:
                    inv_message_contents = self._add_message_contents_to_list(
                        elt, inv_message_contents
                    )

                # Only add context if `inv_message_contents` is non-empty. Also, if the
                # invokable already exists in `all_context_messages` (e.g., maybe from a
                # previous Beehive), then edit the existing `Context` object instead of
                # creating a new one.
                if inv_message_contents:
                    if inv.name in list(all_context_messages.keys()):
                        all_context_messages[inv.name].agent_messages.extend(
                            inv_message_contents
                        )
                    else:
                        inv_context = Context(
                            agent_backstory=inv.backstory,
                            agent_messages=inv_message_contents,
                        )
                        all_context_messages[inv.name] = inv_context

        return all_context_messages

    def remove_duplicate_context(
        self, context_dict: dict[str, Context]
    ) -> dict[str, Context]:
        context_for_current_invokation: dict[str, Context] = {}
        for inv_name, inv_context in context_dict.items():
            # If this agent has never seen context from the invokable, then add it to
            # the context for the current invokation.
            if inv_name not in self._context_messages:
                context_for_current_invokation[inv_name] = inv_context

                # Add new context to self._context_messages
                self._context_messages[inv_name] = inv_context

            # If the agent has seen context from the invokable, only add the messages it
            # has not seen.
            else:
                # The backstories should match
                if (
                    inv_context.agent_backstory
                    != self._context_messages[inv_name].agent_backstory
                ):
                    raise ValueError("Backstories for invokables do not match!")

                new_messages = list(
                    set(inv_context.agent_messages)
                    - set(self._context_messages[inv_name].agent_messages)
                )
                if new_messages:
                    context_for_current_invokation[inv_name] = Context(
                        agent_backstory=inv_context.agent_backstory,
                        agent_messages=new_messages,
                    )

                    # Add new messages to self._context_messages
                    self._context_messages[inv_name].agent_messages.extend(new_messages)
        return context_for_current_invokation

    def augment_invokable_with_context(
        self,
        invokables: list["Invokable"] | None,
        context_template: type[BHPrompt] = FullContextPrompt,
    ) -> BHMessage | None:
        context_dict = self.construct_context_dictionary(invokables)
        context_dict_current_invokation = self.remove_duplicate_context(context_dict)

        # If the dictionary is empty, return None
        if not context_dict_current_invokation:
            return None

        # Convert context to string
        context_strings: list[str] = []
        for name, c in context_dict_current_invokation.items():
            c_dict = c.model_dump()
            c_dict["agent_name"] = name
            context_strings.append(json.dumps(c_dict))
        return BHMessage(
            role=MessageRole.CONTEXT,
            content=context_template(  # type: ignore
                context="\n".join([f"- {c}" for c in context_strings])
            ).render(),
        )

    def _invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[Any]:
        """Invoke the Invokable to execute a task.

        args:
        - `task` (str): task to execute.
        - `retry_limit` (int): maximum number of retries before the Invokable returns an error. Default is `100`.
        - `pass_back_model_errors` (bool): boolean controlling whether to pass the contents of an error back to the LLM via a prompt.  Default is `False`.
        - `verbose` (bool): beautify stdout logs with the `rich` package. Default is `True`.
        - `stream` (bool): stream the output of the agent character-by-character. Default is `False`.
        - `stdout_printer` (`output.printer.Printer` | None): Printer object to handle stdout messages. Default is `None`.

        returns:
        - list[BHMessage | BHToolMessage] | list[BaseMessage] | list[BHStateElt]
        """
        raise NotImplementedError()

    def invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        context: list["Invokable"] | None = None,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> ExecutorOutput:
        """Invoke the Invokable to execute a task.

        args:
        - `task` (str | `BHMessage`): task to execute.
        - `retry_limit` (int): maximum number of retries before the Invokable returns an error. Default is `100`.
        - `pass_back_model_errors` (bool): boolean controlling whether to pass the contents of an error back to the LLM via a prompt.  Default is `False`.
        - `verbose` (bool): beautify stdout logs with the `rich` package. Default is `True`.
        - `context` (list[Invokable] | None): list of Invokables whose state should be treated as context for this invokation.
        - `stream` (bool): stream the output of the agent character-by-character. Default is `False`.
        - `stdout_printer` (`output.printer.Printer` | None): Printer object to handle stdout messages. Default is `None`.

        returns:
        - `invokabable.types.ExecutorOutput`, which is a dictionary with three keys:
          - `task_id` (str | None)
          - `messages` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`] | list[`BHStateElt`])
          - `printer` (`output.printer.Printer`)
        """
        raise NotImplementedError()

    def _stream(self, task: str | None = None) -> Any:
        raise NotImplementedError()


class Agent(Invokable):
    """Agents are invokables that execute complex tasks by combining planning, memory,
    and tool usage. You will never need to instantiate this class directly — you should
    always use one of the more complete child classes: `BeehiveAgent` or
    `BeehiveLangchainAgent`.

    args:
    - `name` (str): the invokable name.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel` | `BaseChatModel`): chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`.
    - `state` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `temperature` (int): temperature setting for the model.
    - `tools` (list[Callable[..., Any]]): functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
    - `docstring_format` (`DocstringFormat` | None): docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
    - `history` (bool): whether to use previous interactions / messages when responding to the current task. Default is `False`.
    - `history_lookback` (int): number of days worth of previous messages to use for answering the current task.
    - `feedback` (bool): whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used.
    - `feedback_embedder` (`BHEmbeddingModel` | None): embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`.
    - `feedback_model` (`BHChatModel` | `BaseChatModel`): language model used to generate feedback for the invokable. If `None`, then default to the `model` attribute.
    - `feedback_embedding_distance` (`EmbeddingDistance`): distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function.
    - `n_feedback_results` (int): amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`.
    - `color` (str): color used to represent the invokable in verbose printing. This can be a HEX code, an RGB code, or a standard color supported by the Rich API. See https://rich.readthedocs.io/en/stable/appendix/colors.html for more details. Default is `chartreuse2`.

    raises:
    - `pydantic_core.ValidationError`
    """

    # The `name`, `backstory`, `state` fields are defined by the Invokable model.
    temperature: int = Field(
        default=0,
        description="Temperature setting for the `Model` object's underlying LLM.",
    )
    tools: list[Callable[..., Any]] = Field(
        default_factory=list,
        description=(
            "Functions that this agent can use to answer questions. These functions are"
            " converted to tools that can be intepreted and executed by LLMs. Note that"
            " the language model must support tool calling for these tools to be properly"
            " invoked."
        ),
    )
    docstring_format: DocstringFormat | None = Field(
        default=None,
        description=(
            "Docstring format. If `None`, then Beehive will autodetect the docstring"
            " format and parse the arg descriptions."
        ),
    )

    _system_message: Any = PrivateAttr()
    _tools_map: dict[str, Any] = PrivateAttr(default_factory=dict)
    _tools_serialized: dict[str, dict[str, str | dict[str, Any]]] = PrivateAttr(
        default_factory=dict
    )

    def __str__(self):
        classname = self.__class__.__name__
        return f"{classname}(name='{self.name}', model={self.model}, tools=[{', '.join(list(self._tools_map.keys()))}])"

    def _set_initial_conversation(self):
        self.state = [self._system_message]

        # Add history
        if self.history:
            self._history_messages = self.grab_history_for_invokable_execution()
            self.state += self._history_messages

    def reset_conversation(self):
        self._set_initial_conversation()
