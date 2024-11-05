import logging
import traceback
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import default_tool_parser
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from beehive.invokable.base import Agent, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.types import ExecutorOutput
from beehive.message import BHMessage
from beehive.mixins.langchain import LangchainMixin
from beehive.prompts import ConciseContextPrompt, FullContextPrompt, ModelErrorPrompt
from beehive.tools.base import create_parser
from beehive.tools.types import FunctionSpec
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


class BeehiveLangchainAgent(Agent, LangchainMixin):
    """BeehiveAgents are invokables that execute complex tasks by combining planning, memory,
    and tool usage.

    args:
    - `name` (str): the invokable name.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BaseChatModel`): chat model used by the invokable to execute its function.
    - `state` (list[`BaseMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `temperature` (int): temperature setting for the model.
    - `tools` (list[Callable[..., Any]]): functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
    - `config` (`RunnableConfig` | None): langchain runnable configuration. This is used inside the ChatModel's `invoke` method. Default is `None`.
    - `stop` (list[str]): list of strings on which the model should stop generating.
    - `docstring_format` (`DocstringFormat` | None): docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
    - `history` (bool): whether to use previous interactions / messages when responding to the current task. Default is `False`.
    - `history_lookback` (int): number of days worth of previous messages to use for answering the current task.
    - `feedback` (bool): whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used.
    - `feedback_embedder` (`BHEmbeddingModel` | None): embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`.
    - `feedback_model` (`BHChatModel` | `BaseChatModel`): language model used to generate feedback for the invokable. If `None`, then default to the `model` attribute.
    - `feedback_embedding_distance` (`EmbeddingDistance`): distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function.
    - `n_feedback_results` (int): amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`.
    - `color` (str): color used to represent the invokable in verbose printing. This can be a HEX code, an RGB code, or a standard color supported by the Rich API. See https://rich.readthedocs.io/en/stable/appendix/colors.html for more details. Default is `chartreuse2`.
    - `**model_kwargs`: extra keyword arguments for invoking the Langchain chat model.

    raises:
    - `pydantic_core.ValidationError`
    """

    model_config = ConfigDict(extra="allow")

    # Updated types
    model: Any = Field(description="Language model used to run the agent.")
    state: list[BaseMessage] = Field(
        default_factory=list,
        description=(
            "List of messages that this actor has seen. This enables the actor to build"
            " off of previous conversations / outputs."
        ),
    )

    # Langchain-specific fields
    config: RunnableConfig | None = Field(
        description=(
            "Langchain runnable configuration. This is used inside the ChatModel's"
            " `invoke` method. Default is None"
        ),
        default=None,
    )
    stop: list[str] | None = Field(
        description="A list of strings on which the model should stop generating.",
        default=None,
    )
    _system_message: SystemMessage = PrivateAttr()
    _tools_map: dict[str, BaseTool | StructuredTool] = PrivateAttr(default_factory=dict)

    @field_validator("model")
    @classmethod
    def model_must_be_langchain_chatmodel(cls, v):
        if not isinstance(v, BaseChatModel):
            raise ValueError(
                (
                    f"LangchainAgents do not support `{v.__class__.__name__}` models."
                    " Must use a model that inherits the `BaseChatModel` interface."
                    " See https://python.langchain.com/v0.1/docs/modules/model_io/chat/custom_chat_model/ for more information."
                )
            )
        return v

    @field_validator("tools")
    def tools_must_be_functions(cls, v):
        for x in v:
            if isinstance(x, BaseTool):
                raise ValueError(
                    (
                        "The `tools` keyword argument must be a list of functions."
                        " Beehive calls the Langchain API to convert these functions into Langchain"
                        " tools and bind these to the LLM"
                    )
                )
        return v

    @model_validator(mode="after")
    def define_llm_bindings_and_set_private_attrs(self) -> "BeehiveLangchainAgent":
        if self.tools:
            specs: list[FunctionSpec] = []
            for x in self.tools:
                specs.append(create_parser(x, self.docstring_format).parse())

            for s, f in zip(specs, self.tools):
                # Langchain doesn't currently accept dynamic models, so don't include
                # the `args_schema` in the instantiation. Instead, define the class
                # attribute after instantiating the object.
                tool = StructuredTool.from_function(
                    func=f,
                    description=s.description,
                )
                tool.args_schema = s.params  # type: ignore
                self._tools_serialized[s.name] = s.serialize()
                self._tools_map[s.name] = tool
            self.model = self.model.bind_tools(list(self._tools_map.values()))
        self.set_system_message(self.backstory)
        return self

    def set_system_message(self, system_message: str):
        self._system_message = SystemMessage(
            content=system_message,
        )
        self._set_initial_conversation()

    def grab_history_for_invokable_execution(self) -> list[BaseMessage]:
        messages = super().grab_history_for_invokable_execution()
        return [self._convert_beehive_message_to_langchain_message(m) for m in messages]

    def _invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[BaseMessage]:
        printer = stdout_printer if stdout_printer else Printer()

        # Make sure the state is composed of Langchain messages
        self.state = self._convert_state_to_langchain_base_message(self.state)

        # Define task message
        task_message = (
            HumanMessage(content=task)
            if isinstance(task, str)
            else HumanMessage(content=task.content)
        )
        self.state.append(task_message)

        # Invoke the chat model. Keep track of number of loops
        total_count = 0
        all_messages: list[BaseMessage] = []
        while True:
            if total_count > retry_limit:
                break
            total_count += 1

            try:
                iter_messages: BaseMessage | list[BaseMessage] = self.model.invoke(
                    self.state, config=self.config, stop=self.stop, **self.model_extra
                )
                if not isinstance(iter_messages, list):
                    iter_messages = [iter_messages]

                # If the model wants to invoke tool(s), parse the tool calls and create
                # separate messages for them.
                for msg in iter_messages:
                    additional_kwargs = msg.additional_kwargs
                    if "tool_calls" in additional_kwargs:
                        valid_tool_calls, _ = default_tool_parser(
                            additional_kwargs["tool_calls"]
                        )
                        for tc in valid_tool_calls:
                            _tool = self._tools_map[tc["name"]]
                            iter_messages.append(
                                ToolMessage(
                                    content=_tool.invoke(tc["args"], self.config),
                                    tool_call_id=tc["id"],
                                )
                            )
                self.state.extend(iter_messages)
                all_messages.extend(iter_messages)
                break
            except Exception:
                total_count += 1
                if total_count > retry_limit:
                    printer.print_standard(
                        "[red]ERROR:[/red] Exceeded total retry limit."
                    )
                    raise
                elif pass_back_model_errors:
                    additional_system_message = SystemMessage(
                        content=ModelErrorPrompt(
                            error=str(traceback.format_exc())
                        ).render(),
                    )
                    self.state.append(additional_system_message)
                else:
                    raise

        # Print
        if verbose:
            if all_messages:
                printer.print_invokable_output(
                    completion_messages=all_messages,
                )
        return all_messages

    def invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        context: list[Invokable] | None = None,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> ExecutorOutput:
        """Invoke the Agent to execute a task.

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
          - `messages` (list[`BaseMessage`])
          - `printer` (`output.printer.Printer`)

        examples:
        - See the documentation here: https://beehivehq.github.io/beehive-ai/
        """
        # Define the printer and create Panel for the invokable
        printer = stdout_printer if stdout_printer else Printer()
        if verbose:
            if not printer._all_beehives:
                printer._console.print(printer.separation_rule())
            printer._console.print(
                printer.invokable_label_text(
                    self.name,
                    self.color,
                    task.content.split("\n")[0]
                    if isinstance(task, BHMessage)
                    else task.split("\n")[0],
                )
            )

        # If the agent has already received context before, then we need to ensure that
        # we don't *repeat* context in our subsequent messages. This reduces the number
        # of tokens used in the context window while still maintaining a coherent
        # context. This is handled in our `augment_invokable_with_context` function.
        context_message = self.augment_invokable_with_context(
            invokables=context,
            context_template=ConciseContextPrompt
            if self._context_messages
            else FullContextPrompt,
        )

        # Convert context and feedback to Langchain message objects as well. Context /
        # feedback should be added before the task
        if context_message:
            lc_context = self._convert_beehive_message_to_langchain_message(
                context_message
            )
            self.state.append(lc_context)

        executor = InvokableExecutor(
            task=task.content if isinstance(task, BHMessage) else task,
            invokable=self,
        )
        exec_output = executor.execute(
            retry_limit=retry_limit,
            pass_back_model_errors=pass_back_model_errors,
            verbose=verbose,
            stream=stream,
            stdout_printer=printer,
        )
        if verbose:
            printer._console.print(printer.separation_rule())
        return exec_output
