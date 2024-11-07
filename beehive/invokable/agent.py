import inspect
import json
import logging
import traceback
from typing import Callable

from pydantic import BaseModel, Field, PrivateAttr, model_validator
from pydantic_core import ValidationError

from beehive.invokable.base import Agent, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.types import AnyBHMessageSequence, ExecutorOutput
from beehive.invokable.utils import _construct_bh_tools_map
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.base import BHChatModel
from beehive.prompts import ConciseContextPrompt, FullContextPrompt, ModelErrorPrompt
from beehive.tools.base import BHTool
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


class BeehiveAgent(Agent):
    """BeehiveAgents are invokables that execute complex tasks by combining memory and tool usage.

    args:
    - `name` (str): the invokable name.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel`): chat model used by the invokable to execute its function.
    - `state` (list[`BHMessage` | `BHToolMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `temperature` (int): temperature setting for the model.
    - `tools` (list[Callable[..., Any]]): functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
    - `response_model` (type[`BaseModel`] | None): Pydantic BaseModel defining the desired schema for the agent's output. When specified, Beehive will prompt the agent to make sure that its responses fit the models's schema. Default is `None`.
    - `termination_condition` (Callable[..., bool] | None): condition which, if met, breaks the BeehiveAgent out of the chat loop. This should be a function that takes a `response_model` instance as input. Default is `None`.
    - `chat_loop` (int): number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns (e.g., COT, reflection) may require more loops. This should always be used with a `response_model` and a `termination_condition`.
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

    # Updated types
    model: BHChatModel = Field(description="Language model used to run the agent.")
    state: list[BHMessage | BHToolMessage] = Field(
        default_factory=list,
        description=(
            "List of messages that this actor has seen. This enables the actor to build"
            " off of previous conversations / outputs."
        ),
    )

    response_model: type[BaseModel] | None = Field(
        description="Pydantic BaseModel defining the desired schema for the agent's output. When specified, Beehive will prompt the agent to make sure that its responses fit the models's schema. Default is `None`.",
        default=None,
    )
    termination_condition: Callable[..., bool] | None = Field(
        description="Condition which, if met, breaks the BeehiveAgent out of the chat loop. This should be a function that takes a `response_model` instance as input. Default is `None`.",
        default=None,
    )
    chat_loop: int = Field(
        description="Number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns (e.g., COT, reflection) may require more loops. This should always be used with a `response_model` and a `termination_condition`.",
        default=1,
    )

    _system_message: BHMessage = PrivateAttr()
    _tools_map: dict[str, BHTool] = PrivateAttr(default_factory=dict)

    def grab_history_for_invokable_execution(self) -> list[BHMessage | BHToolMessage]:
        return super().grab_history_for_invokable_execution()

    @model_validator(mode="after")
    def set_private_attrs(self) -> "BeehiveAgent":
        self._tools_map, self._tools_serialized = _construct_bh_tools_map(
            self.tools, self.docstring_format
        )
        self.set_system_message(self.backstory)

        # If the chat loop is greater than 1, than the user must specify a
        # response_model and a termination condition.
        if self.chat_loop > 1:
            if not self.response_model:
                raise ValueError(
                    "Agents that use a `chat_loop` must use a `response_model` for their output."
                )
            if not self.termination_condition:
                raise ValueError(
                    "Agents that use a `chat_loop` must use a `termination_condition` to control the loop flow."
                )

            # The termination condition is a function that takes one argument — an
            # instance of the response_model.
            spec = inspect.getfullargspec(self.termination_condition)
            if spec.kwonlyargs:
                raise ValueError(
                    "Keyword-only arguments not allowed in `termination_condition` function."
                )
            if len(spec.args) > 1:
                raise ValueError(
                    "Multiple arguments not allowed in `termination_condition` function."
                )

        return self

    def check_backstory_format(self) -> "BeehiveAgent":
        # If the user has specified a response model but does not have <schema> tags in
        # their backstory, raise a warning.
        if self.response_model and (
            "<schema>" not in self.backstory or "</schema>" not in self.backstory
        ):
            logger.warning(
                (
                    "Found `response_model`, but did not find <schema>...</schema> tags in the agent's backstory. "
                    "Without these tags, the agent's response may not be compatible with the `response_model`. "
                    "See the documentation here https://beehivehq.github.io/beehive-ai/advanced/reasoning/ for an example."
                )
            )

        return self

    def set_system_message(self, system_message: str):
        self._system_message = BHMessage(
            role=MessageRole.SYSTEM,
            content=system_message,
        )
        self._set_initial_conversation()

    def _invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> AnyBHMessageSequence:
        if retry_limit < self.chat_loop:
            raise ValueError(
                "`retry_limit` must be greater than `chat_loop` attribute."
            )
        printer = stdout_printer if stdout_printer else Printer()

        # Define task message
        task_message = (
            task
            if isinstance(task, BHMessage)
            else BHMessage(role=MessageRole.USER, content=task)
        )
        self.state.append(task_message)

        # Keep track of number of loops. If total the number of iterations exceeds our
        # recursion limit, then break. If the number of successful iterations exceeds
        # the `chat_loop` attribute, break.
        success_count = 0
        total_count = 0
        all_messages: list[BHMessage | BHToolMessage] = []
        flag_terminate: bool = False
        while success_count < self.chat_loop:
            if total_count > retry_limit:
                break
            total_count += 1

            # Chat with model. Wrap this in a try-except block so that we can pass the
            # errors back to the model, if needed.
            try:
                if not stream:
                    iter_messages = self.model.chat(
                        task_message if success_count == 0 else None,
                        self.temperature,
                        self._tools_map,
                        self.state,
                    )
                else:
                    iter_messages = self.model.stream(
                        task_message if success_count == 0 else None,
                        self.temperature,
                        self._tools_map,
                        self.state,
                        printer,
                    )

                # If the user has specified a response model, confirm that the output of
                # the first message adheres to that schema. Note that we only need to
                # look at the first message, since anything beyond that will be tool
                # calls.
                if self.response_model:
                    # Mypy
                    if not self.termination_condition:
                        raise ValueError(
                            "Agents that use a `chat_loop` must use a `termination_condition` to control the loop flow."
                        )
                    # This will trigger some sort of validation error or
                    # JSONDecodeError, which we can then use to re-prompt the agent.
                    if iter_messages[0].content:
                        response_obj = self.response_model(
                            **json.loads(iter_messages[0].content)
                        )
                        flag_terminate = self.termination_condition(response_obj)

                self.state.extend(iter_messages)
                all_messages.extend(iter_messages)
                success_count += 1

                # Print
                if verbose and not stream:
                    if iter_messages:
                        printer.print_invokable_output(
                            completion_messages=iter_messages,
                        )

                # Terminate
                if flag_terminate:
                    break

                if verbose and not stream:
                    if iter_messages and self.chat_loop > 1:
                        printer.print_standard(
                            ""
                        )  # to make sure there is a new line after each step
            except (json.decoder.JSONDecodeError, ValidationError):
                total_count += 1
                if total_count > retry_limit:
                    printer.print_standard(
                        "[red]ERROR:[/red] Exceeded total retry limit."
                    )
                    raise
                elif pass_back_model_errors:
                    if verbose and not stream:
                        printer.print_standard(
                            "[yellow]WARNING:[/yellow] Encountered a JSONDecodeError / Pydantic ValidationError. Passing the error back to the LLM and trying again.\n"
                        )
                    additional_system_message = BHMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Encountered a `JSONDecodeError` / Pydantic `ValidationError` with the following content: <content>{iter_messages[0].content}</content>. **All output must be formatted according to the JSON schema described in the instructions**. Do not make this same mistake again.",
                    )
                    self.state.append(additional_system_message)
                else:
                    raise

            # TODO - we can probably clean this up
            except Exception:
                total_count += 1
                if total_count > retry_limit:
                    printer.print_standard("ERROR: Exceeded total retry limit.")
                    raise
                elif pass_back_model_errors:
                    if verbose and not stream:
                        printer.print_standard(
                            "[yellow]WARNING:[/yellow] Encountered an Exception. Passing the error back to the LLM and trying again.\n"
                        )
                    additional_system_message = BHMessage(
                        role=MessageRole.SYSTEM,
                        content=ModelErrorPrompt(
                            error=str(traceback.format_exc())
                        ).render(),
                    )
                    self.state.append(additional_system_message)
                else:
                    raise

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
          - `messages` (list[`BHMessage` | `BHToolMessage`])
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

        # Context should be added before the task
        if context_message:
            self.state.append(context_message)

        executor = InvokableExecutor(
            task=task,
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
