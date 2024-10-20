import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from beehive.invokable.base import Agent, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.types import AnyBHMessageSequence, ExecutorOutput
from beehive.invokable.utils import _construct_bh_tools_map, _process_json_output
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.base import BHChatModel
from beehive.prompts import (
    ConciseContextPrompt,
    COTReflectionPrompt,
    FullContextPrompt,
    ModelErrorPrompt,
)
from beehive.tools.base import BHTool
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


class BeehiveCOTReflectionAgent(Agent):
    """BeehiveCOTReflectionAgent are invokables that execute complex tasks by combining
    memory and tool usage. Internally, BeehiveCOTReflectionAgents use reasoning chains
    with extended self-reflection to improve output accuracy.

    args:
    - `name` (str): the invokable name.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel`): chat model used by the invokable to execute its function.
    - `chat_loop` (int): number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns may require more loops (e.g., chain-of-thought prompting).
    - `state` (list[`BHMessage` | `BHToolMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `temperature` (int): temperature setting for the model.
    - `tools` (list[Callable[..., Any]]): functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
    - `docstring_format` (`DocstringFormat` | None): docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
    - `step_budget` (int): Initial step budget. Default is `10`.
    - `enable_step_budget_requests` (boolean): Enable the agent to request additional steps to add to the budget for complex problems. USE THIS WITH CAUTION — without well-crafted prompts, this can dramatically increase runtime and token usage. Default is `False`.
    - `minimum_step_count` (int): Minimum number of steps the agent must complete before returning the final answer. Default is `5`.
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

    step_budget: int = Field(
        description="Initial step budget. Default is `10`",
        default=10,
    )
    enable_step_budget_requests: bool = Field(
        description="Enable the agent to request additional steps to add to the budget for complex problems. USE THIS WITH CAUTION — without well-crafted prompts, this can dramatically increase runtime and token usage. Default is `False`.",
        default=False,
    )
    minimum_step_count: int = Field(
        description="Minimum number of steps the agent must complete before returning the final answer. Default is `5`.",
        default=5,
    )

    # Updated types
    model: BHChatModel = Field(description="Language model used to run the agent.")
    state: list[BHMessage | BHToolMessage] = Field(
        default_factory=list,
        description=(
            "List of messages that this actor has seen. This enables the actor to build"
            " off of previous conversations / outputs."
        ),
    )
    _system_message: BHMessage = PrivateAttr()
    _tools_map: dict[str, BHTool] = PrivateAttr(default_factory=dict)

    def grab_history_for_invokable_execution(self) -> list[BHMessage | BHToolMessage]:
        return super().grab_history_for_invokable_execution()

    @model_validator(mode="after")
    def set_private_attrs(self) -> "BeehiveCOTReflectionAgent":
        if self.chat_loop > 1:
            logger.warning(
                "Ignoring `chat_loop` parameter and using the `step_budget` value to control the executor loop."
            )
            self.chat_loop = 1

        self._tools_map, self._tools_serialized = _construct_bh_tools_map(
            self.tools, self.docstring_format
        )

        # Set system message
        self.set_system_message(
            COTReflectionPrompt(
                backstory=self.backstory,
                step_budget=str(self.step_budget),
                enable_step_budget_requests=self.enable_step_budget_requests,
                tool_names=", ".join(
                    ['"' + name + '"' for name in list(self._tools_map.keys())]
                ),
            ).render()
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
        # We usually check that the `chat_loop` attribute is less than the
        # `retry_limit`. However, we ignore the `chat_loop` attribute in this agent.
        printer = stdout_printer if stdout_printer else Printer()

        # Define the step output BaseModel
        valid_next_actions = ["continue", "reflect", "final_answer"]
        if self.enable_step_budget_requests:
            valid_next_actions.append("request_more_steps")

        class StepOutput(BaseModel):
            title: str
            content: str
            action: Literal[tuple(list(self._tools_map.keys()))] | None  # type: ignore
            next_action: Literal[tuple(valid_next_actions)]  # type: ignore
            confidence: float
            step_number: int
            remaining_step_budget: int

            @field_validator("confidence")
            @classmethod
            def confidence_between_0_and_1(cls, confidence: float) -> float:
                if confidence < 0 or confidence > 1:
                    raise ValueError("Confidence must be between 0 and 1.")
                return confidence

        # Define task message
        task_message = (
            task
            if isinstance(task, BHMessage)
            else BHMessage(role=MessageRole.USER, content=task)
        )
        self.state.append(task_message)

        # Keep track of number of loops. If total the number of iterations exceeds our
        # retry limit, then break.
        step_count = 1
        total_count = 0
        all_messages: list[BHMessage | BHToolMessage] = []
        while True:
            if total_count > retry_limit:
                break
            total_count += 1

            # If we have exceeded our step budget, break
            if step_count > self.step_budget:
                break

            # Chat with model. Wrap this in a try-except block so that we can pass the
            # errors back to the model, if needed.
            try:
                if not stream:
                    iter_messages = self.model.chat(
                        task_message if step_count == 1 else None,
                        self.temperature,
                        self._tools_map,
                        self.state,
                    )
                else:
                    iter_messages = self.model.stream(
                        task_message if step_count == 1 else None,
                        self.temperature,
                        self._tools_map,
                        self.state,
                        printer,
                    )

                # Determine the next action to take. The step message should be the
                # first message in the output. Subsequent messages will be tool calls
                # and tool messages.
                step_message = iter_messages[0]
                if not isinstance(step_message, BHMessage):
                    raise ValueError(
                        f"Unrecognized message class `{step_message.__class__.__name__}`."
                    )
                step = StepOutput(
                    **json.loads(_process_json_output(step_message.content))
                )
                if step.next_action == "final_answer":
                    if step_count < self.minimum_step_count:
                        printer.print_standard(
                            "Final answer received before achieving the minimum step count. Manually setting the next step to 'continue'."
                        )
                        step.action = "continue"

                        # Replace the the first message. We do this it via a new
                        # variable for mypy.
                        new_first_message: list[BHMessage | BHToolMessage] = [
                            BHMessage(
                                role=MessageRole.ASSISTANT,
                                content=step.model_dump_json(),
                            )
                        ]
                        iter_messages = new_first_message + iter_messages[1:]
                    else:
                        # Print the final answer and exit the loop
                        if verbose:
                            printer.print_standard(f"Final answer: {step.content}")
                        break

                # Otherwise, if the LLM is requesting more steps, then add that to the
                # budget.
                elif step.action == "request_more_steps":
                    self.step_budget += int(step.content)

                # Otherwise, print the intermediate step and continue
                else:
                    printer._console.print_json(step.model_dump_json())

                self.state.extend(iter_messages)
                all_messages.extend(iter_messages)

                # Increment the step count
                step_count += 1
            except Exception as e:
                total_count += 1
                if pass_back_model_errors:
                    additional_system_message = BHMessage(
                        role=MessageRole.SYSTEM,
                        content=ModelErrorPrompt(error=str(e)).render(),
                    )
                    self.state.append(additional_system_message)
                else:
                    print(
                        f"Encountered an issue with when prompting {self.name}: {str(e)}"
                    )
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
