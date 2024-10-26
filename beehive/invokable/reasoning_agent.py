import json
import logging
import traceback
from pathlib import Path
from typing import Literal

from jinja2 import Environment, meta
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from pydantic_core import ValidationError

from beehive.invokable.base import Agent, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.types import AnyBHMessageSequence, ExecutorOutput
from beehive.invokable.utils import _construct_bh_tools_map, _process_json_output
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.base import BHChatModel
from beehive.prompts import (
    REASONING_PROMPT,
    BHPrompt,
    ConciseContextPrompt,
    FullContextPrompt,
    ModelErrorPrompt,
)
from beehive.tools.base import BHTool
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


class BeehiveReasoningAgent(Agent):
    """BeehiveReasoningAgents are invokables that execute complex tasks by combining
    memory and tool usage. Internally, ReasoningAgents use several iterations to
    examine the question, plan their response, reflect on their previous messages, and
    produce a final answer.

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
    - `minimum_step_count` (int): Minimum number of steps the agent must complete before returning the final answer. Default is `5`.
    - `reasoning_prompt` (string | Path): Reasoning prompt to use. This should be either the prompt itself or the path to a `.txt` file containing the prompt. Look at the `reasoning_prompt` section for more details. The default is Beehive's internal `COTReflectionPrompt`.
    - `step_output_model` (type[BaseModel]): Pydantic `BaseModel` to use to parse the output of a single step. See the `step_output_model` section for more information.
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

    reasoning_prompt:
        The `reasoning_prompt` template must have input variables:
            - backstory: the agent's backstory
            - tools: a list of tools and their descriptions
            - step_output_schema: JSON output schema for each step in the thinking process. will be filled in with the inputted `step_output_schema` model's schema.
            - step_budget: number of steps that the agent can take

    step_output_model:
        The `step_output_model` must contain the following fields:
            - content (str): the agent's response
            - action (str): one of the available tools
            - next_action (str): next action to take, one of the options should be `final answer`
            - step_number (int): 1
            - remaining_step_budget (str): 9

            Here is an example:
            ```python
            class StepOutput(BaseModel):
                title: str
                content: str
                action: Literal["tool1", "tool2"] | None
                next_action: Literal["continue", "reflect", "final_answer"]
                confidence: float
                step_number: int
                remaining_step_budget: int
            ```

            This model will be used to populate the `step_output_schema` variable
            in your prompt template, e.g.,
            ```python
            schema = dict(StepOutput.model_json_schema().items())
            ```
            ```
    """

    step_budget: int = Field(
        description="Initial step budget. Default is `10`",
        default=10,
    )
    minimum_step_count: int = Field(
        description="Minimum number of steps the agent must complete before returning the final answer. Default is `5`.",
        default=5,
    )
    reasoning_prompt: str | Path = Field(
        description="Jinja prompt template as a string or path to a `.txt` file. See the `prompt` section of the docstring for more information, or check out the documentation here: TODO",
        default=REASONING_PROMPT,
    )
    step_output_model: type[BaseModel] | None = Field(
        description="Output schema for processing the result of a single step. See the `prompt` section of the docstring for more information, or check out the documentation here: TODO",
        default=None,
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
    def set_private_attrs(self) -> "BeehiveReasoningAgent":
        if self.chat_loop > 1:
            logger.warning(
                "Ignoring `chat_loop` parameter and using the `step_budget` value to control the executor loop."
            )
            self.chat_loop = 1

        self._tools_map, self._tools_serialized = _construct_bh_tools_map(
            self.tools, self.docstring_format
        )

        # Check variables inside the Jinja2 template
        env = Environment()
        if isinstance(self.reasoning_prompt, Path):
            with open(self.reasoning_prompt) as f:
                ast = env.parse(f.read())
        else:
            ast = env.parse(self.reasoning_prompt)
        all_template_variables: set[str] = meta.find_undeclared_variables(ast)
        for required_var in ["backstory", "tools", "step_output_schema", "step_budget"]:
            if required_var not in all_template_variables:
                raise ValueError(
                    f"Variable `{required_var}` not found in ReasoningAgent's prompt template."
                )

        # Check schema for pydantic model
        next_step_options: list[str]
        if self.step_output_model:
            model_fields = self.step_output_model.model_fields
            for required_field in [
                "content",
                "action",
                "next_action",
                "step_number",
                "remaining_step_budget",
            ]:
                if required_field not in list(model_fields.keys()):
                    raise ValueError(
                        f"Field `{required_field} missing from {self.step_output_model.__name__} definition."
                    )

            # One of the options for `next_action` should be "final_answer". Note that
            # `next_action` could be an Enum or a Literal, so we need to check for both
            # cases.
            final_answer_schema = self.step_output_model.schema()["properties"][
                "next_action"
            ]

            # Handle Literal
            if "enum" in list(final_answer_schema.keys()):
                if "final_answer" not in final_answer_schema["enum"]:
                    raise ValueError(
                        f"{self.step_output_model.__name__}'s `next_action` field should have option 'final_answer'."
                    )

            # Handle enum
            if "$ref" in list(final_answer_schema.keys()):
                enum_class_name = final_answer_schema["$ref"].split("/")[-1]
                defs = self.step_output_model.schema().get("$defs", {})
                if defs == {}:
                    raise ValueError(
                        "Invalid Pydantic model schema. Found `$ref`, but no `$defs`."
                    )
                enum_class_def = defs.get(enum_class_name, {})
                if enum_class_def == {}:
                    raise ValueError(
                        f"Could not find enum class {enum_class_name} in model schema's $defs."
                    )
                next_step_options = enum_class_def.get("enum", [])
                if not next_step_options:
                    raise ValueError(
                        f"Could not options for enum class {enum_class_name}."
                    )
                if "final_answer" not in next_step_options:
                    raise ValueError(
                        f"Enum class {enum_class_name} should have option 'final_answer'."
                    )

        else:
            next_step_options = ["continue", "reflect", "final_answer"]

            # Define the step output BaseModel
            class StepOutput(BaseModel):
                title: str
                content: str
                action: Literal[tuple(list(self._tools_map.keys()))] | None  # type: ignore
                next_action: Literal[tuple(next_step_options)]  # type: ignore
                confidence: float
                step_number: int
                remaining_step_budget: int

                @field_validator("confidence")
                @classmethod
                def confidence_between_0_and_1(cls, confidence: float) -> float:
                    if confidence < 0 or confidence > 1:
                        raise ValueError("Confidence must be between 0 and 1.")
                    return confidence

            self.step_output_model = StepOutput

        # Tool names and descriptions
        tool_names_descriptions = []
        for name, _ in self._tools_map.items():
            func = self._tools_serialized[name]["function"]
            if isinstance(func, dict):
                description = func["description"]
                tool_names_descriptions.append(f"- {name}: {description}")

        # Set system message
        self.set_system_message(
            BHPrompt(  # type: ignore
                template=self.reasoning_prompt,
                **{
                    "backstory": self.backstory,
                    "tools": "\n".join(tool_names_descriptions),
                    "step_budget": str(self.step_budget),
                    "step_output_schema": self.step_output_model.schema(),
                },
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
        if self.step_output_model is None:
            raise ValueError("`step_output_model` is None.")

        # We usually check that the `chat_loop` attribute is less than the
        # `retry_limit`. However, we ignore the `chat_loop` attribute in this agent.
        printer = stdout_printer if stdout_printer else Printer()

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

        # Keep track of the previous step action
        current_action: str | None = None

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
                if step_message.content != "":
                    step = self.step_output_model(
                        **json.loads(_process_json_output(step_message.content))
                    )

                    # Check if the LLM has posted its final answer
                    if current_action == "final_answer":
                        # Check if minimum step count has been reached
                        if step_count < self.minimum_step_count:
                            iter_messages.append(
                                BHMessage(
                                    role=MessageRole.USER,
                                    content=f"You arrived at the final answer in {step_count} steps. Take {self.minimum_step_count - step_count} additional steps to refine your answer.",
                                )
                            )
                        else:
                            # Print the final answer and exit the loop
                            if verbose:
                                printer.print_standard(f"Final answer: {step.content}")  # type: ignore
                            break

                    # Otherwise, print the intermediate step and continue
                    else:
                        printer._console.print_json(step.model_dump_json())

                    self.state.extend(iter_messages)
                    all_messages.extend(iter_messages)

                    # Increment the step count
                    step_count = step.step_number  # type: ignore
                    current_action = step.next_action  # type: ignore

                # Print content of any remaining messages, e.g., from tool calls.
                if len(iter_messages) > 1:
                    printer.print_invokable_output(iter_messages[1:])

            # If there is a JSON decoder error, then the agent likely returned two steps
            # together.
            except (json.decoder.JSONDecodeError, ValidationError):
                total_count += 1
                if pass_back_model_errors:
                    additional_system_message = BHMessage(
                        role=MessageRole.SYSTEM,
                        content=ModelErrorPrompt(
                            error=f"Encountered a JSONDecodeError with the following content: <content>{step_message.content}</content>. Review the original instructions and make sure your output adheres to all of the requirements. Do not make this same mistake again."
                        ).render(),
                    )
                    self.state.append(additional_system_message)
                else:
                    raise

            # For all other exceptions, just use the traceback.
            except Exception:
                total_count += 1
                if pass_back_model_errors:
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
