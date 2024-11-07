import json
import os
from multiprocessing.pool import AsyncResult, Pool
from typing import Any, Callable, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Agent, Feedback, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.invokable.types import AnyChatModel, AnyMessageSequence
from beehive.invokable.utils import _construct_bh_tools_map, _convert_messages_to_string
from beehive.message import BHMessage, BHToolMessage
from beehive.models.base import BHChatModel
from beehive.tools.base import BHTool
from beehive.tools.types import DocstringFormat
from beehive.utilities.printer import Printer


def _invoke_agent_in_process(
    state: AnyMessageSequence | None,
    name: str,
    backstory: str,
    model_cls: type[AnyChatModel],
    model_kwargs: dict[str, Any],
    temperature: int,
    tools: list[Callable[..., Any]],
    task: str | BHMessage,
    response_model: type[BaseModel] | None,
    termination_condition: Callable[..., bool] | None,
    chat_loop: int,
    retry_limit: int,
    pass_back_model_errors: bool,
) -> Tuple[AnyMessageSequence, Feedback | None]:
    model = model_cls(
        **model_kwargs,
    )

    # These fields reflect the core agent's behavior. The other fields are used to
    # control how the agent handles feedback and history. That's not relevant for this
    # task, because we are simply invoking the model and retrieving the output. We
    # handle feedback and history at the BeehiveEnsemble-level.
    fake_agent: BeehiveAgent | BeehiveLangchainAgent
    if isinstance(model, BHChatModel):
        fake_agent = BeehiveAgent(
            name=name,
            backstory=backstory,
            model=model,
            temperature=temperature,
            tools=tools,
            response_model=response_model,
            termination_condition=termination_condition,
            chat_loop=chat_loop,
            feedback=True,
        )
    else:
        fake_agent = BeehiveLangchainAgent(
            name=name,
            backstory=backstory,
            model=model,
            temperature=temperature,
            tools=tools,
            feedback=True,
        )
    if state:
        fake_agent.state = state
    messages = fake_agent._invoke(
        task,
        retry_limit,
        pass_back_model_errors,
        False,
    )

    # Create an Executor so that we can record concatenate all the feedback for this
    # task across all agents in the BeehiveEnsemble.
    executor = InvokableExecutor(
        task=task,
        invokable=fake_agent,
    )
    executor._feedback = True  # manually set feedback flag
    executor._output = " ".join([str(msg.content) for msg in messages])
    task_feedback = executor.evaluate_task()
    return messages, task_feedback


class AgentTeam(Invokable):
    """Base class for representing a team of Agents. In teams, `n` agents are given the
    same task and produce `n` different responses. These responses are then synthesized
    together to produce a higher-quality final answer. You will never need to
    instantiate this class directly — you should always use one of the more complete
    child classes, e.g., BeehiveDebateTeam, BeehiveEnsemble.

    args:
    - `num_members` (int): number of members on the team.
    - `name` (str): the invokable name. Team members will be given the name `{name}-{i}`, where `i` is any number between 1 and `num_members`.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel` | `BaseChatModel`): chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`.
    - `state` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
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
    - `**agent_kwargs`: extra keyword arguments for agent instantiation. This is ONLY used for Langchain agents.

    raises:
    - `pydantic_core.ValidationError`
    """

    model_config = ConfigDict(extra="allow")

    num_members: int = Field(description="Number of agents in the team.")
    model: Any = Field(
        description=(
            "Model used by the member agents to generate `n` responses to the"
            " original task."
        )
    )
    temperature: int = Field(
        default=0,
        description="Temperature setting for the model's underlying LLM.",
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
    docstring_format: DocstringFormat | None = Field(
        default=None,
        description=(
            "Docstring format. If `None`, then Beehive will autodetect the docstring"
            " format and parse the arg descriptions."
        ),
    )

    _agent_list: list[Agent] = PrivateAttr(default_factory=list)
    _tools_map: dict[str, BHTool] = PrivateAttr(default_factory=dict)
    _tools_serialized: dict[str, dict[str, str | dict[str, Any]]] = PrivateAttr(
        default_factory=dict
    )

    # We want to concatenate all the feedback from the member agents and associate it
    # with the parent AgentTeam instance.
    _agent_feedback: dict[int, Feedback] = PrivateAttr(default_factory=dict)
    _compatible_with_memory: bool = True

    @model_validator(mode="after")
    def setup(self) -> "AgentTeam":
        # Tools attributes
        self._tools_map, self._tools_serialized = _construct_bh_tools_map(
            self.tools, self.docstring_format
        )

        # History
        self._history_messages = (
            self.grab_history_for_invokable_execution() if self.history else []
        )

        # Agent list
        self._agent_list: list[Agent] = []
        new_agent: BeehiveAgent | BeehiveLangchainAgent
        if isinstance(self.model, BHChatModel):
            for i in range(1, self.num_members + 1):
                new_agent = BeehiveAgent(
                    name=f"{self.name}-{i}",
                    backstory=self.backstory,
                    model=self.model,
                    tools=self.tools,
                    response_model=self.response_model,
                    termination_condition=self.termination_condition,
                    chat_loop=self.chat_loop,
                    state=self.state,
                    history=False,  # we treat all agents as having a single history / feedback
                    feedback=False,  # we treat all agents as having a single history / feedback
                    color=self.color,
                )
                new_agent.state.extend(self._history_messages)
                self._agent_list.append(new_agent)
        else:
            for i in range(1, self.num_members + 1):
                extra_kw: dict[str, Any] = self.model_extra if self.model_extra else {}
                new_agent = BeehiveLangchainAgent(
                    name=f"{self.name}-{i}",
                    backstory=self.backstory,
                    model=self.model,
                    tools=self.tools,
                    state=self.state,
                    history=False,  # we treat all agents as having a single history / feedback
                    feedback=False,  # we treat all agents as having a single history / feedback
                    color=self.color,
                    **extra_kw,
                )
                for hmes in self._history_messages:
                    new_agent.state.append(
                        new_agent._convert_beehive_message_to_langchain_message(hmes)
                    )
                self._agent_list.append(new_agent)
        return self

    @property
    def members(self):
        return self._agent_list

    def _convert_agent_responses_to_string(
        self,
        agent_responses: dict[int, AnyMessageSequence],
        skip_index: int | None = None,
    ) -> str:
        # Format outputs
        outputs_list: list[str] = []
        for i, msgs in agent_responses.items():
            if skip_index and i == skip_index:
                continue
            msg_content = _convert_messages_to_string(msgs)
            agent_output = f"Agent {i}:\n{msg_content}"
            outputs_list.append(agent_output)
        return "\n\n".join(outputs_list)

    def _create_async_task(
        self,
        pool: Pool,
        callback: Callable[..., Any],
        invokable: Agent,
        state: list[Any],
        task: str | BHMessage,
        retry_limit: int,
        pass_back_model_errors: bool,
    ) -> AsyncResult[Any]:
        # Confirm that the state has messages
        for msg in state:
            if not (
                isinstance(msg, BHMessage)
                or isinstance(msg, BHToolMessage)
                or isinstance(msg, BaseMessage)
            ):
                raise ValueError(
                    f"`state` keyword argument contains unknown instance of type `{msg.__class__.__name__}`"
                )

        model_kwargs = json.loads(invokable.model.json())
        # Get secret values
        if isinstance(invokable.model, BaseChatModel):
            for k, v in model_kwargs.items():
                if v == "**********":
                    model_kwargs[k] = getattr(invokable.model, k).get_secret_value()

        res = pool.apply_async(
            _invoke_agent_in_process,
            args=(
                state,
                invokable.name,
                invokable.backstory,
                invokable.model.__class__,
                model_kwargs,
                self.temperature,
                invokable.tools,
                task,
                self.response_model,
                self.termination_condition,
                self.chat_loop,
                retry_limit,
                pass_back_model_errors,
            ),
            callback=callback,
        )
        return res

    def _invoke_agents_in_process(
        self,
        task: str | BHMessage,
        printer: Printer,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
    ) -> Tuple[dict[int, AnyMessageSequence], dict[int, Feedback]]:
        agent_messages: dict[int, AnyMessageSequence] = {}
        feedback_suggestions: dict[int, Feedback] = {}

        def callback(result: AnyMessageSequence):
            if not result:
                printer.print_standard(
                    "Unable to invoke chat model for one of the BeehiveEnsemble's agents."
                )
            return

        with Pool(processes=min(os.cpu_count(), len(self._agent_list), 1)) as pool:  # type: ignore
            results: dict[int, AsyncResult[Any]] = {}
            for i, inv in enumerate(self._agent_list, start=1):
                model_kwargs = json.loads(inv.model.json())

                # Get secret values
                if isinstance(inv.model, BaseChatModel):
                    for k, v in model_kwargs.items():
                        if v == "**********":
                            model_kwargs[k] = getattr(inv.model, k).get_secret_value()

                # Copy the state before creating the async task. For whatever
                # reason, this make it easier to test the arguments passed to the async
                # function in our testing.
                state = inv.state.copy()
                res = self._create_async_task(
                    pool=pool,
                    callback=callback,
                    invokable=inv,
                    state=state,
                    task=task,
                    retry_limit=retry_limit,
                    pass_back_model_errors=pass_back_model_errors,
                )
                results[i] = res

            # Wait for all the processes to finish
            all_ready = all([r.ready() for _, r in results.items()])
            while not all_ready:
                all_ready = all([r.ready() for _, r in results.items()])

            # Store messages
            for i, r in results.items():
                r_output = r.get()
                agent_messages[i] = r_output[0]
                feedback_suggestions[i] = r_output[1]
            pool.close()
            pool.join()

        return agent_messages, feedback_suggestions
