"""Inspired by the following paper:
https://arxiv.org/pdf/2305.14325
"""
import logging
import os
from multiprocessing.pool import AsyncResult, Pool
from typing import Any, Literal

from langchain_core.messages import BaseMessage
from pydantic import ConfigDict, Field, PrivateAttr, model_validator

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Agent, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.invokable.team import AgentTeam
from beehive.invokable.types import (
    AnyBHMessageSequence,
    AnyMessage,
    AnyMessageSequence,
    BHStateElt,
    ExecutorOutput,
)
from beehive.message import BHMessage, MessageRole
from beehive.models.base import BHChatModel
from beehive.output.printer import Printer
from beehive.prompts import (
    BHPrompt,
    ConciseContextPrompt,
    DebateJudgePrompt,
    DebateJudgeSummaryPrompt,
    FullContextPrompt,
    LongFormDebatePrompt,
    ShortFormDebatePrompt,
)
from tools.base import BHTool

logger = logging.getLogger(__file__)


class BeehiveDebateTeam(AgentTeam):
    """Multi-agent debate, inspired by Du et. al (2023):

    https://arxiv.org/pdf/2305.14325

    In an debate, `n` agents are initially given the same task and produce `n` different
    responses. The agents then "debate" with one another, i.e., they look at the output
    of the other `n-1` agents and update their own response. This happens over several
    rounds. Finally, a "judge" (another LLM agent) evaluates all of the responses and
    chooses the one answer the initial query best.

    args:
    - `num_members` (int): number of members on the team.
    - `name` (str): the invokable name. Team members will be given the name `{name}-{i}`, where `i` is any number between 1 and `num_members`.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel` | `BaseChatModel`): chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`.
    - `chat_loop` (int): number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns may require more loops (e.g., chain-of-thought prompting).
    - `state` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `temperature` (int): temperature setting for the model.
    - `tools` (list[Callable[..., Any]]): functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
    - `docstring_format` (`DocstringFormat` | None): docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
    - `num_rounds` (int): number of debate rounds.
    - `debate_length` (Literal['short', 'long']): length of debate rounds. This argument affects how we prompt an individual agent to trust its own outputs over those generated by other models. Short-form debates encourage agents to be more agreeable to other agent responses, whereas long-form debates encourage agents to be more stubborn. Long-form debates tend to produce better outputs, but also take longer and use more resources.
    - `judge_model` (`BHChatModel` | `BaseChatModel`): model used to synthesize responses from agents and generate a final response.
    - `history` (bool): whether to use previous interactions / messages when responding to the current task. Default is `False`.
    - `history_lookback` (int): number of days worth of previous messages to use for answering the current task.
    - `feedback` (bool): whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used.
    - `feedback_embedder` (`BHEmbeddingModel` | None): embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`.
    - `feedback_model` (`BHChatModel` | `BaseChatModel`): language model used to generate feedback for the invokable. If `None`, then default to the `model` attribute.
    - `feedback_embedding_distance` (`EmbeddingDistance`): distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function.
    - `n_feedback_results` (int): amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`.
    - `color` (str): color used to represent the invokable in verbose printing. This can be a HEX code, an RGB code, or a standard color supported by the Rich API. See https://rich.readthedocs.io/en/stable/appendix/colors.html for more details. Default is `chartreuse2`.
    - `**agent_kwargs`: extra keyword arguments for agent instantiation. This is ONLY used for Langchain agents, and this is used for both the member agent and synthesizer agent instantiation.

    raises:
    - `pydantic_core.ValidationError`
    """

    model_config = ConfigDict(extra="allow")

    num_rounds: int = Field(description="Number of debate rounds.")
    debate_length: Literal["short", "long"] = Field(
        default="short",
        description=(
            "Length of debate rounds. This argument affects how we prompt an individual"
            " agent to trust its own outputs over those generated by other models."
            " Short-form debates encourage agents to be more agreeable to other agent"
            " responses, whereas long-form debates encourage agents to be more stubborn."
            " Long-form debates tend to produce better outputs, but also take longer"
            " and use more resources."
        ),
    )
    judge_model: Any = Field(
        description=(
            "Model used to synthesize responses from agents and generate a final"
            " response."
        ),
    )

    _tools_map: dict[str, BHTool] = PrivateAttr(default_factory=dict)
    _tools_serialized: dict[str, dict[str, str | dict[str, Any]]] = PrivateAttr(
        default_factory=dict
    )
    _judge_agent: Agent | None = PrivateAttr()
    _compatible_with_memory: bool = True

    @model_validator(mode="after")
    def create_judge_agent(self) -> "BeehiveDebateTeam":
        # Judge agent. This takes the outputs from all the agents in _agent_list and
        # synthesizes them into one output that answers the user's original question.
        if self.judge_model.__class__.__name__ != self.model.__class__.__name__:
            raise ValueError(
                "`model` and `judge_model` class must match! Found `{cls1}` and `{cls2}`, respectively.".format(
                    cls1=self.model.__class__.__name__,
                    cls2=self.judge_model.__class__.__name__,
                )
            )

        if isinstance(self.judge_model, BHChatModel):
            self._judge_agent = BeehiveAgent(
                name=f"{self.name}-judge",
                backstory=self.backstory,
                model=self.judge_model,
                tools=[],  # the judge agent doesn't need tools
                state=self.state,
                history=self.history,
                history_lookback=self.history_lookback,
                feedback=self.feedback,
                feedback_embedder=self.feedback_embedder,
                feedback_model=self.feedback_model,
                feedback_embedding_distance=self.feedback_embedding_distance,
                n_feedback_results=self.n_feedback_results,
                color=self.color,
            )
        else:
            extra_kw: dict[str, Any] = self.model_extra if self.model_extra else {}
            self._judge_agent = BeehiveLangchainAgent(
                name=self.name,
                backstory=self.backstory,
                model=self.judge_model,
                tools=[],  # the judge agent doesn't need tools
                state=self.state,
                history=self.history,
                history_lookback=self.history_lookback,
                feedback=self.feedback,
                feedback_embedder=self.feedback_embedder,
                feedback_model=self.feedback_model,
                feedback_embedding_distance=self.feedback_embedding_distance,
                n_feedback_results=self.n_feedback_results,
                color=self.color,
                **extra_kw,
            )
        return self

    @property
    def debaters(self):
        return self._agent_list

    def _prepare_debate_prompt(
        self,
        task: str,
        current_agent_idx: int | None,
        agent_messages: dict[int, AnyMessageSequence],
    ) -> str:
        if self.debate_length == "short":
            return ShortFormDebatePrompt(
                other_agent_responses=self._convert_agent_responses_to_string(
                    agent_messages, current_agent_idx
                ),
                task=task,
            ).render()
        else:
            return LongFormDebatePrompt(
                other_agent_responses=self._convert_agent_responses_to_string(
                    agent_messages, current_agent_idx
                ),
                task=task,
            ).render()

    def _prepare_judge_prompt(
        self, task: str, agent_messages: dict[int, AnyMessageSequence]
    ):
        return DebateJudgePrompt(
            num_agents=str(self.num_members),
            task=task,
            responses=self._convert_agent_responses_to_string(agent_messages),
        ).render()

    def _invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> AnyMessageSequence:
        if not self._judge_agent:
            raise ValueError("`_judge_agent` attribute is not specified!")

        printer = stdout_printer if stdout_printer else Printer()
        agent_messages, agent_feedback = self._invoke_agents_in_process(
            task=task,
            printer=printer,
            retry_limit=retry_limit,
            pass_back_model_errors=pass_back_model_errors,
        )
        # Add to the agents' state
        for i, agent in enumerate(self._agent_list, start=1):
            agent.state.append(
                task
                if isinstance(task, BHMessage)
                else BHMessage(role=MessageRole.USER, content=task)
            )
            agent.state.extend(agent_messages[i])
        self._agent_feedback = agent_feedback

        def callback(result: list[AnyMessage]):
            if not result:
                printer.print_standard(
                    "Unable to invoke chat model for one of the debaters."
                )
            return

        # Start debate rounds — this should also be executed via a Pool of workers
        for _ in range(self.num_rounds):
            with Pool(processes=min(os.cpu_count(), len(self._agent_list), 1)) as pool:  # type: ignore
                results: dict[int, AsyncResult[Any]] = {}
                for i, inv in enumerate(self._agent_list, start=1):
                    new_prompt = self._prepare_debate_prompt(
                        task=task.content if isinstance(task, BHMessage) else task,
                        current_agent_idx=i,
                        agent_messages=agent_messages,
                    )

                    # Copy the state before creating the async task. For whatever
                    # reason, this make it easier to test the arguments passed to the
                    # async function in our testing.
                    state = inv.state.copy()
                    res = self._create_async_task(
                        pool=pool,
                        callback=callback,
                        invokable=inv,
                        state=state,
                        task=new_prompt,
                        retry_limit=retry_limit,
                        pass_back_model_errors=pass_back_model_errors,
                    )
                    results[i] = res

                    # Add prompt to the invokable state
                    inv.state.append(
                        BHMessage(role=MessageRole.USER, content=new_prompt)
                    )

                # Wait for all the processes to finish
                all_ready = all([r.ready() for _, r in results.items()])
                while not all_ready:
                    all_ready = all([r.ready() for _, r in results.items()])

                # Store messages
                for i, r in results.items():
                    r_output = r.get()
                    agent_messages[i] = r_output[0]
                    self._agent_list[i - 1].state.extend(agent_messages[i])

                pool.close()
                pool.join()

        # Use the judge agent to analyze the debater responses and converge to a final
        # answer.
        judge_prompt = self._prepare_judge_prompt(
            task.content if isinstance(task, BHMessage) else task, agent_messages
        )
        judge_messages = self._judge_agent._invoke(
            task=judge_prompt,
            retry_limit=retry_limit,
            pass_back_model_errors=pass_back_model_errors,
            verbose=verbose,
            stream=stream,
            stdout_printer=printer,
        )

        # Empty list for return
        final_messages: AnyMessageSequence = []
        final_bh_messages: AnyBHMessageSequence = []
        final_lc_messages: list[BaseMessage] = []

        # Type checking
        for output_elt in judge_messages:
            assert not isinstance(output_elt, BHStateElt)
            if isinstance(output_elt, BaseMessage):
                final_lc_messages.append(output_elt)
            else:
                final_bh_messages.append(output_elt)

        # Only one of the `final_` messages should be non-empty
        if final_bh_messages and final_lc_messages:
            raise ValueError("Judge agent returned incompatible message classes.")

        # mypy complains here, and I'm not entirely sure why.
        final_messages.extend(final_bh_messages)  # type: ignore
        final_messages.extend(final_lc_messages)  # type: ignore

        # Update the judge's state
        self._judge_agent.state.append(
            BHMessage(role=MessageRole.USER, content=judge_prompt)
        )
        self._judge_agent.state.extend(final_messages)

        # Finally, add a summary message to the member agents' states
        summary_message = DebateJudgeSummaryPrompt(
            num_agents=str(self.num_members),
            num_rounds=str(self.num_rounds),
            task=task.content if isinstance(task, BHMessage) else task,
            final_answer=" ".join(
                filter(None, [str(msg.content) for msg in final_messages])
            ),
        ).render()
        for inv in self._agent_list:
            inv.state.append(
                BHMessage(role=MessageRole.ASSISTANT, content=summary_message)
            )
        return final_messages

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
        """Start a multi-agent debate in order to execute a task. In an debate, `n`
        agents are initially given the same task and produce `n` different
        responses. The agents then "debate" with one another, i.e., they look at the
        output of the other `n-1` agents and update their own response. This happens
        over several rounds. Finally, a "judge" (another LLM agent) evaluates all of the
        responses and chooses the one answer the initial query best.

        args:
        - `task` (str): task to execute.
        - `retry_limit` (int): maximum number of retries before the Invokable returns an error. Default is `100`.
        - `pass_back_model_errors` (bool): boolean controlling whether to pass the contents of an error back to the LLM via a prompt.  Default is `False`.
        - `verbose` (bool): beautify stdout logs with the `rich` package. Default is `True`.
        - `context` (list[Invokable] | None): list of Invokables whose state should be treated as context for this invokation.
        - `stream` (bool): stream the output of the agent character-by-character. Default is `False`.
        - `stdout_printer` (`output.printer.Printer` | None): Printer object to handle stdout messages. Default is `None`.

        returns:
        - `invokabable.types.ExecutorOutput`, which is a dictionary with three keys:
          - `task_id` (str | None)
          - `messages` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`])
          - `printer` (`output.printer.Printer`)

        examples:
        - See the documentation here: [TODO]
        """
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

        # Context message
        context_template: type[BHPrompt] = (
            ConciseContextPrompt if self._context_messages else FullContextPrompt
        )
        context_message = self.augment_invokable_with_context(context, context_template)
        if context_message:
            for inv in self._agent_list:
                inv.state.append(context_message)

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
