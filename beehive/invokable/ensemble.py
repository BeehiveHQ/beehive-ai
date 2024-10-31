"""Inspired by the following paper:
https://arxiv.org/pdf/2402.05120
"""
import logging
from typing import Any, Callable, Literal, TypeVar

from langchain_core.messages import BaseMessage
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
    validate_call,
)

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Agent, Invokable
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.invokable.team import AgentTeam
from beehive.invokable.types import (
    AnyBHMessageSequence,
    AnyChatModel,
    AnyMessageSequence,
    ExecutorOutput,
)
from beehive.invokable.utils import _convert_messages_to_string
from beehive.message import BHMessage, MessageRole
from beehive.models.base import BHChatModel
from beehive.prompts import (
    ConciseContextPrompt,
    EnsembleFuncSummaryPrompt,
    EnsembleLLMSummaryPrompt,
    FullContextPrompt,
    SynthesizerPrompt,
)
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


AnyCallableT = TypeVar("AnyCallableT", bound=Callable[..., Any])


class BeehiveEnsemble(AgentTeam):
    """An ensemble of `n` agents, inspired by Li et. al (2024):

    https://arxiv.org/pdf/2402.05120

    In an ensemble, `n` agents are given the same task and produce `n` different
    responses. These responses are then synthesized together to produce a final anwer.
    Beehive currently supports two different synthesis methods: an LLM agent or a
    similarity function. In the former, Beehive creates a new LLM agent whose task
    is to combine all `n` responses into a better, final response. In the latter,
    Beehive computes the similarity between all pairs of responses and returns the
    answer that had the highest cumulative similarity.

    args:
    - `num_members` (int): number of members on the team.
    - `name` (str): the invokable name. Team members will be given the name `{name}-{i}`, where `i` is any number between 1 and `num_members`.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel` | `BaseChatModel`): chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`.
    - `state` (list[`BHMessage` | `BHToolMessage`] | list[`BaseMessage`]): list of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs.
    - `temperature` (int): temperature setting for the model.
    - `tools` (list[Callable[..., Any]]): functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
    - `response_model` (type[`BaseModel`] | None): response model for this agent. This should be a Pydantic BaseModel. Default is `None`.
    - `termination_condition` (Callable[..., bool] | None): condition which, if met, breaks the BeehiveAgent out of the chat loop. This should be a function that takes a `response_model` instance as input. Default is None.
    - `chat_loop` (int): number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns may require more loops. This should always be used with a `response_model` and a `termination_condition`.
    - `docstring_format` (`DocstringFormat` | None): docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
    - `final_answer_method` (Literal['llm', 'similarity']): method used to obtain the final answer from the agents. Either `llm` or `similarity`. If `llm`, then Beehive will create an agent with the inputted `synthesizer_model` and use that to synthesize the responses from the agents and generate a single, final response. If `similarity`, then Beehive will choose the answer that has the highest cumulative similarity to the other agents.
    - `synthesizer_model` (`BHChatModel` | `BaseChatModel`): model used to synthesize responses from agents and generate a final response. Only necessary if `final_answer_method`='llm'. This class *must* match the `model` class.
    - `similarity_score_func` (Callable[[str, str], float] | None): function used to compute the similarity score. Only necessary if `final_answer_method`='similarity'. The function must take two string arguments and return a float. If the callable is not specified, then Beehive defaults to the BLEU score from Papineni et al., 2002. Default is `None`.
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

    final_answer_method: Literal["llm", "similarity"] = Field(
        default="llm",
        description=(
            "Method used to obtain the final answer from the agents. Either `llm` or"
            " `similarity`. If `llm`, then Beehive will create an agent with the"
            " inputted `synthesizer_model` and use that to synthesize the responses"
            " from the agents and generate a single, final response. If `similarity`,"
            " then Beehive will choose the answer that has the highest cumulative"
            " similarity to the other agents."
        ),
    )

    # final_answer_method = llm
    synthesizer_model: AnyChatModel | None = Field(
        default=None,
        description=(
            "Model used to synthesize responses from agents and generate a final"
            " response."
        ),
    )

    # final_answer_method = similarity
    similarity_score_func: Callable[[str, str], float] | None = Field(
        default=None,
        description="Function used to compute the similarity score. The function definition"
        " must follow the format:\n"
        "```python\n"
        "def similarity_function(response1: str, response2: str) -> float\n"
        "    ...\n"
        "```\n"
        "where response1 and response2 represent string responses from an LLM"
        " and the returned value is the float representation of how similar the"
        " two responses are. This is only used if `final_answer_method=similarity`."
        " If the callable is not specified, then Beehive defaults to the BLEU score"
        " from Papineni et al., 2002.",
    )

    _synthesizer_agent: Agent | None = PrivateAttr()

    @field_validator("similarity_score_func")
    @classmethod
    def validate_similarity_score_func(
        cls, similarity_score_func: AnyCallableT | None
    ) -> Any:
        if not similarity_score_func:
            return None

        # Make sure each call to the similarity_score_func is validated. This requires
        # that the function is type-hinted. This function is generally used as a
        # decorator, so mypy complains that there's no overload variable that matches
        # the argument types, even though there is.
        return validate_call(similarity_score_func, validate_return=True)  # type: ignore

    @model_validator(mode="after")
    def define_convergence_method(self) -> "BeehiveEnsemble":
        # Summarizer agent. This takes the outputs from all the agents in _agent_list
        # and synthesizes them into one output that answers the user's original
        # question.
        if self.final_answer_method == "llm":
            if not self.synthesizer_model:
                raise ValueError(
                    "Must specify `synthesizer_model` if final_answer_method='llm'."
                )

            # Synthesizer model and member model should match
            if (
                self.synthesizer_model.__class__.__name__
                != self.model.__class__.__name__
            ):
                raise ValueError(
                    "`model` and `synthesizer_model` class must match! Found `{cls1}` and `{cls2}`, respectively.".format(
                        cls1=self.model.__class__.__name__,
                        cls2=self.synthesizer_model.__class__.__name__,
                    )
                )

            if isinstance(self.synthesizer_model, BHChatModel):
                self._synthesizer_agent = BeehiveAgent(
                    name=self.name,
                    backstory=self.backstory,
                    model=self.synthesizer_model,
                    tools=[],  # the synthesizer agent doesn't need tools
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
                self._synthesizer_agent = BeehiveLangchainAgent(
                    name=self.name,
                    backstory=self.backstory,
                    model=self.synthesizer_model,
                    tools=[],  # the synthesizer agent doesn't need tools
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
        elif self.final_answer_method == "similarity":
            if not self.similarity_score_func:
                logger.info(
                    "Defaulting to BLEU similarity score from Papineni et al., 2002"
                )

                # Default to the BLEU score from Papineni, Kishore, Salim Roukos, Todd
                # Ward, and Wei-Jing Zhu. 2002
                from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

                def calculate_bleu(response1: str, response2: str) -> float:
                    """
                    Calculate the BLEU score between two responses.

                    :param response1: First LLM response.
                    :type response1: str
                    :param response2: Second LLM response.
                    :type response2: str
                    :return: The BLEU score.
                    """
                    # Tokenize the sentences and calculate the BLEU score
                    response1_tokens = [response1.split()]
                    response2_tokens = response2.split()
                    smoothie = SmoothingFunction().method4
                    bleu_score = sentence_bleu(
                        response1_tokens, response2_tokens, smoothing_function=smoothie
                    )
                    return float(bleu_score)

                # Validate function call. `validate_call` is generally used as a
                # decorator, so mypy complains that there's no overload variable that
                # matches the argument types, even though there is.
                self.similarity_score_func = validate_call(  # type: ignore
                    calculate_bleu, validate_return=True
                )
        return self

    def _prepare_summarizer_prompt(
        self, task: str | BHMessage, agent_messages: dict[int, AnyMessageSequence]
    ) -> str:
        return SynthesizerPrompt(
            num_agents=str(self.num_members),
            task=task.content if isinstance(task, BHMessage) else task,
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
                BHMessage(role=MessageRole.USER, content=task)
                if isinstance(task, str)
                else task
            )
            agent.state.extend(agent_messages[i])
        self._agent_feedback = agent_feedback

        # Empty list for return
        final_messages: AnyMessageSequence = []
        final_bh_messages: AnyBHMessageSequence = []
        final_lc_messages: list[BaseMessage] = []

        # Send outputs to synthesizer
        synthesizer_task = self._prepare_summarizer_prompt(task, agent_messages)
        if self.final_answer_method == "llm":
            assert self._synthesizer_agent
            synthesizer_messages = self._synthesizer_agent._invoke(
                task=synthesizer_task,
                retry_limit=retry_limit,
                pass_back_model_errors=pass_back_model_errors,
                verbose=verbose,
                stream=stream,
                stdout_printer=stdout_printer,
            )

            # Type checking
            for output_elt in synthesizer_messages:
                if isinstance(output_elt, BaseMessage):
                    final_lc_messages.append(output_elt)
                else:
                    final_bh_messages.append(output_elt)

            # Only one of the `final_` messages should be non-empty
            if final_bh_messages and final_lc_messages:
                raise ValueError(
                    "Synthesizer agent returned incompatible message classes."
                )

            # mypy complains here, and I'm not entirely sure why.
            final_messages.extend(final_bh_messages)  # type: ignore
            final_messages.extend(final_lc_messages)  # type: ignore
            self._synthesizer_agent.state.append(
                BHMessage(role=MessageRole.USER, content=synthesizer_task)
            )
            self._synthesizer_agent.state.extend(final_messages)

        # For similarity scores, utilize the sampling-and-voting algorithm described
        # here: https://arxiv.org/pdf/2402.05120
        else:
            if not self.similarity_score_func:
                raise ValueError("`similarity_score_func` is not defined!")
            max_idx = -1
            max_score: float = 0.0
            for i_idx, i_msgs in agent_messages.items():
                i_str = _convert_messages_to_string(i_msgs)
                score: float = 0.0
                for j_idx, j_msgs in agent_messages.items():
                    if i_idx == j_idx:
                        continue
                    j_str = _convert_messages_to_string(j_msgs)
                    score += self.similarity_score_func(i_str, j_str)

                # Keep track of the index with the highest cumulative score
                if score > max_score:
                    max_idx = i_idx
                    max_score = score
            final_messages = agent_messages[max_idx]
            printer.print_invokable_output(final_messages)

        # Finally, add a summary message to the member agents' states
        if self.final_answer_method == "llm":
            summary_message = EnsembleLLMSummaryPrompt(
                num_agents=str(self.num_members),
                task=task.content if isinstance(task, BHMessage) else task,
                final_answer=" ".join(
                    filter(None, [str(msg.content) for msg in final_messages])
                ),
            ).render()
        else:
            summary_message = EnsembleFuncSummaryPrompt(
                num_agents=str(self.num_members),
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
        """Start a multi-agent ensemble in order to execute a task. In an ensemble, `n`
        agents are given the same task and produce `n` different responses. These
        responses are then synthesized together to produce a final anwer.

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

        # Context message
        context_template = (
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
