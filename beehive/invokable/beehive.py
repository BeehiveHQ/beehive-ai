import json
import logging
import traceback
from typing import Literal, Tuple

from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator
from pydantic_core import ValidationError

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Agent, Invokable, Route
from beehive.invokable.executor import InvokableExecutor
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.invokable.types import (
    AnyBHMessageSequence,
    BHStateElt,
    ExecutorOutput,
    InvokableQuestion,
)
from beehive.invokable.utils import _process_json_output
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.mixins.langchain import LangchainMixin
from beehive.models.base import BHChatModel
from beehive.prompts import (
    AskQuestionPrompt,
    ModelErrorPrompt,
    RouterPromptingPrompt,
    RouterQuestionPrompt,
    RouterRoutingPrompt,
)
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


class NextInvokableTask(BaseModel):
    task: str = Field(
        description="The specific task that should be assigned to the next agent."
    )


class WorkerNode(BaseModel):
    invokable: Invokable = Field(
        description="Invokable actor, e.g., an Agent, a Beehive, etc."
    )
    edges: list[Route] = Field(
        default_factory=list,
        description=(
            "List of edges X >> Y, where X and Y are invokable actors. This edge"
            " indicates that the LLM can route the conversation to Y after X has"
            " finished acting."
        ),
    )
    _prev: list[Invokable] = PrivateAttr(default_factory=list)
    _next: list[Invokable] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def set_private_attrs(self) -> "WorkerNode":
        self._next: list[Invokable] = []
        for x in self.edges:
            # We already know that edges are defined between two invokables. So, the
            # `this` and `other` edge attribute should have a name.
            if x.this.name == self.invokable.name:
                self._next.append(x.other)
            elif x.other.name == self.invokable.name:
                self._prev.append(x.this)

        return self

    @property
    def name(self):
        return self.invokable.name

    @property
    def backstory(self):
        return self.invokable.backstory


class FixedExecution(BaseModel):
    route: Route = Field(
        description=(
            "The route of invokables through which the Beehive should pass the"
            " conversation. Routes are specified via the `>>` operator, i.e.,\n"
            "```python\n"
            "    route=(inv1 >> inv2 >> inv1 >> inv3)\n"
            "```"
            "This gives users fine-tuned control over how the conversation gets passed"
            " to a set of invokables. This is only used if `use_llm_router = False`."
        ),
    )


class DynamicExecution(BaseModel):
    entrypoint: Invokable = Field(
        description=(
            "The first actor that receives the task. If `None`, then the task is passed"
            " to the router model, and the router determines who should act first."
        ),
    )
    edges: list[Route] = Field(
        default_factory=list,
        description=(
            "Edges between different invokables. The LLM router uses these edges to"
            " construct the conversation graph and determine which invokable should act"
            " next. This is only used if `use_llm_router = True`."
        ),
    )
    llm_router_additional_instructions: dict[Invokable, list[str]] = Field(
        default_factory=dict,
        description=(
            "Additional instructions that the LLM router can use to inform their choice"
            " for the next actor to act. This should be a dictionary mapping the invokable"
            " to a list of strings. Note that these instructions should be compatible with"
            " the specified `edges`."
        ),
    )


class Beehive(Invokable):
    """Beehive class to building complex LLM workflows. Beehives enable different
    invokables to collaborate with one another to achieve a task.

    args:
    - `name` (str): the invokable name.
    - `backstory` (str): backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.'
    - `model` (`BHChatModel` | `BaseChatModel`): chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`.
    - `state` (list[`BHStateElt`]): list of invokables and their completion messages.
    - `execution_process` (`FixedExecution` | `DynamicExecution`): execution process, either `FixedExecution` or `DynamicExecution`. If `FixedExecution`, then the Beehive will execute the Invokables in the specified `route` in order. If `DynamicExecution`, then Beehive uses an internal router agent to determine which `Invokable` to act given the previous messages / conversation."
    - `enable_questioning` (bool): Enable invokables to ask one another clarifying questions.
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

    examples:
    - See the documentation here: https://beehivehq.github.io/beehive-ai/
    """

    model_config = ConfigDict(extra="forbid")
    state: list[BHStateElt] = Field(
        default_factory=list,
        description="List of invokables and their completion messages.",
    )
    execution_process: FixedExecution | DynamicExecution = Field(
        description=(
            "Execution process, either `FixedExecution` or `DynamicExecution`. If"
            " `FixedExecution`, then the Beehive will execute the Invokables in the"
            " specified `route` in order. If `DynamicExecution`, then Beehive uses an"
            " internal router agent to determine which `Invokable` to act given the"
            " previous messages / conversation."
        )
    )
    enable_questioning: bool = Field(
        default=False,
        description=("Enable invokables to ask one another clarifying questions."),
    )

    _flag_dynamic: bool = PrivateAttr()
    _flag_fixed: bool = PrivateAttr()
    _compatible_with_memory: bool = False
    _router: Agent = PrivateAttr()
    _invokables: list[Invokable] = PrivateAttr(default_factory=list)
    _invokable_map: dict[str, Invokable] = PrivateAttr()
    _nodes: dict[str, WorkerNode] = PrivateAttr(default_factory=dict)
    _entrypoint_node: WorkerNode | None = PrivateAttr()

    # Context to provide to the invokables prior to invoking the Beehive. This is set in
    # the `context` keyword argument in `invoke` and is passed to the `_invoke` method
    # via this attribute. This prevents us from having to supply `_invoke` with a
    # context keyword argument.
    _invokation_context: list[Invokable] = PrivateAttr(default_factory=list)

    def _create_router(self, name: str, backstory: str) -> Agent:
        """This function makes it slightly easier to test the router's behavior under
        different conditions.
        """
        if isinstance(self.model, BHChatModel):
            return BeehiveAgent(model=self.model, name=name, backstory=backstory)
        elif isinstance(self.model, BaseChatModel):
            return BeehiveLangchainAgent(
                model=self.model,
                name=name,
                backstory=backstory,
            )
        else:
            raise ValueError(
                f"Unrecognized model class `{self.model.__class__.__name__}`."
            )

    @model_validator(mode="after")
    def setup(self) -> "Beehive":
        # Router — if we're using dynamic routing, then we're using an agent to route the
        # conversation between invokables.
        self._flag_dynamic = isinstance(self.execution_process, DynamicExecution)
        self._flag_fixed = isinstance(self.execution_process, FixedExecution)

        # Dynamic routing
        if self._flag_dynamic:
            # For mypy
            assert isinstance(self.execution_process, DynamicExecution)

            for e in self.execution_process.edges:
                if not (
                    isinstance(e.this, Invokable) or isinstance(e.other, Invokable)
                ):
                    raise ValueError("Edges should be defined between two Invokables!")
                if e.this == e.other:
                    raise ValueError("Edges must point to a different Invokable!")
                self._invokables.append(
                    e.this
                ) if e.this not in self._invokables else None
                self._invokables.append(
                    e.other
                ) if e.other not in self._invokables else None

            name = "Router"
            backstory = "You are an expert manager that specializes in directing a conversation between different LLM agents."
            self._nodes = {
                x.name: WorkerNode(invokable=x, edges=self.execution_process.edges)
                for x in self._invokables
            }
            self._entrypoint_node = WorkerNode(
                invokable=self.execution_process.entrypoint,
                edges=self.execution_process.edges,
            )

        # Prompter — if `use_llm_router=False`, then the user is passing the
        # conversation through a specific route of invokables. Use an LLM to dynamically
        # prompt the invokables appropriately.
        else:
            # For mypy
            assert isinstance(self.execution_process, FixedExecution)

            for x in self.execution_process.route._invokable_order:
                self._invokables.append(x) if x not in self._invokables else None

            # We use a router agent to craft prompts for invokables in the route.
            name = "Prompter"
            backstory = "You are an expert prompter. You specialize in developing concise prompts for LLM agents given a task and some context."
            self._entrypoint_node = None

        # Create router agent
        self._router = self._create_router(name, backstory)

        self._invokable_map = {x.name: x for x in self._invokables}
        return self

    def convert_non_bh_executor_output_to_message_list(
        self, invokable: Invokable, completion_output: ExecutorOutput
    ) -> Tuple[AnyBHMessageSequence, BHMessage | BHToolMessage]:
        """Convert ExecutorOutput from a non-Beehive invokable to a list of messages.
        This list is then used to generate a new `BHStateElt` which is appended onto
        the `state` attribute.

        Also, return the last message, since we sometimes use this to check if
        a question was asked.
        """
        completion_messages: AnyBHMessageSequence = []
        for output_elt in completion_output["messages"]:
            if isinstance(output_elt, BHStateElt):
                raise ValueError(
                    "Expected `BHMessage | BHToolMessage | BaseMessage` from invokable output, got `BHStateElt`."
                )
            if isinstance(output_elt, BaseMessage):
                # The invokable needs contain the `LangchainMixin` class.
                if not isinstance(invokable, LangchainMixin):
                    raise ValueError(
                        "Node invokable does not support Langchain conversion!"
                    )
                output_as_bh_message = (
                    invokable._convert_langchain_message_to_beehive_message(output_elt)
                )
            else:
                output_as_bh_message = output_elt
            completion_messages.append(output_as_bh_message)

        # Return all completion messages and the last message
        return completion_messages, output_as_bh_message

    def get_invokables_names_descriptions(
        self, other_invokables: list[Invokable]
    ) -> Tuple[list[str], list[str]]:
        other_names = [x.name for x in other_invokables]
        other_descriptions = [f"- {x.name}: {x.backstory}" for x in other_invokables]
        return other_names, other_descriptions

    def create_next_agent_actor_pydantic_class(
        self, valid_agent_names: list[str]
    ) -> type[BaseModel]:
        if "FINISH" not in valid_agent_names:
            valid_agent_names.append("FINISH")

        class NextAgentActor(BaseModel):
            agent: Literal[tuple(valid_agent_names)] = Field(  # type: ignore
                description="The next agent to act. 'FINISH' if the user's original question has been answered."
            )
            reason: str = Field(description="Rationale for choosing the next agent.")
            task: str = Field(
                description="The specific task that should be assigned to the next agent."
            )

        return NextAgentActor

    def create_question_pydantic_class(
        self, allowed_invokable_names: list[str]
    ) -> type[BaseModel]:
        class OtherInvokableQuestion(InvokableQuestion):
            invokable: Literal[tuple(allowed_invokable_names)] = Field(  # type: ignore
                description="Agent to whom the question will be asked."
            )

        return OtherInvokableQuestion

    def prompt_router_to_direct_question(
        self,
        question: str,
    ) -> str:
        """Ask router who should get the question"""
        (
            invokable_names,
            invokable_descriptions,
        ) = self.get_invokables_names_descriptions(other_invokables=self._invokables)
        question_prompt = RouterQuestionPrompt(
            question=question,
            agents=", ".join(invokable_names),
            agent_descriptions="\n".join(invokable_descriptions),
        ).render()
        return question_prompt

    def prompt_router_for_next_agent(
        self,
        original_task: str,
        node: WorkerNode,
        prev_invokables: list[Invokable],
    ) -> Tuple[str, list[Invokable]]:
        if not isinstance(self.execution_process, DynamicExecution):
            raise ValueError(
                "Execution process is fixed! Does not support routing to agents."
            )

        # Format instructions using a Pydantic parser
        next_agents = node._next

        # Format instructions
        valid_agent_names: list[str] = [x.name for x in next_agents]
        valid_agent_names.append("FINISH")
        NextAgentActor = self.create_next_agent_actor_pydantic_class(valid_agent_names)
        pydantic_parser = PydanticOutputParser(pydantic_object=NextAgentActor)
        format_instructions = pydantic_parser.get_format_instructions()

        # Construct and return message
        (
            next_agent_names_list,
            next_agent_descriptions_list,
        ) = self.get_invokables_names_descriptions(next_agents)
        next_agent_names = ", ".join(next_agent_names_list + ["FINISH"])
        next_agent_descriptions = "\n".join(
            next_agent_descriptions_list
            + ["- FINISH: The user's original question has been answered."]
        )
        additional_instructions = "\n".join(
            [
                f"- {inst}"
                for inst in self.execution_process.llm_router_additional_instructions.get(
                    node.invokable, []
                )
            ]
        )
        context = self.augment_invokable_with_context(prev_invokables)
        prompt = RouterRoutingPrompt(
            agents=next_agent_names,
            agent_descriptions=next_agent_descriptions,
            task=original_task,
            context=context.content if context else None,
            additional_instructions=additional_instructions,
            format_instructions=format_instructions,
        ).render()
        return prompt, next_agents

    def prompt_router_for_next_task(
        self, original_task: str, backstory: str, prev_invokables: list[Invokable]
    ) -> str:
        pydantic_parser = PydanticOutputParser(pydantic_object=NextInvokableTask)
        format_instructions = pydantic_parser.get_format_instructions()
        context = self.augment_invokable_with_context(prev_invokables)
        return RouterPromptingPrompt(
            backstory=backstory,
            task=original_task,
            context=context.content if context else None,
            format_instructions=format_instructions,
        ).render()

    def generate_pose_a_question_prompt(
        self,
        invokable_asking_question: Invokable,
        valid_invokables_to_question: list[Invokable],
    ) -> str:
        """Add ability for invokable to pose a question to another invokable. This gets
        appended to the invokable task so that Invokables can ask questions to other
        Invokables in the Beehive.
        """
        if not valid_invokables_to_question:
            return ""
        if invokable_asking_question._flag_has_seen_question_schema:
            question_format_instructions = "IF YOU WANT TO ASK A QUESTION, use the same JSON schema you used previously to format your question."
        else:
            question_pydantic_object = self.create_question_pydantic_class(
                allowed_invokable_names=[x.name for x in valid_invokables_to_question]
            )
            pydantic_parser = PydanticOutputParser(
                pydantic_object=question_pydantic_object
            )
            question_format_instructions = pydantic_parser.get_format_instructions()

            # Replace some of the words in the format instructions
            question_format_instructions = question_format_instructions.replace(
                "The output should be formatted as a JSON instance that conforms to the JSON schema below.",
                "IF YOU WANT TO ASK A QUESTION, format the output as a JSON instance that conforms to the JSON schema below.",
            )

        (
            other_agent_names_list,
            other_agent_descriptions_list,
        ) = self.get_invokables_names_descriptions(valid_invokables_to_question)
        question_prompt = AskQuestionPrompt(
            agent_names=", ".join(other_agent_names_list),
            agent_descriptions="\n".join(other_agent_descriptions_list),
            format_instructions=question_format_instructions,
        ).render()
        return question_prompt

    def update_state_with_question_role(
        self,
        invokable: Invokable,
    ) -> None:
        last_elt_of_state = invokable.state.pop()
        if not isinstance(last_elt_of_state, BHMessage):
            raise ValueError(
                f"Expected last element of state to have type `BHMessage`, instead found `{last_elt_of_state.__class__.__name__}"
            )

        # Change the role
        new_message = BHMessage(
            role=MessageRole.QUESTION, content=last_elt_of_state.content
        )
        invokable.state.append(new_message)

    def invoke_router(
        self,
        counter: int,
        retry_limit: int,
        task: str,
        pass_back_model_errors: bool,
        pydantic_model: type[BaseModel],
        printer: Printer,
    ) -> Tuple[BaseModel, int]:
        # Wrap the router prompt in a `while` loop in case there are parsing
        # errors in the router's response. Make sure to only pass the
        # conversation *back* to the router if `pass_back_model_errors=True`
        while True:
            if counter > retry_limit:
                break
            counter += 1

            # Invoke the router (without any feedback or history)
            router_messages = self._router._invoke(
                task,
                retry_limit=retry_limit,
                pass_back_model_errors=pass_back_model_errors,
                verbose=False,
                stdout_printer=None,
            )
            next_agent_message = router_messages[-1]

            try:
                next_agent_json = json.loads(
                    _process_json_output(next_agent_message.content)
                )
                model_obj = pydantic_model(**next_agent_json)
                break

            except (json.decoder.JSONDecodeError, ValidationError):
                counter += 1
                if counter > retry_limit:
                    printer.print_standard(
                        "[red]ERROR:[/red] Router exceeded total retry limit."
                    )
                    raise
                elif pass_back_model_errors:
                    additional_system_message = BHMessage(
                        role=MessageRole.SYSTEM,
                        content=f"Encountered a `JSONDecodeError` / Pydantic `ValidationError` with the following content: <content>{next_agent_message.content}</content>. **All output must be formatted according to the JSON schema described in the instructions**. Do not make this same mistake again.",
                    )
                    self._router.state.append(additional_system_message)
                else:
                    raise

            # TODO - we can probably clean this up
            except Exception:
                counter += 1
                if counter > retry_limit:
                    printer.print_standard(
                        "[red]ERROR:[/red] Router exceeded total retry limit."
                    )
                    raise
                elif pass_back_model_errors:
                    additional_system_message = BHMessage(
                        role=MessageRole.SYSTEM,
                        content=ModelErrorPrompt(
                            error=str(traceback.format_exc())
                        ).render(),
                    )
                    self._router.state.append(additional_system_message)
                else:
                    raise
        return model_obj, counter

    def invoke_router_without_route(
        self,
        original_task: str,
        current_node: WorkerNode,
        counter: int,
        retry_limit: int,
        pass_back_model_errors: bool,
        printer: Printer,
    ) -> Tuple[str, str, int]:
        # Set the router's state. It should contain data from all the active agents
        # to ensure that it routes the conversation appropriately.
        self._router.reset_conversation()
        for elt in self.state:
            self._router.state.append(
                BHMessage(
                    role=MessageRole.USER,
                    content=elt.task,
                )
            )
            self._router.state.extend(elt.completion_messages)

        # Prompt the router
        prompt_task, next_agents = self.prompt_router_for_next_agent(
            original_task=original_task,
            node=current_node,
            prev_invokables=[
                elt.invokable for elt in self.state if elt.invokable is not None
            ],
        )
        NextAgentActor = self.create_next_agent_actor_pydantic_class(
            [x.name for x in next_agents]
            if next_agents
            else [x.name for x in current_node._next]
        )
        next_agent, counter = self.invoke_router(
            counter=counter,
            retry_limit=retry_limit,
            task=prompt_task,
            pass_back_model_errors=pass_back_model_errors,
            pydantic_model=NextAgentActor,
            printer=printer,
        )
        assert isinstance(next_agent, NextAgentActor)

        # mypy complains when we try to directly access the model attributes. Instead,
        # convert the model to a dictionary and grab the attributes via keys.
        next_agent_dict = next_agent.model_dump()
        next_agent_name: str | None = next_agent_dict.get("agent", None)
        next_agent_reason: str | None = next_agent_dict.get("reason", None)
        task: str | None = next_agent_dict.get("task", None)
        if next_agent_name is None or next_agent_reason is None or task is None:
            raise ValueError(
                f"Router did return properly formatted output: {next_agent_dict}!"
            )
        return next_agent_name, task, counter

    def invoke_router_with_route(
        self,
        original_task: str,
        backstory: str,
        counter: int,
        retry_limit: int,
        pass_back_model_errors: bool,
        printer: Printer,
    ) -> Tuple[str, int]:
        # Set the router's state. It should contain data from all the active agents
        # to ensure that it routes the conversation appropriately.
        self._router.reset_conversation()
        for elt in self.state:
            self._router.state.append(
                BHMessage(
                    role=MessageRole.USER,
                    content=elt.task,
                )
            )
            self._router.state.extend(elt.completion_messages)

        # Prompt the router
        prompt_task = self.prompt_router_for_next_task(
            original_task=original_task,
            backstory=backstory,
            prev_invokables=[
                elt.invokable for elt in self.state if elt.invokable is not None
            ],
        )
        next_task, counter = self.invoke_router(
            counter=counter,
            retry_limit=retry_limit,
            task=prompt_task,
            pass_back_model_errors=pass_back_model_errors,
            pydantic_model=NextInvokableTask,
            printer=printer,
        )
        assert isinstance(next_task, NextInvokableTask)
        return next_task.task, counter

    def grab_history_for_invokable_execution(self) -> list[BHMessage | BHToolMessage]:
        history: list[BHMessage | BHToolMessage] = []
        for inv in self._invokables:
            history.extend(inv.grab_history_for_invokable_execution())
        return history

    def _invoke_with_question(
        self,
        question: str,
        retry_limit: int,
        pass_back_model_errors: bool,
        printer: Printer,
    ) -> list[BHStateElt]:
        # Set the router's state. It should contain data from all the active agents
        # to ensure that it routes the conversation appropriately.
        self._router.reset_conversation()
        for elt in self.state:
            self._router.state.append(
                BHMessage(
                    role=MessageRole.USER,
                    content=elt.task,
                )
            )
            self._router.state.extend(elt.completion_messages)

        class NextInv(BaseModel):
            invokable: str = Field("Invokable to answer question.")

        question_prompt = self.prompt_router_to_direct_question(question)
        next_invokable_obj, _ = self.invoke_router(
            counter=0,  # Questions are handled first, so the counter will be 0.
            retry_limit=retry_limit,
            task=question_prompt,
            pass_back_model_errors=pass_back_model_errors,
            pydantic_model=NextInv,
            printer=printer,
        )
        if not isinstance(next_invokable_obj, NextInv):
            raise ValueError(
                f"Router did not produce correctly formatted output: {next_invokable_obj.model_dump_json()}"
            )

        # Pose the question to the next invokable.
        next_invokable = self._invokable_map[next_invokable_obj.invokable]
        completion_output = next_invokable.invoke(
            task=question,
            retry_limit=retry_limit,
            pass_back_model_errors=pass_back_model_errors,
            context=self._invokation_context,
            stdout_printer=printer,
        )

        # `completion_output` should always be from a non-Beehive invokable. Suppose
        # `next_invokable` is a Beehive. Since `question` is a BHMessage, the Beehive's
        # `invoke` method will direct the question to a child invokable. This will
        # continue recursively until we finally hit a non-Beehive invokable (though we
        # may want to put some sort of recursive limit on this).
        completion_messages, _ = self.convert_non_bh_executor_output_to_message_list(
            invokable=next_invokable, completion_output=completion_output
        )
        state_elt = BHStateElt(
            index=0,  # Questions are handled first, so index should be 0
            task_id=completion_output["task_id"],
            task=question,
            invokable=next_invokable,
            completion_messages=completion_messages,
        )
        self.state.append(state_elt)
        return [state_elt]

    def _invoke_without_route(
        self,
        task: str,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[BHStateElt]:
        assert self._entrypoint_node

        # Keep track of the original task
        original_task = task

        # Define the printer
        printer = stdout_printer if stdout_printer else Printer()

        # First, direct the conversation to the entrypoint.
        previous_node: WorkerNode | None = None
        current_node: WorkerNode = self._entrypoint_node

        # Keep track of number of LLM calls
        counter = 0

        # First invokable's task
        prompt_task = self.prompt_router_for_next_task(
            original_task=original_task,
            backstory=current_node.invokable.backstory,
            prev_invokables=[],
        )
        next_task, counter = self.invoke_router(
            counter=counter,
            retry_limit=retry_limit,
            task=prompt_task,
            pass_back_model_errors=pass_back_model_errors,
            pydantic_model=NextInvokableTask,
            printer=printer,
        )
        if not isinstance(next_task, NextInvokableTask):
            raise ValueError("Pydantic validation failed for `next_task`.")

        # First invokable's task. This variable gets overridden during every iteration.
        child_invokable_task: str | BHMessage = next_task.task

        # Global variable to keep track of questions
        flag_question_being_asked: bool = False

        # Current elements
        elts_curr_invokation: list[BHStateElt] = []

        idx = 0
        while True:
            # Task as a string. This is necessary for the BHStateElt type.
            child_invokable_task_str = (
                child_invokable_task.content
                if isinstance(child_invokable_task, BHMessage)
                else child_invokable_task
            )

            if counter > retry_limit:
                break
            counter += 1

            # Invoke the current agent and overwrite the completion messages. The
            # `child_invokable_task` here is overwritten at each loop.
            idx += 1
            completion_output = current_node.invokable.invoke(
                task=child_invokable_task,
                retry_limit=retry_limit,
                pass_back_model_errors=pass_back_model_errors,
                verbose=verbose,
                stream=stream,
                context=self._invokation_context,
                stdout_printer=printer,
            )

            # Printer
            printer = completion_output["printer"]  # type: ignore

            # Messages
            flag_beehive_output = isinstance(current_node.invokable, Beehive)
            if flag_beehive_output:
                for output_elt in completion_output["messages"]:
                    if not isinstance(output_elt, BHStateElt):
                        raise ValueError(
                            f"Expected `BHStateElt` with Beehive output, got `{output_elt.__class__.__name__}`."
                        )
                    elts_curr_invokation.append(output_elt)
                    self.state.append(output_elt)
                    output_as_bh_message = output_elt.completion_messages[-1]

            # Otherwise, we need to convert the messages to Beehive messages and create
            # a new BHStateElt.
            else:
                (
                    completion_messages,
                    output_as_bh_message,
                ) = self.convert_non_bh_executor_output_to_message_list(
                    invokable=current_node.invokable,
                    completion_output=completion_output,
                )
                state_elt = BHStateElt(
                    index=idx,
                    task_id=completion_output["task_id"],
                    task=child_invokable_task_str,
                    invokable=current_node.invokable,
                    completion_messages=completion_messages,
                )
                elts_curr_invokation.append(state_elt)
                self.state.append(state_elt)

            # The Beehive's state should consist of all messages for all the child
            # invokables. Note that this state is never used to invoke a chat model.
            if not previous_node or previous_node.name != current_node.name:
                if current_node.invokable not in self._invokation_context:
                    self._invokation_context.append(current_node.invokable)

            # `output_as_bh_message` is set to invokable response's last message. Check
            # if the invokable is asking a question. Only enable this if
            # `enable_questioning` is True.
            try:
                if self.enable_questioning:
                    question_obj = InvokableQuestion(
                        **json.loads(output_as_bh_message.content)  # type: ignore
                    )

                    # Update the role of `output_as_bh_message` to have role QUESTION.
                    self.update_state_with_question_role(current_node.invokable)

                    # Set current node / task and automatically proceed to the next
                    # iteration of the loop.
                    child_invokable_task = question_obj.question
                    previous_node = current_node

                    # The question could either be posed to both nodes inside the
                    # Beehive or agents outside the Beehive.
                    if question_obj.invokable in list(self._nodes.keys()):
                        current_node = self._nodes[question_obj.invokable]
                    else:
                        # If the invokable exists outside of the Beehive, then we need
                        # to create a temporary node for it. This node should have one
                        # edge that enables communication between this external node and
                        # the agent that's actually asking the question.
                        next_inv = [
                            invoked
                            for invoked in self._invokation_context
                            if invoked.name == question_obj.invokable
                        ][0]
                        current_node = WorkerNode(
                            invokable=next_inv,
                            edges=[(next_inv >> previous_node.invokable)],
                        )

                    # Set flags. These help us direct the conversation appropriately.
                    flag_question_being_asked = True

                    continue

            except IndexError:
                raise ValueError(
                    "Trying to ask a question to an invokable that has not acted!"
                )

            # No question is being asked...continue
            except Exception:
                pass

            # If there are no agents that need to act next, then return
            if len(current_node._next) == 0:
                break

            # Otherwise, the current node has a few potential routes that it can go
            # down. Figure out the next agent, if any, should act next.
            else:
                if not self._router:
                    raise ValueError(
                        "Need to specify a router to direct communcation between agents!"
                    )

                # If a question is being asked, then `previous_node` asked
                # `current_node`. We need to send the conversation back to
                # `previous_node`.
                if flag_question_being_asked:
                    if not previous_node:
                        raise ValueError(
                            "A question is being asked even though no previous invokables have acted. This is a mistake."
                        )
                    if not self.enable_questioning:
                        raise ValueError(
                            "A question is being asked even though `enable_questioning` is False! This is a mistake."
                        )
                    current_node, previous_node = previous_node, current_node

                    # Use the previous task. This will be the task associated with the
                    # second-to-last object in self.state (since the last object will
                    # just be the response to the question)
                    child_invokable_task = self.state[-2].task

                    # Reset global variable
                    flag_question_being_asked = False
                    continue

                else:
                    (
                        next_agent_name,
                        child_invokable_task,
                        counter,
                    ) = self.invoke_router_without_route(
                        original_task=original_task,
                        current_node=current_node,
                        counter=counter,
                        retry_limit=retry_limit,
                        pass_back_model_errors=pass_back_model_errors,
                        printer=printer,
                    )

                    # If we're not finished, then invoke the next agent
                    if next_agent_name != "FINISH":
                        try:
                            previous_node, current_node = (
                                current_node,
                                self._nodes[next_agent_name],
                            )

                            if self.enable_questioning:
                                # Enable dynamic questioning — only enable the invokable
                                # to ask questions to other invokables that have already
                                # acted
                                valid_invokables_to_question: list[Invokable] = []
                                for inv in self._invokation_context:
                                    if (
                                        inv.model_dump()
                                        != current_node.invokable.model_dump()
                                        and inv not in valid_invokables_to_question
                                    ):
                                        valid_invokables_to_question.append(inv)

                                question_prompt = self.generate_pose_a_question_prompt(
                                    current_node.invokable, valid_invokables_to_question
                                )
                                child_invokable_task = "\n\n".join(
                                    filter(
                                        None, [child_invokable_task, question_prompt]
                                    )
                                )

                                # Set flags. This enables us to minimize the tokens in
                                # the context window.
                                current_node.invokable._flag_has_seen_question_schema = True

                            printer.print_router_text(next_agent_name)

                        except KeyError:
                            print(f"Could not find agent `{next_agent_name}`!")
                            break

                    # The router has determined that we have finished the current task.
                    else:
                        break

        return elts_curr_invokation

    def _invoke_with_route(
        self,
        task: str,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[BHStateElt]:
        """Invoke the Beehive using a pre-defined route."""
        if not isinstance(self.execution_process, FixedExecution):
            raise ValueError(
                "Execution process is dynamic! Does not support fixed route."
            )
        # Define the printer
        printer = stdout_printer if stdout_printer else Printer()

        # Keep track of number of LLM calls and the original task
        counter = 0
        original_task = task

        # First invokable's task
        prompt_task = self.prompt_router_for_next_task(
            original_task=original_task,
            backstory=self.execution_process.route._invokable_order[0].backstory,
            prev_invokables=[],
        )
        next_task, counter = self.invoke_router(
            counter=counter,
            retry_limit=retry_limit,
            task=prompt_task,
            pass_back_model_errors=pass_back_model_errors,
            pydantic_model=NextInvokableTask,
            printer=printer,
        )
        if not isinstance(next_task, NextInvokableTask):
            raise ValueError("Pydantic validation failed for `next_task`.")

        # First invokable's task. This variable gets overridden during every iteration.
        child_invokable_task: str | BHMessage = next_task.task

        # Global variable to keep track of questions
        flag_question_being_asked: bool = False

        # Current elements
        elts_curr_invokation: list[BHStateElt] = []

        # Invoke the route
        i = 1
        prev_inv: Invokable | None = None
        curr_inv: Invokable = self.execution_process.route._invokable_order[i - 1]
        while True:
            # Task as a string. This is necessary for the BHStateElt type.
            child_invokable_task_str = (
                child_invokable_task.content
                if isinstance(child_invokable_task, BHMessage)
                else child_invokable_task
            )

            # Typing is rough given the import dependencies
            if not isinstance(curr_inv, Invokable):
                raise ValueError(
                    f"Route's `invokable_order` contains non-Invokable: `{curr_inv.__class__.__name__}`"
                )
            completion_output = curr_inv.invoke(
                task=child_invokable_task,
                retry_limit=retry_limit,
                pass_back_model_errors=pass_back_model_errors,
                verbose=verbose,
                stream=stream,
                context=self._invokation_context,
                stdout_printer=printer,
            )

            # Update the printer
            printer = completion_output["printer"]  # type: ignore

            # If the output is from a Beehive, then just append the output element.
            flag_beehive_output = isinstance(curr_inv, Beehive)
            if flag_beehive_output:
                for output_elt in completion_output["messages"]:
                    if not isinstance(output_elt, BHStateElt):
                        raise ValueError(
                            f"Expected `BHStateElt` with Beehive output, got `{output_elt.__class__.__name__}`."
                        )
                    elts_curr_invokation.append(output_elt)
                    self.state.append(output_elt)
                    output_as_bh_message = output_elt.completion_messages[-1]

            else:
                (
                    completion_messages,
                    output_as_bh_message,
                ) = self.convert_non_bh_executor_output_to_message_list(
                    invokable=curr_inv, completion_output=completion_output
                )
                state_elt = BHStateElt(
                    index=i,
                    task_id=completion_output["task_id"],
                    task=child_invokable_task_str,
                    invokable=curr_inv,
                    completion_messages=completion_messages,
                )
                elts_curr_invokation.append(state_elt)
                self.state.append(state_elt)

            # Previous invokables used for next invokable's context
            self._invokation_context.append(curr_inv)

            # `output_as_bh_message` is set to invokable response's last message. Check
            # if the invokable is asking a question. Only enable this if
            # `enable_questioning` is True.
            try:
                if self.enable_questioning:
                    question_obj = InvokableQuestion(
                        **json.loads(output_as_bh_message.content)  # type: ignore
                    )

                    # Update the role of `output_as_bh_message` to have role QUESTION.
                    self.update_state_with_question_role(curr_inv)

                    # Set current inv / task and automatically proceed to the next
                    # iteration of the loop.
                    child_invokable_task = question_obj.question
                    prev_inv = curr_inv

                    # Our question prompts only allows questions to be asked to
                    # invokables that have already acted.
                    curr_inv = [
                        invoked
                        for invoked in self._invokation_context
                        if invoked.name == question_obj.invokable
                    ][0]

                    # Set flags. These help us direct the conversation appropriately.
                    # Then, proceed to the next iteration of the loop.
                    flag_question_being_asked = True
                    continue

            except IndexError:
                raise ValueError(
                    "Trying to ask a question to an invokable that has not acted!"
                )

            # No question is being asked...continue
            except Exception:
                pass

            # If a question was asked, then we need to route the conversation back to
            # the invokable that asked the question.
            if flag_question_being_asked:
                if not prev_inv:
                    raise ValueError(
                        "A question is being asked even though no previous invokables have acted. This is a mistake."
                    )
                if not self.enable_questioning:
                    raise ValueError(
                        "A question is being asked even though `enable_questioning` is False! This is a mistake."
                    )

                curr_inv, prev_inv = prev_inv, curr_inv

                # Use the previous task. This will be the task associated with the
                # second-to-last object in self.state (since the last object will just
                # be the response to the question)
                child_invokable_task = self.state[-2].task

                # Reset global variable
                flag_question_being_asked = False
                continue

            # Otherwise, route the conversation to the next invokable in the route.
            else:
                i += 1
                if i > len(self.execution_process.route._invokable_order):
                    break
                prev_inv = curr_inv

                # If we're on the first invokable, then just pass the inputted task.
                # Otherwise, define the next invokable and prompt the router to define the
                # next task. We only do this if a question is not being asked.
                curr_inv = self.execution_process.route._invokable_order[i - 1]
                child_invokable_task, counter = self.invoke_router_with_route(
                    original_task=original_task,
                    backstory=curr_inv.backstory,
                    counter=counter,
                    retry_limit=retry_limit,
                    pass_back_model_errors=pass_back_model_errors,
                    printer=printer,
                )

                # If we're enabling questions, then augment the child_invokable_task
                # with the question prompt.
                if self.enable_questioning:
                    # Enable dynamic questioning — only enable the invokable to ask
                    # questions to other invokables that have already acted
                    valid_invokables_to_question: list[Invokable] = []
                    for inv in self._invokation_context:
                        if (
                            inv.model_dump() != curr_inv.model_dump()
                            and inv not in valid_invokables_to_question
                        ):
                            valid_invokables_to_question.append(inv)

                    question_prompt = self.generate_pose_a_question_prompt(
                        curr_inv, valid_invokables_to_question
                    )
                    child_invokable_task = "\n\n".join(
                        filter(None, [child_invokable_task, question_prompt])
                    )

                    # Set flags. This enables us to minimize the tokens in
                    # the context window.
                    curr_inv._flag_has_seen_question_schema = True

                printer.print_router_text(curr_inv.name)

        return elts_curr_invokation

    def _invoke(
        self,
        task: str | BHMessage,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[BHStateElt]:
        printer = stdout_printer if stdout_printer else Printer()

        # Check if a question is being asked. This is only possible if the Beehive has
        # some invokation context (i.e., it's not the first invokable to act).
        last_msg: BHMessage | BHToolMessage | None = None
        if self._invokation_context:
            last_inv_in_context = self._invokation_context[-1]
            if isinstance(last_inv_in_context, Beehive):
                last_elt_of_state = last_inv_in_context.state[-1]
                if not isinstance(last_elt_of_state, BHStateElt):
                    raise ValueError(
                        f"Expected last element of state to be `BHStateElt`, instead found `{last_elt_of_state.__class__.__name__}`"
                    )
                last_msg = last_elt_of_state.completion_messages[-1]
            else:
                last_msg = last_inv_in_context.state[-1]

        # Invokation task
        invokation_task = task.content if isinstance(task, BHMessage) else task

        # If the Beehive is being asked a question, the last message will have role
        # `QUESTION`.
        if (
            last_msg
            and isinstance(last_msg, BHMessage)
            and last_msg.role == MessageRole.QUESTION
        ):
            return self._invoke_with_question(
                task.content if isinstance(task, BHMessage) else task,
                retry_limit,
                pass_back_model_errors,
                printer,
            )

        elif self._flag_dynamic:
            return self._invoke_without_route(
                task=invokation_task,
                retry_limit=retry_limit,
                pass_back_model_errors=pass_back_model_errors,
                verbose=verbose,
                stream=stream,
                stdout_printer=printer,
            )
        else:
            return self._invoke_with_route(
                task=invokation_task,
                retry_limit=retry_limit,
                pass_back_model_errors=pass_back_model_errors,
                verbose=verbose,
                stream=stream,
                stdout_printer=printer,
            )

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
        """Invoke the Beehive to execute a task.

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
          - `messages` (list[`BHStateElt`])
          - `printer` (`output.printer.Printer`)

        examples:
        - See the documentation here: https://beehivehq.github.io/beehive-ai/
        """
        printer = stdout_printer if stdout_printer else Printer()
        if verbose:
            printer.register_beehive(self.name)

        # Set the invokation context
        if context:
            self._invokation_context += context

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
        bh_db_id = executor._db_storage.add_beehive(
            self, task.content if isinstance(task, BHMessage) else task
        )
        bh_state = exec_output["messages"]
        for elt in bh_state:
            if not isinstance(elt, BHStateElt):
                raise ValueError("Incorrect return element type!")
            if not elt.task_id:
                raise ValueError("Executor task does not have an ID!")
            task = executor._db_storage.add_task_to_beehive(
                elt.task_id,
                bh_db_id,
            )
        if verbose:
            printer.unregister_beehive(self.name)
        exec_output["printer"] = printer
        return exec_output
