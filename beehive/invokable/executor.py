import json

# Logger
import logging

from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field, InstanceOf, PrivateAttr, model_validator
from sqlalchemy import select

from beehive.invokable.base import Feedback, Invokable
from beehive.invokable.types import (
    AnyChatModel,
    BHStateElt,
    EmbeddingDistance,
    ExecutorOutput,
)
from beehive.invokable.utils import _process_json_output
from beehive.memory.db_storage import DbStorage, TaskModel
from beehive.memory.feedback_storage import FeedbackStorage
from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.models.base import BHEmbeddingModel
from beehive.prompts import EvaluationPrompt, FeedbackPrompt
from beehive.utilities.printer import Printer

logger = logging.getLogger(__file__)


class InvokableExecutor(BaseModel):
    task: str | BHMessage = Field(
        description="The task that the invokable wishes to accomplish."
    )
    invokable: Invokable = Field(
        description=(
            "Invokable that we wish to execute. This could be an Agent, a LangchainAgent"
            " an BeehiveEnsemble, or a Beehive."
        )
    )

    _task_str: str = PrivateAttr()

    # From invokable
    _history: bool = PrivateAttr()
    _history_lookback: int = PrivateAttr()
    _feedback: bool = PrivateAttr()
    _feedback_embedder: BHEmbeddingModel | None = PrivateAttr()
    _feedback_model: AnyChatModel | None = PrivateAttr()
    _feedback_embedding_distance: EmbeddingDistance = PrivateAttr()
    _n_feedback_results: int = PrivateAttr()

    # Other attributes. These are intentionally private.
    _output: str = PrivateAttr()
    _db_storage: InstanceOf[DbStorage] = PrivateAttr()
    _feedback_storage: InstanceOf[FeedbackStorage] = PrivateAttr()

    @model_validator(mode="after")
    def define_private_attributes(self) -> "InvokableExecutor":
        self._task_str = (
            self.task.content if isinstance(self.task, BHMessage) else self.task
        )

        # Most of the private attributes are taken from the invokable
        self._history = self.invokable.history
        self._history_lookback = self.invokable.history_lookback
        self._feedback = self.invokable.feedback
        self._feedback_embedder = self.invokable.feedback_embedder
        self._feedback_model = self.invokable.feedback_model
        self._feedback_embedding_distance = self.invokable.feedback_embedding_distance
        self._n_feedback_results = self.invokable.n_feedback_results

        # Storage stuff
        self._db_storage = self.invokable._db_storage
        self._feedback_storage = FeedbackStorage(
            embedding_distance=self._feedback_embedding_distance
        )

        return self

    def evaluate_task(self) -> Feedback | None:
        if not self._feedback:
            return None

        # Otherwise, invoke the feedback agent
        pydantic_parser = PydanticOutputParser(pydantic_object=Feedback)
        prompt = EvaluationPrompt(
            task=self._task_str,
            output=self._output,
            format_instructions=pydantic_parser.get_format_instructions(),
        ).render()
        feedback_output: list[BHMessage | BHToolMessage] | list[BaseMessage]
        if isinstance(self._feedback_model, BaseChatModel):
            feedback_output = [self._feedback_model.invoke(input=prompt)]

        # Otherwise, the model must be a BHChatModel. Creating an explicit type check
        # prevents us from mocking the model and testing feedback generation, so ignore
        # this mypy error for now.
        else:
            feedback_output = self._feedback_model.chat(  # type: ignore
                task_message=BHMessage(role=MessageRole.USER, content=prompt),
                temperature=0,
                tools={},
                conversation=[],
            )

        # There shouldn't be any tool calls — the chat model should only produce one
        # message.
        assert len(feedback_output) == 1
        feedback_message = feedback_output[0].content

        # Hopefully, the LLM followed our format instructions
        try:
            feedback_json = json.loads(_process_json_output(str(feedback_message)))
            return Feedback(**feedback_json)
        except json.JSONDecodeError:
            print(f"Could not parse feedback: {feedback_message}")
            return None

    def grab_feedback_for_invokable_execution(self) -> BHMessage | None:
        # If this function is called, then the user wants to incorporate feedback into
        # their LLM response. Our embedding model must be defined.
        assert self._feedback_embedder

        feedback_results = self._feedback_storage.grab_feedback_from_similar_tasks(
            invokable=self.invokable,
            task=self._task_str,
            embedder=self._feedback_embedder,
            n_results=self._n_feedback_results,
        )
        if not feedback_results:
            return None

        # Construct messages
        if feedback_results["documents"]:
            # ChromaDB's `query` method accepts embeddings as a list of lists and
            # returns the associated documents as a list of lists. Each nested document
            # list correponds to matching nested embedding list. Since we only embed the
            # singular task, there should only be one set of nested lists in the
            # results.
            feedback_documents = feedback_results["documents"]
            system_message_content = FeedbackPrompt(
                feedback="\n".join([f"- {str(x)}" for x in feedback_documents]),
            ).render()
            return BHMessage(role=MessageRole.ASSISTANT, content=system_message_content)

        else:
            return None

    def execute(
        self,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> ExecutorOutput:
        feedback_message: BHMessage | None = (  # noqa
            self.grab_feedback_for_invokable_execution() if self._feedback else None
        )
        # Feedback message should be added before invoking the invokable
        if feedback_message:
            self.invokable.state.append(feedback_message)

        # Invoke the agent
        output_messages = self.invokable._invoke(
            task=self.task,
            retry_limit=retry_limit,
            pass_back_model_errors=pass_back_model_errors,
            verbose=verbose,
            stream=stream,
            stdout_printer=stdout_printer,
        )

        # Store stuff in memory
        db_task_id: str | None = None
        self._db_storage.add_invokable(self.invokable)
        if self.invokable._compatible_with_memory:
            db_task_id = self._db_storage.add_task(
                self._task_str,
                self.invokable,
            )
            db_task_model = self._db_storage.get_model_objects(
                select(TaskModel).where(TaskModel.id == db_task_id)
            )[0]

            # If we have return BHStateElt instances (e.g., the parent invokable is a
            # Beehive), then skip. We handle our db operations elsewhere in these cases.
            for msg in output_messages:
                if isinstance(msg, BHStateElt):
                    continue
                assert (
                    isinstance(msg, BHMessage)
                    or isinstance(msg, BHToolMessage)
                    or isinstance(msg, BaseMessage)
                )

                db_message_id = self._db_storage.add_message(
                    db_task_id,
                    msg,
                )
                if (
                    isinstance(msg, BHMessage) or isinstance(msg, BaseMessage)
                ) and hasattr(msg, "tool_calls"):
                    self._db_storage.add_tool_calls(msg, db_message_id)

            # Evaluate task and store feedback in vector database
            feedback_obj: Feedback | None
            if self._feedback:
                # AgentTeams store feedback for all the member agents in a dictionary
                # called `_agent_feedback.` Check if that exists.
                if hasattr(self.invokable, "_agent_feedback"):
                    total_confidence: int = 0
                    feedback_content: list[str] = []
                    for _, f in self.invokable._agent_feedback.items():
                        total_confidence += f.confidence
                        feedback_content.extend(f.suggestions)
                    feedback_obj = Feedback(
                        confidence=int(
                            total_confidence
                            / len(self.invokable._agent_feedback.keys())
                        ),
                        suggestions=feedback_content,
                    )
                else:
                    self._output = " ".join(
                        [
                            str(msg.content)
                            for msg in output_messages
                            if not isinstance(msg, BHStateElt)
                        ]
                    )
                    feedback_obj = self.evaluate_task()

                self._feedback_storage.embed_task_and_record_feedback(
                    self.invokable,
                    db_task_model,
                    feedback_obj,
                    self._feedback_embedder,
                )

        return {
            "task_id": db_task_id,
            "messages": output_messages,
            "printer": stdout_printer,
        }
