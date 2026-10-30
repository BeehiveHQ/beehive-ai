from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field, PrivateAttr, ValidationError, model_validator

from beehive.message import BHMessage, BHToolMessage, MessageRole
from beehive.tools.base import BHTool


class Model(BaseModel):
    model: str = Field(description="LLM model, e.g., `gpt-3.5-turbo`")
    _client: Any | None = PrivateAttr()

    @model_validator(mode="after")
    def create_client(self) -> "Model":
        self._client = self._create_client(
            **self.model_extra if self.model_extra else {}
        )
        return self

    def _create_client(self, **client_kwargs):
        raise NotImplementedError()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model='{self.model}')"


class BHChatModel(Model):
    def _structured_completion(
        self,
        response_model: type[BaseModel],
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
    ) -> str:
        """Provider hook: return JSON text, or raise for transport/provider errors."""
        raise NotImplementedError(
            "This provider does not implement structured outputs."
        )

    def chat(
        self,
        task_message: BHMessage | None,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
        response_model: type[BaseModel] | None = None,
        max_retries: int = 2,
    ) -> list[BHMessage | BHToolMessage]:
        """Chat, optionally validating against a Pydantic response model.

        Without a response model, return a single ordinary completion. For
        structured outputs, retries exclude the first attempt:

        1. If the output fails validation, we ask the model to correct it. Those
           correction messages are **not** saved in the agent's conversation history.
        2. Provider errors (network failures, authentication errors, and API errors) are
           never retried as validation errors
        3. The final error is raised. With max_retries=2, we try once plus two
           correction attempts. If all three fail validation, the caller receives the
           last Pydantic ValidationError
        """
        if task_message and task_message.processed_role != MessageRole.USER:
            raise ValueError(
                f"Invalid task_message role. Expected 'user', found '{task_message.processed_role}'."
            )
        if response_model is None:
            if task_message and task_message not in conversation:
                conversation.append(task_message)
            return self.call_completions_api(temperature, tools, conversation)
        if max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError("response_model must be a Pydantic model class")
        messages = list(conversation)
        if task_message and task_message not in messages:
            messages.append(task_message)
        tool_messages: list[BHMessage | BHToolMessage] = []
        for attempt in range(max_retries + 1):
            previous_length = len(messages)
            content = self._structured_completion(
                response_model, temperature, tools, messages
            )
            tool_messages.extend(messages[previous_length:])
            try:
                parsed = response_model.model_validate_json(content)
            except ValidationError as exc:
                if attempt == max_retries:
                    raise
                messages.extend(
                    [
                        BHMessage(role=MessageRole.ASSISTANT, content=content),
                        BHMessage(
                            role=MessageRole.USER,
                            content=f"Correct your output to satisfy the schema. Validation errors: {exc}",
                        ),
                    ]
                )
            else:
                return [
                    *tool_messages,
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content=parsed.model_dump_json(),
                        parsed=parsed,
                    ),
                ]
        raise AssertionError("Unreachable")

    def call_completions_api(
        self,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
    ) -> list[BHMessage | BHToolMessage]:
        """Provider hook for a single completion and any ordinary tool results."""
        raise NotImplementedError()

    def stream(
        self,
        task_message: BHMessage | None,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
        printer: Optional["Printer"] = None,  # type: ignore # noqa: F821
    ) -> list[BHMessage | BHToolMessage]:
        raise NotImplementedError()


class BHEmbeddingModel(Model):
    def get_embeddings(self, text: str) -> Sequence[float]:
        raise NotImplementedError()
