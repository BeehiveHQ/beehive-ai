from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from beehive.message import BHMessage, BHToolMessage
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
    def chat(
        self,
        task_message: BHMessage | None,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
    ) -> list[BHMessage | BHToolMessage]:
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
