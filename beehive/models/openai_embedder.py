from pydantic import Field, PrivateAttr

from beehive.models.base import BHEmbeddingModel

# OpenAI imports
try:
    from openai import OpenAI

# do nothing — if the user tries to instantiate an OpenAI model, they'll receive an
# error indicating that they need to pip install additional packages.
except ImportError:
    pass


# Logger
import logging

logger = logging.getLogger(__file__)


class OpenAIEmbedder(BHEmbeddingModel):
    model: str = Field(
        description="Open AI embedding model. Default is `text-embedding-3-small`.",
        default="text-embedding-3-small",
    )
    _client: OpenAI = PrivateAttr()

    def _create_client(self, **client_kwargs) -> OpenAI:
        return OpenAI(**client_kwargs)

    def get_embeddings(self, text: str) -> list[float]:
        text = text.replace("\n", " ")
        return (
            self._client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
