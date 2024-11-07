
# Creating your own `Invokable` class

In the [Invokables](/beehive-ai/core_concepts/invokables/#invoke-method) section, we talked about how one can use the `invoke` method to actually execute the invokable. The internal logic for this is determined by the `_invoke` method. In order to create your own invokable, one simply needs to create a class that inherits the `Invokable` base class and implement this `_invoke` method.

Here's what this method looks like. Note the differences between this and `invoke`:

```python
class Invokable(BaseModel):
    ...

    def _invoke(
        self,
        task: str,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[Any]:
        """Invoke the Invokable to execute a task.

        args:
        - `task` (str): task to execute.
        - `retry_limit` (int): maximum number of retries before the Invokable returns an error. Default is `100`.
        - `pass_back_model_errors` (bool): boolean controlling whether to pass the contents of an error back to the LLM via a prompt.  Default is `False`.
        - `verbose` (bool): beautify stdout logs with the `rich` package. Default is `True`.
        - `stream` (bool): stream the output of the agent character-by-character. Default is `False`.
        - `stdout_printer` (`output.printer.Printer` | None): Printer object to handle stdout messages. Default is `None`.

        returns:
        - list[BHMessage | BHToolMessage] | list[BaseMessage] | list[BHStateElt]
        """
        raise NotImplementedError()
```

Creating your own Invokable class is as simple as:

- Creating a class `CustomRAGInvokable` that inherits from the `Invokable` class
- Endowing this class with custom fields that are specific to our RAG implementation. The `Invokable` class is a Pydantic `BaseModel`, so runtime type-checking is automatically enforced.
- Adding some light model validation and field instantiation.
- Implementing the `_invoke` method.

Here's an example of a custom RAG invokable. Note that this invokable assumes that you've already set up a vector store with your chunked document embeddings.

!!! warning
    This is just for illustrative purposes. This has not been tested!

```python
from beehive.invokable.base import Invokable
from beehive.message import BHMessage, BHToolMessage, MessageRole
from chromadb import PersistentClient, Collection
from openai import OpenAI
from pydantic import Field, PrivateAttr, model_validator


class CustomRAGInvokable(Invokable):
    model_config = ConfigDict(extra="allow")

    client: Any = Field(
        description="ChromaDB `PersistentClient` object"
    )
    embedding_model: str = Field(
        description="Embedding model used to create document embeddings.",
        default="text-embedding-3-small",
    )
    n_results: int = Field(
        description="Number of documents to retrieve.",
        default=10,
    )

    # Embedding client and collection containing documents
    _embedding_client: OpenAI = PrivateAttr()
    _document_collection: Collection = PrivateAttr()

    @model_validator(mode="after")
    def define_document_collection(self) -> "CustomRAGInvokable":
        self._embedding_client = OpenAI(**self.model_extra if self.model_extra else {})

        # Type-check the client keyword-argument
        if not isinstance(self.client, PersistentClient):
            raise ValueError("`client` must be a `PersistentClient`!")
        self._document_collection = self.client.get_or_create_collection(
            name="documents", metadata={"hnsw:space": "l2"}
        )
        return self

    def _invoke(
        self,
        task: str,
        context: list["Invokable"] | None = None,
        feedback: BHMessage | None = None,
        retry_limit: int = 100,
        pass_back_model_errors: bool = False,
        verbose: bool = True,
        stream: bool = False,
        stdout_printer: Printer | None = None,
    ) -> list[BHMessage | BHToolMessage]:
        """For this simple RAG invokable, we will:
        - Embed the task
        - Grab the documents most similar to this task
        - Create a prompt for the model
        - Return the output
        """
        task_embeddings = (
            self._embedding_client.embeddings.create(input=[task], model=self.embedding_model)
            .data[0]
            .embedding
        )
        documents = self._document_collection.query(
            query_embeddings=[embeddings],
            n_results=self.n_results,
        )

        # Handle our context and feedback

        # Construct the prompt
        document_context = "\n\n".join(documents["documents"])
        prompt = f"{task}\nHere is some context to help you answer this question:\n<context>{document_context}</context>"
        task_message = BHMessage(
            role=MessageRole.USER,
            content=prompt
        )

        output = self.model.chat(
            input_messages=[task_message],
            temperature=self.temperature,
            tools={},
            conversation=[]
        )

        # Add the model's response to the agent's tate
        self.state.extend(output)

        # Return
        return output
```
