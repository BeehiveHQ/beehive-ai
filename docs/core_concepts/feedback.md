# Feedback

Feedback enables Beehive `Invokables` to preserve valuable insights and learnings from past executions, allowing agents to build and refine their knowledge over time.

Feedback can be configured with the following invokable [attributes](/beehive-ai/core_concepts/invokables/#base-attributes):

| Attribute<br>`type` | Description |
| ------------------- | ----------- |
| feedback<br>`bool` | Whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used. |
| feedback_embedder<br>`BHEmbeddingModel | None` | Embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`. |
| feedback_prompt_template<br>`str | None` | Prompt for evaluating the output of an LLM. Used to generate suggestions that can be used to improve the model's output in the future. If `None`, then Beehive will use the default prompt template. Default is `None`. |
| feedback_model<br>`BHChatModel | BaseChatModel` | Language model used to generate feedback for the invokable. If `None`, then default to the Invokable's `model` attribute. |
| feedback_embedding_distance<br>`EmbeddingDistance` | Distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function. |
| n_feedback_results<br>`int` | Amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`. |

When `feedback=True`, here's what happens under the hood:

- When an `Invokable` is invoked with a task, Beehive embeds the task using the embedding model provided in `feedback_embedder`
- Beehive then searches in a vector database ([ChromaDB](https://docs.trychroma.com/)) for similar tasks. The number of similar tasks is determined by `n_feedback_results.`
- Beehive grabs the feedback for those like tasks, concatenates it into a single message, and augments the `Invokable` state with that feedback.
- Beehive then invokes the `Invokable`.

!!! note
    Like the SQLite database powering `history`, the Chroma database containing task feedback lives in `~/.beehive`.

## BHEmbeddingModel

The `feedback_embedder` is a core part of Beehive's feedback process. This model is an instance of the `BHEmbeddingModel` base class.

Like `BHChatModels` `BHEmbeddingModels` are fairly simple. In fact, the attributes are the exact same:

| Attribute<br>`type` | Description |
| ------------------- | ----------- |
| model<br>`str` | Open AI embedding model. Default is `text-embedding-3-small`. |
| **model_config | Additional keyword arguments accepted by the LLM provider's client. |

Embedding models can be used via the `get_embeddings`:

```python
class BHEmbeddingModel:
    def get_embeddings(self, text: str) -> Sequence[float]:
        ...
```

This method takes as input some text and returns a sequence of floats.

### OpenAIEmbedder

Beehive's `OpenAIEmbedder` class provides a thin wrapper over the `openai.OpenAI` Python client. The available embedding models can be found [here](https://platform.openai.com/docs/guides/embeddings/embedding-models).

```python
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_embedder import OpenAIEmbedder

embedding_model = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key="<your_api_key">,  # keyword argument accepted by openai.OpenAI client
    max_retries=10,  # keyword argument accepted by openai.OpenAI client
)
embeddings = embedding_model.get_embeddings(text="Embed this, please!")
print(embeddings)
# [
#   -0.015081570483744144,
#   -0.005574650131165981,
#   -0.023674558848142624,
#   0.0027788940351456404,
#   -0.013219981454312801,
#   -0.029758447781205177,
#   -0.01786046475172043,
#   -0.010474811308085918,
#   -0.04527169093489647,
#   -0.01892615668475628,
#   0.020275134593248367,
# ...
# ]
```
