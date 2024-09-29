# Chat Models

## Overview

Chat models are the _engine_ that power all invokable classes. When a user invokes an `Invokable`, the task is passed through to the underlying chat model. The chat model executes the task according to the `Invokable`'s persona, history, and configuration, and then passes the result back to the Invokable.

## Attributes

`BHChatModels` are fairly simple — they are meant to be a *light* wrapper around the underling LLM provider's client:

| Attribute<br>`type` | Description |
| ------------------- | ----------- |
| model<br>`str` | LLM model, e.g., `gpt-3.5-turbo`. |
| **model_config | Additional keyword arguments accepted by the LLM provider's client. |

Chat models can be used via the `chat` or `stream` methods:

```python
class BHChatModel(BaseModel):
    ...
    def chat(
        self,
        task_message: BHMessage | None,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
    ) -> list[BHMessage | BHToolMessage]:
        ...

    def stream(
        self,
        task_message: BHMessage | None,
        temperature: int,
        tools: dict[str, BHTool],
        conversation: list[BHMessage | BHToolMessage],
        printer: Optional["Printer"] = None,
    ) -> list[BHMessage | BHToolMessage]:
        ...
```

These methods have nearly identical definitions. The `stream` method has one additional argument `printer`:

| Argument | Description |
| ---------| ----------- |
| task_message | User message containing the task. If this message does not have a 'user' role, Beehive will throw an error. |
| temperature | Temperature setting for underlying LLM. |
| tools | Any tools to prove the LLM. Note that the language model must support tool calling in order to properly make use of tools.  |
| conversation | List of messages that the LLM should treat as context for the current task.  |
| printer<br>**`stream` only** | `output.printer.Printer` instance. Used to prettify the streamed output. |

The supported chat models are listed below.

## OpenAI
Beehive's `OpenAIModel` class provides a thin wrapper over the `openai.OpenAI` Python client:

```python
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_model import OpenAIModel

openai_chat_model = OpenAIModel(
    model="gpt-3.5-turbo",
    api_key="<your_api_key">,  # keyword argument accepted by openai.OpenAI client
    max_retries=10,  # keyword argument accepted by openai.OpenAI client
)
joke = openai_chat_model.chat(
    input_messages=[
        BHMessage(
            role=MessageRole.USER,
            content="Tell me a joke!"
        )
    ],
    temperature=0,
    tools={},
    conversation=[],
)
print(joke)
```

This will print the following:
```python
[BHMessage(role=<MessageRole.ASSISTANT: 'assistant'>, content="Why couldn't the bicycle stand up by itself?\n\nBecause it was two tired!", tool_calls=[])]
```

!!! warning
    You can certainly chat with `BHChatModels` directly. However, doing so require that you use Beehive-native types. We highly recommend using Invokables instead — these are simpler to work with "out-of-the-box".
