# 🐝 Welcome to Beehive!

[![CI Linux](https://github.com/BeehiveHQ/beehive-ai/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/BeehiveHQ/beehive-ai/actions/workflows/ci-linux.yml/badge.svg)
[![CI MacOS](https://github.com/BeehiveHQ/beehive-ai/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/BeehiveHQ/beehive-ai/actions/workflows/ci-macos.yml)
[![mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Beehive is an open-source framework for building AI agents and enabling these agents to cooperate with one another to solve tasks. This project was heavily inspired by the awesome work at Langgraph, CrewAI, and PyAutogen.

**We're still in the early stages — expect breaking changes to the API.** Any and all feedback is welcome! If you notice a bug or want to suggest an improvement, please open a Github PR.

## Why use Beehive?

In traditional software applications, the chain of actions taken by the application in response to user input is hardcoded. Any "reasoning" that these applications employ (e.g., if the user does "X", do "Y", otherwise do "Z") can be traced to a few lines of code.

On the other hand, agents rely on a language model to decide which actions to take and in what order. Unlike traditional software applications, where the sequence of actions is predefined in the code, the language model itself is the decision-making engine.

Beehive, in particular, is great for rapidly creating complex chat patterns between agents (or invokables, in Beehive nomenclature). This includes:

- Sequential chats
- Hierarchical chats
- Multi-agent collaboration / debates
- Nested patterns

In addition, Beehive shares many features with other popular agentic frameworks:

- Loops and conditionals between agents
- State management
- Streaming support
- Memory / feedback


## Installation

You can install Beehive with `pip`:
```bash
pip install beehive-ai
```

Note that the Python OpenAI client is included in the standard Beehive installation.

## Creating your first Beehive

Let's create the following Beehive:

![LanguageGeneratorBeehive](/docs/images/language_generator_beehive.png)

This simple Beehive instructs two agents to work together to create a new language:

```python
linguist_agent = BeehiveAgent(
    name="Linguist",
    backstory="You are an expert in linguistics. You work alongside another linguist to develop new languages."
    model=OpenAIModel(
        model="gpt-4-turbo",
    ),
)

linguist_critic = BeehiveAgent(
    name="LinguistCritic",
    backstory="You are an expert in linguistics. Specifically, you are great at examining grammatical rules of new languages and suggesting improvements.",
    model=OpenAIModel(
        model="gpt-4-turbo",
    ),
)

beehive = Beehive(
    name="LanguageGeneratorBeehive",
    backstory="You are an expert in creating new languages.",
    model=OpenAIModel(
        model="gpt-4-turbo",
    ),
    execution_process=FixedExecution(route=(linguist_agent >> linguist_critic)),
    chat_loop=2,
    enable_questioning=True,
)
beehive.invoke(
    "Develop a new language using shapes and symbols. After you have developed a comprehensive set of grammar rules, provide some examples of sentences and their representation in the new language.",
    pass_back_model_errors=True
)
```

Note that this Beehive uses two `BeehiveAgents`, which are one of the more basic invokable constructions. You can scale up the complexity by creating Beehives within Beehives ("nesting") or using more complex invokables, e.g., `BeehiveEnsembles` or `BeehiveDebateTeams`.


## Documentation

Please check out the documentation [here](https://beehivehq.github.io/beehive-ai/) to get started!

### Pydantic structured outputs

Pass a Pydantic model class as an agent's existing `response_model`:

```python
from pydantic import BaseModel, Field
from beehive.invokable.agent import BeehiveAgent
from beehive.models.openai_model import OpenAIModel

class Answer(BaseModel):
    summary: str
    confidence: float = Field(ge=0, le=1)

agent = BeehiveAgent(
    name="Analyst",
    model=OpenAIModel(model="gpt-4o-2024-08-06"),
    response_model=Answer,
    structured_output_retries=2,
)
result = agent.invoke("Explain why the sky is blue.")
answer = result["messages"][-1].parsed  # An Answer instance
```

OpenAI uses strict native JSON Schema output by default. In `auto` mode,
an explicit provider rejection of the response format triggers a synthetic
output-tool fallback. Use `structured_output_mode="tool"` for a provider known
to lack native support, or `"native"` to disable fallback. Invalid schemas and
unrelated API errors do not trigger fallback.

Every final response is validated with `model_validate_json`, including custom
Pydantic validators. `structured_output_retries` counts additional attempts
(default: two); exhausted validation errors propagate even when general model
error retries are enabled. A termination condition is only required for multiple
chat loops. Ordinary tools remain available, with a limit of ten tool turns per
attempt. Structured responses are buffered before printing, including with
`stream=True`. JSON content remains available for storage and context;
the validated instance is in `BHMessage.parsed`.

Custom `BHChatModel` providers can implement `_structured_completion` to reuse
the provider-independent validation and retry loop.

For direct model calls, use `model.chat(..., response_model=Answer, max_retries=2)`.
Omit `response_model` for ordinary chat. Both paths share the OpenAI
`call_completions_api` request and message conversion code.
