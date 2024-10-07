# Memory

Beehive supports both short-term memory and long-term memory.

## Short-term memory

Short-term memory allows `Invokables` to temporarily store recent interactions and outcomes using RAG, enabling agents to recall and utilize information relevant to their current context during the current executions.

This memory is handled by the `state` [attribute](/beehive-ai/core_concepts/invokables/#base-attributes) in Invokables. `state` is a list of messages (e.g., `BHMessage | BHToolMessage` or `BaseMessage` depending on the type of `Invokable`).

Wnen an Invokable is instantiated, `state` is instantiated as an empty list. Whenever a user [invokes](/beehive-ai/core_concepts/invokables/#invoke-method) the invokable, Beehive automatically stores the messages in the `state`. Then, when the user invoke the invokable again, this state is used as context.

For example, for `Invokables` that use the `OpenAIModel`, the state is passed as the `messages` argument in the [chat completions API](https://platform.openai.com/docs/guides/chat-completions).

Here's an example:
```python
joke_agent = BeehiveAgent(
    name="JokeAgent",
    backstory="You are an expert comedian.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    )
)
joke_agent.invoke("Tell me a joke!")
# Why did the scarecrow win an award? Because he was outstanding in his field!

print(joke_agent.state)
# [
#   BHMessage(role=<MessageRole.SYSTEM: 'system'>, content='You are an expert comedian.', tool_calls=[]),
#   BHMessage(role=<MessageRole.USER: 'user'>, content='Tell me a joke!', tool_calls=[]),
#   BHMessage(role=<MessageRole.ASSISTANT: 'assistant'>, content='Why did the scarecrow win an award?\n\nBecause he was outstanding in his field!', tool_calls=[])
# ]

joke_agent.invoke("Tell me a different joke, but use the same subject!")
# Why did the scarecrow become a successful stand-up comedian? Because he was a master at delivering corny jokes!

print(joke_agents.state)
# [
#   BHMessage(role=<MessageRole.SYSTEM: 'system'>, content='You are an expert comedian.', tool_calls=[]),
#   BHMessage(role=<MessageRole.USER: 'user'>, content='Tell me a joke!', tool_calls=[]),
#   BHMessage(role=<MessageRole.ASSISTANT: 'assistant'>, content='Why did the scarecrow win an award?\n\nBecause he was outstanding in his field!', tool_calls=[])
#   BHMessage(role=<MessageRole.USER: 'user'>, content='Tell me a different joke, but use the same subject!', tool_calls=[]),
#   BHMessage(role=<MessageRole.ASSISTANT: 'assistant'>, content='Why did the scarecrow become a successful stand-up comedian?\n\nBecause he was a master at corny jokes!', tool_calls=[])
# ]
```

As you can see, the `state` agent is preserved for the life of the `joke_agent`. Moreover, the output of the first invokation is used as context for the second invokation.

## Long-term memory

Whereas short-term memory only exists for the life of the `Invokable`, Long-term memory enables access invokations from previous days, weeks, and months.

When you create and invoke your first `Invokable`, Beehive creates a memory store in your local filesystem at `~/.beehive`. This memory store contains two files: `beehive.db` and `feedback/`. The latter is used for [feedback](/beehive-ai/core_concepts/feedback), which we'll cover in a bit.

Long-term memory is controlled by two [attributes](/beehive-ai/core_concepts/invokables/#base-attributes): `history`, and `history_lookback`.

When `history=True`, Beehive augments the `Invokable` state with messages from `history_lookback` days before.

Suppose you are invoking an agent daily via some sort of CRON job. Each day, the agent performs a specific task (e.g., summarize today's news). In this case, you may want to set `history=True` and `num_lookback_days=7`, since specific news stories may evolve over time:

```python
# Day 1
news_agent = BeehiveAgent(
    name="NewsAgent",
    backstory="You are an expert reporter that specializes in summarizing major news stories",
    model=OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="<your_api_key>",
    ),
    tools=[
        parse_nyt_news_stories,
        parse_wsj_news_stories,
        parse_ft_news_stories,
    ],  # functions to grab news stories from various sources
    history=True,
    num_lookback_days=7,
)
print(news_agent.state)
# [
#   BHMessage(role=<MessageRole.SYSTEM: 'system'>, content='You are an expert comedian.', tool_calls=[])
# ]

news_agent.invoke(
    task="Summarize the major recent U.S. news stories. Include relevant context for each news story summary (e.g., developments from previous days).",
)
```

```python
# Day 2
news_agent = BeehiveAgent(
    name="NewsAgent",
    backstory="You are an expert reporter that specializes in summarizing major news stories",
    model=OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="<your_api_key>",
    ),
    tools=[
        parse_nyt_news_stories,
        parse_wsj_news_stories,
        parse_ft_news_stories,
    ],  # functions to grab news stories from various sources
    history=True,
    num_lookback_days=7,
)
print(news_agent.state)
# [
#   BHMessage(role=<MessageRole.SYSTEM: 'system'>, content="You are an expert reporter that specializes in summarizing major news stories", tool_calls=[])
#   BHMessage(role=<MessageRole.SYSTEM: 'user'>, content="Summarize the major recent U.S. news stories. Include relevant context for each news story summary (e.g., developments from previous days).", tool_calls=[])
#   BHMessage(role=<MessageRole.ASSISTANT: 'assistant'>, content="Here's what you need to know about today:", tool_calls=["tool_call_id1", "toolcall_id2", ...])
#   BHToolMessage(role="tool", tool_call_id="tool_call_id1", content="In the New York Times, here are the major stories:")
#   BHToolMessage(role="tool", tool_call_id="tool_call_id2", content="In the Wall Street Journal, here are the major stories:")
#   ...
# ]

news_agent.invoke(
    task="Summarize the major recent U.S. news stories. Include relevant context for each news story summary (e.g., developments from previous days).",
)
```

!!! note
    The above is example uses dummy messages / tool calls to convey how long-term memory works. It is for illustrative purposes only.

As you can see, on day 2, the invokables state *includes* the messages and tool calls from the previous day.
