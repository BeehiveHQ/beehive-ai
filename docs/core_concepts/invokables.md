# Invokables

!!! note
    You will **never** need to instantiate this class directly. You should always use one of the child classes.

`Invokables` are a core construct in Beehive. An `Invokable` is anything that uses an LLM in its internal architecture to reason through and execute a user's task.

## Base Attributes

!!! info
    Note that the `Invokable` class is a Pydantic `BaseModel`.

| Attribute<br>`type` | Description |
| ------------------- | ----------- |
| name<br>`str` | The invokable name. |
| backstory<br>`str` | Backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.' |
| model<br>`BHChatModel | BaseChatModel` | Chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`. |
| state<br>`list[BHMessage | BHToolMessage] | list[BaseMessage]` | List of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs. |
| history<br>`bool` | Whether to use previous interactions / messages when responding to the current task. Default is `False`. |
| history_lookback<br>`int` | Number of days worth of previous messages to use for answering the current task. |
| feedback<br>`bool` | Whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used. |
| feedback_embedder<br>`BHEmbeddingModel | None` | Embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`. |
| feedback_model<br>`BHChatModel | BaseChatModel` | Language model used to generate feedback for the invokable. If `None`, then default to the `model` attribute. |
| feedback_embedding_distance<br>`EmbeddingDistance` | Distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function. |
| n_feedback_results<br>`int` | Amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`. |
| color<br>`str` | Color used to represent the invokable in verbose printing. This can be a HEX code, an RGB code, or a standard color supported by the Rich API. See https://rich.readthedocs.io/en/stable/appendix/colors.html for more details. Default is `chartreuse2`. |


## "invoke"  method
In order to have your invokable execute a task, you can use the `invoke` method. You'll see several examples of this throughout the documentation.

| Argument<br>`type` | Description |
| ------------------ | ----------- |
| task<br>`str` | Task to execute. |
| retry_limit<br>`str` | Maximum number of retries before the Invokable returns an error. Default is `100`. |
| pass_back_model_errors<br>`bool` | Boolean controlling whether to pass the contents of an error back to the LLM via a prompt.  Default is `False`. |
| verbose<br>`bool` | Beautify stdout logs with the `rich` package. Default is `True`. |
| context<br>`list[Invokable] | None` | List of Invokables whose state should be treated as context for this invokation. |
| stream<br>`bool` | Stream the output of the agent character-by-character. Default is `False`. |
| stdout_printer<br>`output.printer.Printer | None` | Printer object to handle stdout messages. Default is `None`.

Beehive offers several invokables out-of-the-box:

- `BeehiveAgent`
- `BeehiveLangchainAgent`
- `BeehiveEnsemble`
- `BeehiveDebate`

We'll cover these in detail next.

## `BeehiveAgent`

`BeehiveAgent`s are the most basic type of `Invokable`. They are autonomous units programmed to execute complex tasks by combining *tool usage* and *memory*.

Here are the additional fields supported by the `BeehiveAgent` class.

| Argument<br>`type`                                    | Description                                                                                                                                                                                                                                |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| temperature<br>`int`                                  | Temperature setting for the agent's chat model
| tools<br>`list[Callable[..., Any]]`                   | Functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
| response_model<br>`type[BaseModel] | None`            | Pydantic BaseModel defining the desired schema for the agent's output. When specified, Beehive will prompt the agent to make sure that its responses fit the models's schema. Default is `None`.
| termination_condition<br>`Callable[..., bool] | None` | Condition which, if met, breaks the agent out of the chat loop. This should be a function that takes a `response_model` instance as input. Default is `None`.
| chat_loop<br>`int`                                    | Number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns (e.g., COT, reflection) may require more loops. This should always be used with a `response_model` and a `termination_condition`.
| docstring_format<br>`DocstringFormat | None`          | Docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.

!!! warning
    Note that `tools` is simply a list of functions. These functions should have docstrings and type-hints. Beehive will throw an error if either of these are missing.

```python
from beehive.invokable.agent import BeehiveAgent
from beehive.models.openai_model import OpenAIModel

math_agent = BeehiveAgent(
    name="MathAgent",
    backstory="You are a helpful AI assistant. You specialize in performing complex calculations.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    tools=[],
    history=True,
    feedback=True,
)
math_agent.invoke("What's 2+2?")
```

## `BeehiveLangchainAgent`

`BeehiveLangchainAgents` are similar to `BeehiveAgents`, except they use Langchain-native types internally.

Here are the additional fields supported by the `BeehiveLangchainAgent` class.

| Argument<br>`type`                           | Description                                                                                                                                                                                                                                |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| temperature<br>`int`                         | Temperature setting for the agent's chat model
| tools<br>`list[Callable[..., Any]]`          | Functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
| docstring_format<br>`DocstringFormat | None` | Docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
| config<br>`RunnableConfig | None`            | Langchain Runnable configuration. This is used inside the ChatModel's `invoke` method. Default is `None`.
| stop<br>`list[str]`                          | List of strings on which the model should stop generating.
| **model_kwargs                               | Extra keyword arguments for invoking the Langchain chat model.

```python
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from langchain_openai.chat_models import ChatOpenAI

math_agent = BeehiveLangchainAgent(
    name="MathAgent",
    backstory="You are a helpful AI assistant. You specialize in performing complex calculations.",
    model=ChatOpenAI(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    tools=[],
    history=True,
    feedback=True,
)
math_agent.invoke("What's 2+2?")
```

## `BeehiveEnsemble`

In a `BeehiveEnsemble`, `n` agents are given the same task and produce `n` different responses. These responses are then synthesized together to produce a final answer.

Beehive currently supports two different synthesis methods: an LLM agent or a similarity function. In the former, Beehive creates a new LLM agent whose task is to combine all `n` responses into a better, final response. In the latter, Beehive computes the similarity between all pairs of responses and returns the answer that had the highest cumulative similarity.

Here are the additional fields supported by the `BeehiveEnsemble` class.

| Argument<br>`type`                                        | Description                                                                                                                                                                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| temperature<br>`int`                                      | Temperature setting for the agent's chat model
| tools<br>`list[Callable[..., Any]]`                       | Functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
| docstring_format<br>`DocstringFormat | None`              | Docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
| response_model<br>`type[BaseModel] | None`                | Pydantic BaseModel defining the desired schema for the agent's output. When specified, Beehive will prompt the agent to make sure that its responses fit the models's schema. Default is `None`.
| termination_condition<br>`Callable[..., bool] | None`     | Condition which, if met, breaks the agent out of the chat loop. This should be a function that takes a `response_model` instance as input. Default is `None`.
| chat_loop<br>`int`                                        | Number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns (e.g., COT, reflection) may require more loops. This should always be used with a `response_model` and a `termination_condition`.
| num_members<br>`int`                                      | Number of members on the team.
| final_answer_method<br>`Literal['llm', 'similarity']`     | Method used to obtain the final answer from the agents. Either `llm` or `similarity`. If `llm`, then Beehive will create an agent with the inputted `synthesizer_model` and use that to synthesize the responses from the agents and generate a single, final response. If `similarity`, then Beehive will choose the answer that has the highest cumulative similarity to the other agents.
| synthesizer_model<br>`BHChatModel | BaseChatModel | None` | Model used to synthesize responses from agents and generate a final response. Only necessary if `final_answer_method`='llm'. This class *must* match the `model` class.
| similarity_score_func<br>`Callable[[str, str], float]`    | Function used to compute the similarity score. Only necessary if `final_answer_method`='similarity'. The function must take two string arguments and return a float. If the callable is not specified, then Beehive defaults to the BLEU score from Papineni et al., 2002. Default is `None`.
| **agent_kwargs                                            | Extra keyword arguments for agent instantiation. This is ONLY used for Langchain agents, and this is used for both the member agent and synthesizer agent instantiation.

This was inspired by the work of [Li et. al](https://arxiv.org/pdf/2402.05120).

```python
from beehive.invokable.ensemble import BeehiveEnsemble
from beehive.models.openai_model import OpenAIModel

# Using similarity scores
ensemble_similarity = BeehiveEnsemble(
    name="TestEnsembleSimilarity",
    backstory="You are an expert software engineer.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    num_members=4,
    history=True,
    final_answer_method="similarity",
)
ensemble_similarity.invoke("Write a script that downloads data from S3.")

# Using synthesizer model
ensemble_synthesizer = BeehiveEnsemble(
    name="TestEnsembleSimilarity",
    backstory="You are an expert software engineer.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    num_members=4,
    history=True,
    final_answer_method="llm",
    synthesizer_model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    )
)
ensemble_similarity.invoke("Write a script that uploads data from S3.")
```

## `BeehiveDebateTeam`

In an `BeehiveDebateTeam`, `n` agents are initially given the same task and produce `n` different responses. The agents then "debate" with one another, i.e., they look at the output of the other `n-1` agents and update their own response. This happens over several rounds. Finally, a "judge" (another LLM agent) evaluates all of the responses and chooses the one answer the initial query best.

Here are the additional fields supported by the `BeehiveDebateTeam` class.

| Argument<br>`type`                                        | Description                                                                                                                                                                                                                                |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| temperature<br>`int`                                      | Temperature setting for the agent's chat model
| tools<br>`list[Callable[..., Any]]`                       | Functions that this agent can use to answer questions. These functions are converted to tools that can be intepreted and executed by LLMs. Note that the language model must support tool calling for these tools to be properly invoked.
| response_model<br>`type[BaseModel] | None`                | Pydantic BaseModel defining the desired schema for the agent's output. When specified, Beehive will prompt the agent to make sure that its responses fit the models's schema. Default is `None`.
| termination_condition<br>`Callable[..., bool] | None`     | Condition which, if met, breaks the agent out of the chat loop. This should be a function that takes a `response_model` instance as input. Default is `None`.
| chat_loop<br>`int`                                        | Number of times the model should loop when responding to a task. Usually, this will be 1, but certain prompting patterns (e.g., COT, reflection) may require more loops. This should always be used with a `response_model` and a `termination_condition`.
| docstring_format<br>`DocstringFormat | None`              | Docstring format in functions. Beehive uses these docstrings to convert functions into LLM-compatible tools. If `None`, then Beehive will autodetect the docstring format and parse the arg descriptions. Default is `None`.
| num_members<br>`int`                                      | Number of members on the team.
| num_rounds<br>`int`                                       | Number of debate rounds.
| judge_model<br>`BHChatModel | BaseChatModel`              | Model used to power the judge agent.
| **agent_kwargs                                            | Extra keyword arguments for agent instantiation. This is ONLY used for Langchain agents, and this is used for both the member agent and synthesizer agent instantiation.

This was inspired by the work of [Du et. al](https://arxiv.org/pdf/2305.14325).

```python
from beehive.invokable.debate import BeehiveDebateTeam
from beehive.models.openai_model import OpenAIModel

debaters = BeehiveDebateTeam(
    name="TestDebateTeam",
    backstory="You are a helpful AI assistant.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    num_members=2,
    num_rounds=2,
    judge_model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    history=True,
    feedback=True,
)
debaters.invoke(" A treasure hunter found a buried treasure chest filled with gems. There were 175 diamonds, 35 fewer rubies than diamonds, and twice the number of emeralds than the rubies. How many of the gems were there in the chest?")
```
