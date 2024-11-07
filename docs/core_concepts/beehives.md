# Beehives

Beehives enable different invokables to collaborate with one another to achieve a task.

## Attributes

| Attribute<br>`type` | Description |
| ------------------- | ----------- |
| name<br>`str` | The invokable name. |
| backstory<br>`str` | Backstory for the AI actor. This is used to prompt the AI actor and direct tasks towards it. Default is: 'You are a helpful AI assistant.' |
| model<br>`BHChatModel | BaseChatModel` | Chat model used by the invokable to execute its function. This can be a `BHChatModel` or a Langchain `ChatModel`. |
| state<br>`list[BHMessage | BHToolMessage] | list[BaseMessage]` | List of messages that this actor has seen. This enables the actor to build off of previous conversations / outputs. |
| execution_process<br>`FixedExecution | DynamicExecution` | Execution process, either `FixedExecution` or `DynamicExecution`. If `FixedExecution`, then the Beehive will execute the Invokables in the `FixedExecution.route`. If `DynamicExecution`, then Beehive uses an internal router agent to determine which `Invokable` to act given the previous messages / conversation. |
| enable_questioning<br>`bool` | Enable invokables to ask one another clarifying questions at runtime. |
| history<br>`bool` | Whether to use previous interactions / messages when responding to the current task. Default is `False`. |
| history_lookback<br>`int` | Number of days worth of previous messages to use for answering the current task. |
| feedback<br>`bool` | Whether to use feedback from the invokable's previous interactions. Feedback enables the LLM to improve their responses over time. Note that only feedback from tasks with a similar embedding are used. |
| feedback_embedder<br>`BHEmbeddingModel | None` | Embedding model used to calculate embeddings of tasks. These embeddings are stored in a vector database. When a user prompts the Invokable, the Invokable searches against this vector database using the task embedding. It then takes the suggestions generated for similar, previous tasks and concatenates them to the task prompt. Default is `None`. |
| feedback_prompt_template<br>`str | None` | Prompt for evaluating the output of an LLM. Used to generate suggestions that can be used to improve the model's output in the future. If `None`, then Beehive will use the default prompt template. Default is `None`. |
| feedback_model<br>`BHChatModel | BaseChatModel` | Language model used to generate feedback for the invokable. If `None`, then default to the `model` attribute. |
| feedback_embedding_distance<br>`EmbeddingDistance` | Distance method of the embedding space. See the ChromaDB documentation for more information: https://docs.trychroma.com/guides#changing-the-distance-function. |
| n_feedback_results<br>`int` | Amount of feedback to incorporate into answering the current task. This takes `n` tasks with the most similar embedding to the current one and incorporates their feedback into the Invokable's model. Default is `1`. |
| color<br>`str` | Color used to represent the invokable in verbose printing. This can be a HEX code, an RGB code, or a standard color supported by the Rich API. See https://rich.readthedocs.io/en/stable/appendix/colors.html for more details. Default is `chartreuse2`. |


!!! note
    Unlike `BeehiveAgents`, `BeehiveLangchainAgents`, `BeehiveEnsembles`, and `BeehiveDebateTeams` the `Beehive` class does **NOT** accept any tools. This is because the `Beehive` instance doesn't actually do the work of executing the task. Rather, it enables the invokables specified in its execution process to execute the task by collaborating together. As such, the `Beehive` will not use tools, but the worker invokables may (and often should) be able to.

Notice that a `Beehive` is actually a subclass of the `Invokable` base class. In fact, most of the fields are those we've already seen!

The two additions are the `execution_process` and `enable_questioning` fields. These controls *how* the conversation flows between the invokables. We'll cover these next.

## Execution process

Beehive supports both fixed routing and dynamic routing:

- Fixed routing: the Beehive invokes `Invokables` according to a pre-defined order
- Dynamic routing: the Beehive uses an internal router agent to determine which `Invokable` to act given the previous messages / conversation.

This routing is controlled by the `execution_process` keyword argument, and the value can be either an instance of the `FixedExecution` class or the `DynamicExecution` class:

```python
class FixedExecution(BaseModel):
    route: Route


class DynamicExecution(BaseModel):
    edges: list[Route]
    entrypoint: Invokable
    llm_router_additional_instructions: dict[Invokable, list[str]]
```

The `Route` type represents a path between any two `Invokables`. It is specified via the `>>` bitwise operator.

### FixedExecution

The `FixedExecution` class accepts a single route — this is the route that the Beehive will use when sending the conversation from one invokable to the next.

```python
from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.beehive import Beehive, FixedExecution
from beehive.models.openai_model import OpenAIModel

backstory_template = "{backstory} You are collaborating with other assistants powered by language models. Answer questions to the best of your ability. It is acceptable if you cannot fully complete the task, as other agents will build on your work. Your goal is to provide valuable input that moves the task forward."

math_agent = BeehiveAgent(
    name="MathAgent",
    backstory=backstory_template.format(backstory="You are a helpful AI assistant that specializes in performing complex calculations."),
    model=OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="<your_api_key>"
    )
)

math_qc_agent = BeehiveAgent(
    name="MathQCAgent",
    backstory=backstory_template.format(backstory="You are math critic who specializes in identifying errors in calculations."),
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
)

workflow = Beehive(
    name="CalculationBeehive",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    execution_process=FixedExecution(
        route=(math_agent >> math_qc_agent)
    )
)
output = workflow.invoke(
    task="A treasure hunter found a buried treasure chest filled with gems. There were 175 diamonds, 35 fewer rubies than diamonds, and twice the number of emeralds than the rubies. How many of the gems were there in the chest?",
)
```

As you can see, in our `FixedExecution` configuration, we specified a route that starts at `math_agent` and ends at `math_qc_agent`. Here's what happens under the hood:

- Beehive starts by prompting `MathAgent` because it is using a fixed execution route and `math_agent` is the first agent in this route. Beehive will use the user's original task when prompting the `Researcher agent.
- After `MathAgent` finishes, Beehive will use its output and craft a prompt for the next invokable in the route, namely `MathQCAgent`.

When the Beehive executes, here is what is printed to `stdout`:
```
------------------------------------------------------------------------------------------------------------------------
CalculationBeehive / MathAgent

Calculate the total number of gems in a treasure chest given the following: there are 175 diamonds, 35 fewer rubies than
diamonds, and twice the number of emeralds as rubies. Provide the calculation steps and the final total.

To find the total number of gems in the treasure chest, we need to calculate the number of each type of gem and then sum
them up.

1. **Number of Diamonds**: Given as 175.

2. **Number of Rubies**: There are 35 fewer rubies than diamonds.
   \[
   \text{Number of Rubies} = 175 - 35 = 140
   \]

3. **Number of Emeralds**: There are twice the number of emeralds as rubies.
   \[
   \text{Number of Emeralds} = 2 \times 140 = 280
   \]

Now, add all these together to find the total number of gems:
\[
\text{Total Number of Gems} = \text{Number of Diamonds} + \text{Number of Rubies} + \text{Number of Emeralds} = 175 +
140 + 280
\]
\[
\text{Total Number of Gems} = 595
\]

Thus, the treasure chest contains a total of 595 gems.
------------------------------------------------------------------------------------------------------------------------
CalculationBeehive / Router

Sending conversation to MathQCAgent next!
------------------------------------------------------------------------------------------------------------------------
CalculationBeehive / MathQCAgent

Please review the calculation provided by the previous agent for any potential errors or improvements. The calculation
involves the total number of gems in a treasure chest, with specific counts for diamonds, rubies, and emeralds based on
given conditions. Confirm the accuracy of the arithmetic operations and the logical application of the conditions
described.

Let's review the calculations provided by the previous agent, MathAgent, for accuracy and logical consistency:

1. **Number of Diamonds**: Given as 175.
   - This is straightforward and requires no calculation.

2. **Number of Rubies**: Calculated as 35 fewer than the number of diamonds.
   - Calculation: \(175 - 35 = 140\)
   - This is correctly calculated based on the information provided.

3. **Number of Emeralds**: Calculated as twice the number of rubies.
   - Calculation: \(2 \times 140 = 280\)
   - This is also correctly calculated based on the information provided.

4. **Total Number of Gems**: Sum of diamonds, rubies, and emeralds.
   - Calculation: \(175 + 140 + 280\)
   - Adding these numbers gives \(595\), which matches the total provided by the previous agent.

Based on the review, the arithmetic operations are correctly executed, and the logical application of the conditions (35
fewer rubies than diamonds and twice as many emeralds as rubies) is accurately applied. The total number of gems
calculated as 595 appears to be correct.

No errors or improvements are needed for the calculations provided. The previous agent has accurately computed the total
number of gems in the treasure chest based on the conditions and numbers given.
------------------------------------------------------------------------------------------------------------------------
```

### DynamicExecution

The `DynamicExecution` class accepts three arguments:

- `edges`: a list of Routes (again, specified via the `>>` bitwise operator). These edges tell the Beehive which Invokables can speak to one another.
- `entrypoint`: the Invokable that the Beehive should start at.
- `llm_router_additional_instructions`: a mapping of Invokables to additional instructions to help Beehive's internal routing agent determine the best next actor.

```python
from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.beehive import Beehive, DynamicExecution
from beehive.models.openai_model import OpenAIModel

backstory_template = "{backstory} You are collaborating with other assistants powered by language models. Answer questions to the best of your ability. It is acceptable if you cannot fully complete the task, as other agents will build on your work. Your goal is to provide valuable input that moves the task forward."

researcher = BeehiveAgent(
    name="Researcher",
    backstory=backstory_template.format(backstory="You are a researcher that has access to the web via your tools."),
    model=OpenAIModel(
        model="gpt-4-turbo",
        api_key="<your_api_key>",
    ),
    tools=[tavily_search_tool],  # tool for surfing the web
    history=True,
    feedback=True,
)

chart_generator = BeehiveAgent(
    name="ChartGenerator",
    backstory=backstory_template.format(backstory="You are a data analyst that specializes in writing and executing code to produce visualizations."),
    model=OpenAIModel(
        model="gpt-4-turbo",
        api_key="<your_api_key>",
    ),
    tools=[python_repl],  # tool for executing Python code
    history=True,
    feedback=True,
)
chart_generator_qc = BeehiveAgent(
    name="ChartGeneratorCritic",
    backstory=backstory_template.format(backstory="You are an expert QA engineer who specializes in identifying errors in Python code."),
    model=OpenAIModel(
        model="gpt-4-turbo",
        api_key="<your_api_key>",
    ),
    history=True,
    feedback=True,
)

workflow = Beehive(
    name="CalculationBeehive",
    model=OpenAIModel(
        model="gpt-4-turbo",
        api_key="<your_api_key>",
    ),
    execution_process=DynamicExecution(
        edges=[
            researcher >> chart_generator,
            chart_generator >> chart_generator_qc,
            chart_generator_qc >> researcher,
        ],
        entrypoint=researcher,
        llm_router_additional_instructions={
            chart_generator_qc: [
                "If you notice a `ModuleNotFoundError` in the code, then finish.
            ]
        },
    )
)
output = workflow.invoke(
    "Fetch the UK's GDP over the past 5 years a draw a line graph of the data."
)
```

!!! note
    The `tavily_search_tool` used by the `Researcher` agent is simply a function. Its implementation can be found [here](/beehive-ai/core_concepts/tools).

In this Beehive, we have three agents: a `Researcher` agent that browses the internet, a `ChartGenerator` agent that generates Python code, and a `ChartGeneratorCritic` agent that QAs Python code. Here's what happens under the hood:

- Since the `entrypoint` is set to the `Researcher` agent, that's where Beehive will start. Beehive will use the user's original task when prompting the `Researcher agent.
- After the `Researcher` finishes, Beehive first determines if additional action is needed. If it is, then it looks at the `edges` and determines which other invokables the `Researcher` is allowed to talk to. In this case, `Researcher` is only allowed to speak to the `ChartGenerator` agent, so Beehive's internal router crafts a prompt for the `ChartGenerator` agent.
- Similarly, after the `ChartGenerator` finishes, Beehive once again determines if additional action is needed. If it does, it looks at the user-specified `edges` and finds that the output of `ChartGenerator` can only be piped to the `ChartGeneratorQC` agent. Assuming Beehive's internal router determines that action is required from the `ChartGeneratorQC` agent, it crafts a new prompt for the agent.
- After the `ChartGeneratorQC` agent finishes, Beehive determines if additional action is needed. Note here that, when making this determination, Beehive uses the `llm_router_additional_instructions` that the user specified for `ChartGeneratorQC` — namely, that if this agent noticed a `ModuleNotFoundError` in the code that the `ChartGenerator` produced, then it should finish. Based on this determination, the Beehive will either finish or will send the conversation back to the `Researcher` agent.

When the Beehive executes, here is what is printed to `stdout`:
```
------------------------------------------------------------------------------------------------------------------------
CalculationBeehive / Researcher

Please search the web to find the most recent data on the UK's GDP for the past five years. Compile this data into a
list, including the GDP values and corresponding years. Then, pass this information to the next agent who will create a
line graph based on the data provided.

 The UK GDP for the past five years is as follows:
- 2023: GDP was not provided in the data.
- 2022: GDP data was not available in the provided sources.
- 2021: GDP was $3,141.51 billion with a growth rate of 8.67%.
- 2020: GDP was $2,697.81 billion with a growth rate of -10.36%.
- 2019: GDP was $2,851.41 billion with a growth rate of 1.64%.

Please note that the most recent data available is for 2021.
------------------------------------------------------------------------------------------------------------------------
CalculationBeehive / Router

Sending conversation to ChartGenerator next!
------------------------------------------------------------------------------------------------------------------------
CalculationBeehive / ChartGenerator

Create a line graph of the UK's GDP for the years 2019, 2020, and 2021 using the provided data: 2019: $2,851.41 billion,
2020: $2,697.81 billion, 2021: $3,141.51 billion.

[10/06/24 11:44:22] WARNING  Python REPL can execute arbitrary code. Use with caution.                                                                              python.py:17
 Successfully executed:
```python
import matplotlib.pyplot as plt

# Data for the UK's GDP for the years 2019, 2020, and 2021
years = [2019, 2020, 2021]
gdp_values = [2851.41, 2697.81, 3141.51]

# Creating the line graph
plt.figure(figsize=(10, 5))
plt.plot(years, gdp_values, marker='o')
plt.title('UK GDP from 2019 to 2021')
plt.xlabel('Year')
plt.ylabel('GDP in Billion USD')
plt.grid(True)
plt.xticks(years)
plt.show()
```
Stdout: ModuleNotFoundError("No module named 'matplotlib'")
------------------------------------------------------------------------------------------------------------------------
```

## Questioning

In addition to specifying the execution process (fixed or dynamic), you can also enable invokables to directly ask one another questions via the `enable_questioning` keyword argument. When set to `True`, invokables can ask clarifying questions to any invokable that has acted before them (or any invokable that has a route leading to them, in the case of a `DynamicExecutionProcess`).

!!! note
    Allowing questions is almost certainly possible via the router through some clever prompt engineering. However, `enable_questioning` is an easier way to support this behavior!

Here's an example where the field is enabled:

```python
...
workflow = Beehive(
    name="CalculationBeehive",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    execution_process=DynamicExecution(
        edges=[
            researcher >> chart_generator,
            chart_generator >> chart_generator_qc,
            chart_generator_qc >> researcher,
        ],
        entrypoint=researcher,
        llm_router_additional_instructions={
            chart_generator_qc: [
                "If you notice a `ModuleNotFoundError` in the code, then finish.
            ]
        },
    ),
    enable_questioning=True,
)
```

When the `ChartGenerator` acts after the `Researcher` agent, it can ask the `Researcher` agent clarifying questions about its output.

!!! note
    Questioning (and in general, direction interaction between child invokables) is more representative of human conversation. We plan on extending invokable-to-invokable direct conversation in future versions of Beehive. If you have any ideas, please request a feature or open a PR!

## Increasing complexity

The examples above represent conversations between different `BeehiveAgents`. However, `Beehives` support conversations between *any* kind of `Invokable`, including other `Beehives`! This is where the real power of Beehive becomes apparent — using nested invokable types allows users to quickly spin up complex conversation structures. Let's look at some examples.

### Nested Beehive
In the above chart generation example, we leave it up to our LLM router to determine whether or not to QC the code produced by `ChartGenerator`. What if we wanted to *force* this conversation between the `ChartGenerator` and `ChartGeneratorCritic` to take place prior to sending the conversation back to the `Researcher`? In that case, we could use a nested Beehive!

Recall the edges / routes in a Beehive accept any class that inherits the `Invokable` base class. This *includes* other `Beehives`!

Here's what that would look like:

```python
from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.beehive import Beehive, DynamicExecution
from beehive.models.openai_model import OpenAIModel

researcher = BeehiveAgent(
    name="Researcher",
    backstory="You are a researcher that has access to the web via your tools."
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    tools=[tavily_search_tool],  # tool for surfing the web
    history=True,
    feedback=True,
)

# Separate Beehive for working creating a chart. This force the ChartGenerator to always
# talk to the ChartGeneratorCritic after producing some code.
chart_generator = BeehiveAgent(
    name="ChartGenerator",
    backstory="You are a data analyst that specializes in writing and executing code to produce visualizations.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    tools=[python_repl],  # tool for executing Python code
    history=True,
    feedback=True,
)
chart_generator_qc = BeehiveAgent(
    name="ChartGeneratorCritic",
    backstory="You are an expert QA engineer who specializes in identifying errors in Python code.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    history=True,
    feedback=True,
)
chart_generator_workflow = Beehive(
    name="ChartGeneratorWorkflow",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    execution_process=FixedExecution(route=(chart_generator >> chart_generator_qc)),
    history=True,
    feedback=True,
)

workflow = Beehive(
    name="CalculationBeehive",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    execution_process=DynamicExecution(
        edges=[
            researcher >> chart_generator_workflow,
            chart_generator_workflow >> researcher,
        ],
        entrypoint=researcher,
        llm_router_additional_instructions={
            chart_generator_workflow: [
                "If you notice a `ModuleNotFoundError` in the code, then finish."
            ]
        },
    )
)
output = workflow.invoke(
    "Fetch the UK's GDP over the past 5 years a draw a line graph of the data."
)
```

Note that our `chart_generator_workflow` is a `Beehive` with a `FixedExecution` process. This forces the conversation to pass from `ChartGenerator` to `ChartGeneratorCritic` whenever the Beehive is invoked.

Here's what is printed to `stdout`:
??? example "Nested Beehive"
    ````
    ------------------------------------------------------------------------------------------------------------------------
    CalculationBeehive / Researcher

    Fetch the UK's GDP over the past 5 years a draw a line graph of the data.

    The UK GDP growth rate in the third quarter of 2023 was 0.30 percent year-on-year, with the economy expanding at a
    slower pace than initially estimated. The GDP Growth Rate in the United Kingdom is expected to be 0.20 percent by the
    end of the current quarter.
    ------------------------------------------------------------------------------------------------------------------------
    CalculationBeehive / Router

    Sending conversation to ChartGeneratorWorkflow next!
    ------------------------------------------------------------------------------------------------------------------------
    CalculationBeehive / ChartGeneratorWorkflow / ChartGenerator

    Generate a line graph of the UK's GDP over the past 5 years.

    [09/20/24 08:52:48] WARNING  Python REPL can execute arbitrary code. Use with caution.                                                                                    python.py:17
    Successfully executed:
    ```python
    import matplotlib.pyplot as plt

    # Data for the UK's GDP over the past 5 years
    years = [2018, 2019, 2020, 2021, 2022]
    gdp_values = [2.855, 2.829, 2.743, 2.678, 2.715]  # in trillion USD

    # Create a line graph
    plt.figure(figsize=(10, 6))
    plt.plot(years, gdp_values, marker='o', color='b', linestyle='-', linewidth=2)
    plt.title('UK GDP Over the Past 5 Years')
    plt.xlabel('Year')
    plt.ylabel('GDP in Trillion USD')
    plt.grid(True)
    plt.show()
    ```
    Stdout: ModuleNotFoundError("No module named 'matplotlib'")
    ------------------------------------------------------------------------------------------------------------------------
    CalculationBeehive / ChartGeneratorWorkflow / Router

    Sending conversation to ChartGeneratorCritic next!
    ------------------------------------------------------------------------------------------------------------------------
    CalculationBeehive / ChartGeneratorWorkflow / ChartGeneratorCritic

    Update the Python code to include error handling for potential ModuleNotFoundError when importing matplotlib. If the
    import fails, provide a message indicating that matplotlib is required for generating the graph. Ensure that the code
    gracefully handles this error scenario.

    To handle the potential `ModuleNotFoundError` when importing `matplotlib`, you can use a `try-except` block to catch the
    error and provide a custom message. Here's an updated version of the Python code with error handling for the
    `matplotlib` import:

    ```python
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Error: Matplotlib is required for generating the graph. Please install matplotlib to proceed.")
    else:
        # Data for the UK's GDP over the past 5 years
        years = [2018, 2019, 2020, 2021, 2022]
        gdp_values = [2.855, 2.829, 2.743, 2.678, 2.715]  # in trillion USD

        # Create a line graph
        plt.figure(figsize=(10, 6))
        plt.plot(years, gdp_values, marker='o', color='b', linestyle='-', linewidth=2)
        plt.title('UK GDP Over the Past 5 Years')
        plt.xlabel('Year')
        plt.ylabel('GDP in Trillion USD')
        plt.grid(True)
        plt.show()
    ```

    In this updated code snippet:
    - We use a `try-except` block to attempt the import of `matplotlib.pyplot`. If the import fails due to a
    `ModuleNotFoundError`, a custom error message is printed.
    - If the import is successful, the code proceeds to create the line graph as before.

    This error handling mechanism ensures that the user is informed about the missing `matplotlib` dependency and gracefully
    handles the situation without causing the script to crash.
    ------------------------------------------------------------------------------------------------------------------------
    ````

Note that the Beehive finishes after `ChartGeneratorCritic`! This is because of our `llm_additional_routing_instructions` — namely, after the `ChartGeneratorWorkflow` finishes executing, we told the router to terminate the Beehive if it noticed a `ModuleNotFoundError`.

!!! note
    When a question is posed to a nested Beehive, the nested Beehive will use its router to determine which of its member invokables would be best suited to answer the question.

### Higher-order invokables

So far, the Beehives we have looked at have just used `BeehiveAgents` (or `BeehiveLangchainAgents`). You can use more complex invokables inside your Beehives as well.

For example, here is a `Beehive` that uses the `BeehiveDebateTeam` as one of its invokables:

```python
# Invokable for planning our essay -- this is a BeehiveDebateTeam, so it will spawn
# multiple agents and that debate it out with one another. Note that these teams
# take longer to invoke than regular agents.
planner = BeehiveDebateTeam(
    num_members=4,
    name="EssayPlanner",
    backstory=(
        "You are an expert essay outliner. You create thoughtful, logical, structured essay outlines."
        " You DO NOT actually write essays, you simply create an outline.",
    ),
    model=OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="<your_api_key>",
    ),
    num_rounds=2,
    debate_length="short",
    judge_model=OpenAIModel(
        model="gpt-3.5-turbo",
        api_key="<your_api_key>",
    ),
)

# Agent for researching topics related to the subjet matter
researcher = BeehiveAgent(
    name="Researcher",
    backstory="You are a researcher that has access to the web via your tools.",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    tools=[tavily_search_tool],
)

# Invokable for writing our essay -- this is a BeehiveEnsemble, so it will spawn
# multiple agents and summarize the various answers. Note that these, like
# BeehiveDebateTeams take longer to invoke than regular agents.
writers = BeehiveEnsemble(
    num_members=4,
    name="WritingEnsemble",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    backstory="You are an expert author. You specialize are writing clear, evidence-based non-fiction writing essays.",
    final_answer_method="llm",
    synthesizer_model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    )
)

# Use a FixedExecution for now.
essay_workflow = Beehive(
    name="EssayWorkflow",
    model=OpenAIModel(
        model="gpt-3.5-turbo-0125",
        api_key="<your_api_key>",
    ),
    execution_process=FixedExecution(route=planner >> new_researcher >> writers),
)
essay_workflow.invoke((
    "Write a short, 1-paragraph essay analyzing the impact of social media on political discourse"
    " in the 21st century, providing specific examples and supporting evidence from credible sources."
))
```
??? example "Beehive with Ensembles / DebaterTeams"
    ```
    ------------------------------------------------------------------------------------------------------------------------
    EssayWorkflow / EssayPlanner

    Create a detailed essay outline analyzing the impact of social media on political discourse in the 21st century. Include
    sections for introduction, overview of social media's role in shaping political discussions, specific examples of social
    media platforms influencing political narratives, effects on public opinion and polarization, and a conclusion
    summarizing key points. Each section should have bullet points outlining main ideas and supporting evidence from
    credible sources.

    I. Introduction
    - Brief overview of the rise of social media in the 21st century
    - Importance of political discourse in society
    - Thesis statement: Social media has significantly impacted political discourse in the 21st century by shaping
    discussions, influencing narratives, affecting public opinion, and contributing to polarization.

    II. Overview of Social Media's Role in Shaping Political Discussions
    - Definition of political discourse in the context of social media
    - Accessibility and reach of social media platforms
    - Speed of information dissemination
    - Ability to engage with diverse audiences
    - Credibility concerns and misinformation (Sources: Pew Research Center, Harvard Kennedy School Misinformation Review)

    III. Specific Examples of Social Media Platforms Influencing Political Narratives
    A. Twitter
    1. Hashtags and trending topics
    2. Direct communication from political figures
    B. Facebook
    1. Targeted advertising and political campaigns
    2. Spread of fake news and echo chambers
    C. Instagram
    1. Visual storytelling and political activism
    2. Influencer endorsements and political messaging

    IV. Effects on Public Opinion and Polarization
    - Echo chambers and filter bubbles
    - Amplification of extreme viewpoints
    - Disinformation campaigns and foreign interference
    - Increased engagement and political activism (Source: Journal of Communication)

    V. Conclusion
    - Recap of social media's impact on political discourse
    - Call for critical thinking and media literacy
    - Suggestions for regulating social media platforms to promote healthy political discussions (Sources: Brookings
    Institution, Journal of Communication)

    In the final analysis, it is evident that social media has played a pivotal role in shaping political discourse in the
    21st century. The accessibility, speed of information dissemination, and ability to engage diverse audiences have
    transformed how political discussions unfold. However, concerns regarding credibility, misinformation, and the
    amplification of extreme viewpoints highlight the need for critical thinking and media literacy among users. Regulation
    of social media platforms is essential to foster healthy political discussions and mitigate the negative impacts of
    polarization and misinformation.
    ------------------------------------------------------------------------------------------------------------------------
    EssayWorkflow / Router

    Sending conversation to Researcher next!
    ------------------------------------------------------------------------------------------------------------------------
    EssayWorkflow / Researcher

    Research and compile specific examples of social media platforms influencing political narratives in the 21st century.
    Utilize credible sources to gather evidence supporting the impact of social media on political discourse. Focus on the
    role of platforms like Twitter, Facebook, and Instagram in shaping political discussions.

    Twitter has had a significant impact on political discourse in the 21st century, contributing to social and political
    polarization, public policies, and electoral outcomes. While it provides a vibrant online space for political
    discussions, it has also been associated with fostering a culture of hostility and incivility, with users engaging in
    insulting or attacking behavior with little fear of consequences. Twitter has become a platform for the dissemination of
    political messages charged with polarity, influencing how politicians, governments, and individuals interact and engage
    with one another. The impact of Facebook on political discourse in the 21st century has been significant, as evidenced
    by studies showing that exposure to political information on Facebook can increase issue salience among participants,
    particularly those with low political interest. Social media, including Facebook, has transformed the dynamics of
    political activism by facilitating rapid dissemination of information and mobilization of individuals, revolutionizing
    how movements form and influence societal change. Social media platforms have reshaped how individuals engage with
    political issues, voice their opinions, and mobilize for change, highlighting the profound impact of social media on
    political discourse in shaping public opinion and influencing elections. The impact of Instagram on political discourse
    in the 21st century has been significant, with social media platforms like Instagram playing a crucial role in shaping
    societal trends and political interactions. Social media has altered how politicians, governments, and individuals
    engage with one another, influencing the nature of political discourse. However, it is important to note that social
    media platforms can also contribute to political polarization and the spread of misinformation, which can hinder
    constructive dialogue and debate.
    ------------------------------------------------------------------------------------------------------------------------
    EssayWorkflow / Router

    Sending conversation to WritingEnsemble next!
    ------------------------------------------------------------------------------------------------------------------------
    EssayWorkflow / WritingEnsemble

    Write a concise, evidence-based essay analyzing the impact of social media on political discourse in the 21st century.
    Begin by introducing the pervasive role of social media in modern society and how it has transformed communication and
    information dissemination. Explore both the positive effects, such as facilitating political engagement and access to
    diverse perspectives, and negative effects, like the spread of misinformation and polarization. Provide specific
    examples of platforms like Twitter contributing to social and political polarization, Facebook increasing issue salience
    among users, and Instagram shaping societal trends. Support your analysis with credible sources and data to illustrate
    the profound influence of social media on political discourse.

    **Title: The Impact of Social Media on Political Discourse in the 21st Century**

    **Introduction**
    Social media has revolutionized communication and information dissemination in modern society, profoundly influencing
    political discourse in the 21st century. It has redefined how individuals engage with political issues, express their
    opinions, and mobilize for change.

    **Positive Effects of Social Media on Political Discourse**
    - **Facilitation of Political Engagement**: Social media platforms have empowered individuals to participate in
    political discussions and activism, exemplified by movements like #BlackLivesMatter and #MeToo.
    - **Access to Diverse Perspectives**: Social media provides access to a wide array of political viewpoints and news
    sources, fostering critical thinking and a more informed citizenry.

    **Negative Effects of Social Media on Political Discourse**
    - **Spread of Misinformation**: Social media's role in disseminating fake news and misinformation has influenced public
    perception and decision-making processes.
    - **Polarization and Echo Chambers**: Algorithms on social media platforms contribute to the formation of echo chambers,
    reinforcing existing beliefs and exacerbating political polarization.

    **Specific Examples of Social Media Platforms Impacting Political Discourse**
    - **Twitter**: Known for fostering social and political polarization, Twitter hosts interactions that can amplify
    societal divisions and influence public opinion.
    - **Facebook**: Exposure to political information on Facebook can heighten issue salience among users, particularly
    those with low political interest, transforming political activism and mobilization.
    - **Instagram**: While shaping societal trends and political interactions, Instagram's influence on political discourse
    can also contribute to polarization and the spread of misinformation.

    **Conclusion**
    The impact of social media on political discourse is multifaceted, offering opportunities for political engagement and
    access to diverse perspectives while also posing challenges such as misinformation and polarization. Understanding these
    dynamics is crucial for navigating the evolving landscape of digital communication and ensuring responsible engagement
    with social media in shaping public opinion and political outcomes.

    **References**
    - Pew Research Center. (2021). The Role of Social Media in American Politics.
    - Harvard Kennedy School. (2020). Misinformation and Polarization on Social Media.
    - Journal of Communication. (2019). The Influence of Social Media on Political Participation and Civic Engagement.
    ```

This wasn't perfect — in our original task, we instructed our Beehive to write a 1-paragraph essay. The final essay is definitely much longer. To correct this, we could do a number of things:
- Modify our prompts
- Modify our Beehive conversation to QC the essay plan and compositions (e.g., via critic invokables)
