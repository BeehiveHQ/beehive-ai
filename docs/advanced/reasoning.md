# Reasoning

Starting in Beehive `v0.0.2`, `BeehiveAgents` are capable of using *reasoning* to accomplish their tasks.

!!! warning
    Only `BeehiveAgents` are currently capable of reasoning. `BeehiveLangchainAgents` are not.

Reasoning can be accomplished via internal loops, response models, and termination conditions. These are controlled via the following attributes:

- **`response_model`**: a Pydantic `BaseModel` that specifies the schema for the agent's output. This response model enables the agent to take conditional action based on the values of certain fields.
- **`chat_loop`**: number of times the model should loop when responding to a task.
- **`termination_condition`**: condition which, if met, breaks the BeehiveAgent out of the chat loop. This should be a function that takes a `response_model` instance as input.

Here's how they work together in tandem:
```python
from pydantic import BaseModel
from typing import Literal

from beehive.invokable.agent import BeehiveAgent

class ReactResponse(BaseModel):
    step: Literal["thought", "action", "observation", "final_answer"]
    action: Literal["tavily_search_tool"] | None
    action_input: str | None
    content: str

backstory = f"""You are an AI reasoning agent that explains your reasoning step by step.

You have access to the following tools:
<tools>
- tavily_search_tool: Use this if you need search the web for information (e.g. current events).
</tools>

Always response with the following schema
<schema>
{json.dumps(ReactResponse.schema())}
</schema>

In addition, make sure you adhere to the following rules:
- "thought" steps should ALWAYS precede "action" steps
- "observation" steps should ALWAYS follow "action" steps. During the "observation" step, you should determine if the previous thoughts and actions have sufficiently answered the user question. The content for these steps should be either "Yes – the user question has been fully answered" or "No - the user question has not been fully answered."
- **ALWAYS RESPOND WITH ONE STEP PER RESPONSE**. This includes "observation" steps.

Here are some examples of how you should respond to user questions:
<examples>
What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?

{{"step": "thought", "action" None, "action_input" None, "content": "I need to search Colorado orogeny, find the area that the eastern sector
of the Colorado orogeny extends into, then find the elevation range of the area."}}

{{"step": "action", "action" "tavily_search_tool", "action_input": "What is the Colorado "orogeny?", "content": "The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas."}}

{{"step": "observation", "action" None, "action_input": None, "content": "No - the user question has not been fully answered."}}

{{"step":"thought", "action" None, "action_input" None, "content": "It does not mention the eastern sector. So I need to look up eastern sector."}}

{{"step": "action", "action" "tavily_search_tool", "action_input": "What is the eastern sector of Colorado "orogeny?", "content": "The eastern sector extends into the High Plains and is called the Central Plains orogeny."}}

{{"step": "observation", "action" None, "action_input": None, "content": "No - the user question has not been fully answered."}}

{{"step":"thought", "action" None, "action_input" None, "content": "The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range."}}

{{"step":"action", "action" "tavily_search_tool", "action_input" "What is the elevation range of the High Plains region of the Colorado orogeny?", "content": "The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130m)."}}

{{"step": "observation", "action" None, "action_input": None, "content": "Yes - the user question has been fully answered."}}

{{"step": "final_answer", "action" None, "action_input": None, "content": "The elevation range of the eastern sector of the Colorado orogeny is 1,800 to 7.000 ft."}}
</examples>

Remember, **ALWAYS RESPOND WITH ONE STEP PER RESPONSE**.
"""

reasoning_agent = BeehiveAgent(
    name="ReasoningAgent",
    backstory=backstory,
    model=OpenAIModel(
        model="gpt-4-turbo",
    ),
    tools=[tavily_search_tool],
    chat_loop=10,
    response_model=ReactResponse,
    termination_condition=lambda x: x.step == "final_answer"
)
reasoning_agent.invoke(
    "Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?",
    pass_back_model_errors=True
)
```
!!! note
    LLMs have a hard time with consistently formatting their responses as a valid JSON. For example, sometimes will respond with multiple JSONs separated by new-line characters. For this reason, we recommend setting `pass_back_model_errors` to `True`. Note that this will increase the runtime and increase token usage, so craft your prompt to try and limit these errors.

Here's what the agent response with:
```stdout
------------------------------------------------------------------------------------------------------------------------
ReasoningAgent

Aside from the Apple Remote, what other devices can control the program Apple Remote was originally designed to interact with?

{"step": "thought", "action": null, "action_input": null, "content": "To answer this question, I need to identify the
program or software that the Apple Remote was originally designed to interact with. Then, I can determine what other
devices can control that program."}

{"step": "action", "action": "tavily_search_tool", "action_input": "What program was the Apple Remote originally
designed to interact with?", "content": "Searching for the program that the Apple Remote was originally designed to
interact with."}

The Apple Remote was originally designed to interact with the Front Row media center program.

{"step": "observation", "action": null, "action_input": null, "content": "No - the user question has not been fully
answered."}

{"step": "thought", "action": null, "action_input": null, "content": "Now that I know the Apple Remote was designed to
interact with the Front Row media center program, I need to find out what other devices can control Front Row."}

{"step": "action", "action": "tavily_search_tool", "action_input": "What devices can control the Front Row program on
Apple computers?", "content": "Searching for devices that can control the Front Row program on Apple computers."}

The Front Row program on Apple computers can be controlled by an Apple Remote or the keyboard function keys.

{"step": "observation", "action": null, "action_input": null, "content": "Yes – the user question has been fully
answered."}

{"step": "final_answer", "action": null, "action_input": null, "content": "Aside from the Apple Remote, the Front Row
program can also be controlled using the keyboard function keys on Apple computers."}
------------------------------------------------------------------------------------------------------------------------
```

Here's what's happening under the hood:

- The `ReasoningAgent` enters an invokation loop that last 10 rounds. During the first round, it is fed the user's initial task. During subsequent rounds, it simply relies on its `state` (i.e., system message, initial user task, and all previous responses).
- During each round of the loop, Beehive casts the agent's response into an instance of the `response_model`.
    - If there is an error and `pass_back_model_errors=True`, then Beehive wraps the traceback in a new system message and passes it back to the agent.
    - If there is an error and `pass_back_model_errors=False`, then the agent terminates and raises the error.
    - Note that Pydantic `ValidationErrors` or a `JSONDecodeErrors` are handled slightly differently. For these errors, Beehive provides a bit more contextual information to help the agent better craft their response.
- Once the `response_model` instance has been created, Beehive checks whether the `termination_condition` has been met. If it has, then the Beehive tells the agent to exit the invokation loop. If not, then Beehive tells the agent to continue until it has responded `chat_loop` times.

In the above example, the `ReasoningAgent` looped 7 times. In its last message, the termination condition was met (i.e., `"step": "final_answer"`), so the agent exited the loop.
