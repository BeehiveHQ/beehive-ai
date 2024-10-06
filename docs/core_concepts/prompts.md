# Prompts

Beehive's prompts control how Beehive invokes chat models, interprets feedback, and routes conversations. These templates are rendered and stored in `~/.beehive/prompts/` when you create your first invokable. Prompt templates are [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/templates/) templates. Variables are represented via the `{{ ... }}` expression. These variables get populated dynamically at runtime.

You can edit the wording of these prompts by directly modifying the text files this folder! Note that Beehive uses [XML tags](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags) to structure prompts — we encourage you to maintain this style in your modifications.

!!! warning
    If you modify the wording, **do NOT add variables to the template**. Additional variables will not be rendered properly, which may affect your invokable's behavior.

Here is a list of Beehive's internal prompts, separated

## General
The following prompts are used across different kinds of invokables.

| <div style="width:250px">prompt | description |
| ------ | ----------- |
`ask_question_prompt` | used to explicitly allow invokables to ask questions to other invokables. |
`context_prompt_concise` | used to represent context, i.e., messages from previous invokables. This prompt is "concise" — i.e., it excludes information about the format for the context. |
`context_prompt_full` | used to represent context, i.e., messages from previous invokables. This prompt icludes information about the format for the context.  |
`evaluation_prompt` | used evaluate the quality of an invokable's output. |
`feedback_prompt` | used to incorporate feedback to an invokable's task. |
`model_error_prompt` | used when there is an error in the agent's output. For example, if the router's output does not meet a specific format, this prompt is used to inform the router of its mistake and to try again. |

## `Beehive`
The following prompts are used within the `Beehive` class.

| <div style="width:250px">prompt | description |
| ------ | ----------- |
`router_prompting_prompt` | used by a Beehive's router agent to create prompts for invokables in a Beehive. |
`router_question_prompt` | used by a Beehive's router agent to direct a clarification question to invokable best posed to answer the question. This is used if a |
`router_routing_prompt` | used by a Beehive's router agent to direct the conversation to the next appropriate invokable. |


## `BeehiveDebateTeam`
The following prompts are used within the `BeehiveDebateTeam` class.

| <div style="width:250px">prompt | description |
| ------ | ----------- |
`debate_judge_prompt` | used by the `BeehiveDebateTeam`'s `_judge_agent` when judging the debater responses. |
`debate_judge_summary_prompt` | used to represent the final answer from `BeehiveDebateTeam`. After the `_judge_agent` judges the debate rounds, Beehive uses this template to append the final answer to each of the member agents' states. This enables the debaters to build off of previous rounds. |
`long_form_debate_prompt` | used to prompt members of a `BeehiveDebateTeam` to another round of debate. |
`short_form_debate_prompt` | used to prompt members of a `BeehiveDebateTeam` to another round of debate. |

## `BeehiveEnsemble`
The following prompts are used within the `BeehiveEnsemble` class.

| <div style="width:250px">prompt | description |
| ------ | ----------- |
`ensemble_func_summary_prompt` | used to represent the final answer from `BeehiveEnsemble` when the ensemble uses a `final_answer_method='similarity'` to compute the final answer. Beehive uses this template to append the final answer to each of the member agents' states. This enables the debaters to build off of previous rounds. |
`ensemble_llm_summary_prompt` | used to represent the final answer from `BeehiveEnsemble` when the ensemble uses a `final_answer_method='llm'` to create the final answer. Beehive uses this template to append the final answer to each of the member agents' states. This enables the debaters to build off of previous rounds. |
`synthesizer_prompt` | used to synthesize the responses of ensemble members into a single, final answer. |
