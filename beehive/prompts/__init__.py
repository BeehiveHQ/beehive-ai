import shutil
from pathlib import Path

from jinja2 import Template
from pydantic import BaseModel, ConfigDict

from beehive.constants import INTERNAL_FOLDER_PATH

PROMPT_DIR = INTERNAL_FOLDER_PATH / "prompts"
PROMPT_DIR.mkdir(exist_ok=True)


def copy_prompt_to_internal_folder(fname: str) -> None:
    if not (PROMPT_DIR / fname).is_file():
        shutil.copy(Path(__file__).parent / fname, PROMPT_DIR / fname)
    return None


def load_template(template_path: Path) -> str:
    copy_prompt_to_internal_folder(template_path.name)
    with open(template_path, "r") as f:
        template_string = f.read()
    return template_string


def prompt_generator(prompt_template: Path | str, **kwargs) -> str:
    if isinstance(prompt_template, Path):
        with open(prompt_template) as f:
            template = Template(f.read())
    else:
        template = Template(prompt_template)
    rendered_template = template.render(**kwargs)
    assert isinstance(rendered_template, str)
    return rendered_template


class BHPrompt(BaseModel):
    model_config = ConfigDict(extra="allow")
    template: str | Path

    def render(self) -> str:
        return prompt_generator(self.template, **self.model_dump(exclude={"template"}))


CONTEXT_PROMPT_FULL: str = load_template(PROMPT_DIR / "context_prompt_full.txt")


class FullContextPrompt(BHPrompt):
    template: str = CONTEXT_PROMPT_FULL
    context: str


CONTEXT_PROMPT_CONCISE: str = load_template(PROMPT_DIR / "context_prompt_concise.txt")


class ConciseContextPrompt(BHPrompt):
    template: str = CONTEXT_PROMPT_CONCISE
    context: str


DEBATE_JUDGE_PROMPT: str = load_template(PROMPT_DIR / "debate_judge_prompt.txt")


class DebateJudgePrompt(BHPrompt):
    template: str = DEBATE_JUDGE_PROMPT
    num_agents: str
    task: str
    responses: str


EVALUATION_PROMPT: str = load_template(PROMPT_DIR / "evaluation_prompt.txt")


class EvaluationPrompt(BHPrompt):
    template: str = EVALUATION_PROMPT
    task: str
    output: str
    format_instructions: str


FEEDBACK_PROMPT: str = load_template(PROMPT_DIR / "feedback_prompt.txt")


class FeedbackPrompt(BHPrompt):
    template: str = FEEDBACK_PROMPT
    feedback: str


LONG_FORM_DEBATE_PROMPT: str = load_template(PROMPT_DIR / "long_form_debate_prompt.txt")


class LongFormDebatePrompt(BHPrompt):
    template: str = LONG_FORM_DEBATE_PROMPT
    other_agent_responses: str
    task: str


MODEL_ERROR_PROMPT: str = load_template(PROMPT_DIR / "model_error_prompt.txt")


class ModelErrorPrompt(BHPrompt):
    template: str = MODEL_ERROR_PROMPT
    error: str


ROUTER_QUESTION_PROMPT: str = load_template(PROMPT_DIR / "router_question_prompt.txt")


class RouterQuestionPrompt(BHPrompt):
    template: str = ROUTER_QUESTION_PROMPT
    question: str
    agents: str
    agent_descriptions: str


ROUTER_PROMPTING_PROMPT: str = load_template(PROMPT_DIR / "router_prompting_prompt.txt")


class RouterPromptingPrompt(BHPrompt):
    template: str = ROUTER_PROMPTING_PROMPT
    backstory: str
    task: str
    context: str | None
    format_instructions: str


ROUTER_ROUTING_PROMPT: str = load_template(PROMPT_DIR / "router_routing_prompt.txt")


class RouterRoutingPrompt(BHPrompt):
    template: str = ROUTER_ROUTING_PROMPT
    agents: str
    agent_descriptions: str
    task: str
    context: str | None
    additional_instructions: str
    format_instructions: str


SHORT_FORM_DEBATE_PROMPT: str = load_template(
    PROMPT_DIR / "short_form_debate_prompt.txt"
)


class ShortFormDebatePrompt(BHPrompt):
    template: str = SHORT_FORM_DEBATE_PROMPT
    other_agent_responses: str
    task: str


SYNTHESIZER_PROMPT: str = load_template(PROMPT_DIR / "synthesizer_prompt.txt")


class SynthesizerPrompt(BHPrompt):
    template: str = SYNTHESIZER_PROMPT
    num_agents: str
    task: str
    responses: str


ASK_QUESTION_PROMPT: str = load_template(PROMPT_DIR / "ask_question_prompt.txt")


class AskQuestionPrompt(BHPrompt):
    template: str = ASK_QUESTION_PROMPT
    agent_names: str
    agent_descriptions: str
    format_instructions: str


DEBATE_JUDGE_SUMMARY_PROMPT: str = load_template(
    PROMPT_DIR / "debate_judge_summary_prompt.txt"
)


class DebateJudgeSummaryPrompt(BHPrompt):
    template: str = DEBATE_JUDGE_SUMMARY_PROMPT
    num_agents: str
    num_rounds: str
    task: str
    final_answer: str


ENSEMBLE_FUNC_SUMMARY_PROMPT: str = load_template(
    PROMPT_DIR / "ensemble_func_summary_prompt.txt"
)


class EnsembleFuncSummaryPrompt(BHPrompt):
    template: str = ENSEMBLE_FUNC_SUMMARY_PROMPT
    num_agents: str
    task: str
    final_answer: str


ENSEMBLE_LLM_SUMMARY_PROMPT: str = load_template(
    PROMPT_DIR / "ensemble_llm_summary_prompt.txt"
)


class EnsembleLLMSummaryPrompt(BHPrompt):
    template: str = ENSEMBLE_LLM_SUMMARY_PROMPT
    num_agents: str
    task: str
    final_answer: str
