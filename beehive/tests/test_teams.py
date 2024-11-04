import random
from typing import Literal
from unittest import mock

import pytest
from langchain_openai.chat_models import ChatOpenAI
from pydantic import BaseModel

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Feedback
from beehive.invokable.debate import BeehiveDebateTeam
from beehive.invokable.ensemble import BeehiveEnsemble
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.invokable.team import _invoke_agent_in_process
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_model import OpenAIModel
from beehive.prompts import (
    DebateJudgeSummaryPrompt,
    EnsembleFuncSummaryPrompt,
    EnsembleLLMSummaryPrompt,
)
from beehive.tests.mocks import (
    MockAsyncResult,
    MockChatCompletion,
    MockOpenAIClient,
    MockPrinter,
)


@pytest.fixture(scope="module")
def test_model():
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        model = OpenAIModel(model="gpt-3.5-turbo")
        yield model


@pytest.fixture(scope="module")
def test_storage():
    with mock.patch(
        "beehive.invokable.executor.InvokableExecutor._db_storage"
    ) as mocked:
        mocked.add_task.return_value = "some_unique_task_id"
        mocked.add_beehive.return_value = "some_unique_beehive_id"
        mocked.get_model_objects.return_value = "Retrieved model objects!"
        mocked.add_message.return_value = "some_unique_message_id"
        yield mocked


@pytest.fixture(scope="module")
def test_feedback_storage():
    with mock.patch(
        "beehive.invokable.executor.InvokableExecutor._feedback_storage"
    ) as mocked:
        mocked.embed_task_and_record_feedback.return_value = None
        mocked.grab_feedback_from_similar_tasks.return_value = None
        mocked.grab_feedback_for_task.return_value = None
        yield mocked


@pytest.fixture(scope="module")
def test_printer():
    with mock.patch("beehive.invokable.agent.Printer") as mocked_printer:
        mocked_printer._all_beehives = []
        mocked_printer.return_value = MockPrinter()
        yield mocked_printer


def test_debate_validation():
    # Judge model and agent model do not match
    with pytest.raises(ValueError) as cm:
        _ = BeehiveDebateTeam(
            name="BeehiveDebateTeam",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            judge_model=ChatOpenAI(model="gpt-3.5-turbo"),
            num_members=3,
            num_rounds=3,
        )
    assert "`model` and `judge_model` class must match!" in str(cm)

    # Create a debate team with BHChatModel
    debate_team = BeehiveDebateTeam(
        name="BeehiveDebateTeam",
        backstory="You are a helpful AI assistant.",
        model=OpenAIModel(model="gpt-3.5-turbo"),
        judge_model=OpenAIModel(model="gpt-3.5-turbo"),
        num_members=3,
        num_rounds=3,
    )
    assert isinstance(debate_team._judge_agent, BeehiveAgent)
    assert debate_team._agent_list
    for ag in debate_team._agent_list:
        assert isinstance(ag, BeehiveAgent)

    # Create a debate team with Langchain chat model
    debate_team = BeehiveDebateTeam(
        name="BeehiveDebateTeam",
        backstory="You are a helpful AI assistant.",
        model=ChatOpenAI(model="gpt-3.5-turbo"),
        judge_model=ChatOpenAI(model="gpt-3.5-turbo"),
        num_members=3,
        num_rounds=3,
    )
    assert isinstance(debate_team._judge_agent, BeehiveLangchainAgent)
    assert debate_team._agent_list
    for ag in debate_team._agent_list:
        assert isinstance(ag, BeehiveLangchainAgent)


def test_bh_create_async_task():
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()

        debate_team = BeehiveDebateTeam(
            name="BeehiveDebateTeam",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            judge_model=OpenAIModel(model="gpt-3.5-turbo"),
            num_members=3,
            num_rounds=3,
        )

        # All chat models return the message "Hello from our mocked class!" This is
        # incompatible with our `Feedback` type
        with mock.patch(
            "beehive.invokable.executor.InvokableExecutor._feedback_model"
        ) as mocked_fb_model:
            mocked_fb_model.chat.return_value = [
                BHMessage(
                    role=MessageRole.ASSISTANT,
                    content='{"confidence": 5, "suggestions": ["Here is some dumb feedback!"]}',
                )
            ]

            # Invoke agent in the process. We've mocked all chat models,
            messages, feedback = _invoke_agent_in_process(
                state=debate_team.state,
                name=debate_team.name,
                backstory=debate_team.backstory,
                model_cls=OpenAIModel,
                model_kwargs={"model": "gpt-3.5-turbo"},
                temperature=0,
                tools=[],
                task="Test task",
                response_model=None,
                termination_condition=None,
                chat_loop=1,
                retry_limit=100,
                pass_back_model_errors=True,
            )
            assert len(messages) == 1
            assert (
                isinstance(messages[0], BHMessage)
                and messages[0].role == MessageRole.ASSISTANT
                and messages[0].content == "Hello from our mocked class!"
            )
            assert isinstance(feedback, Feedback)
            assert feedback.confidence == 5
            assert (
                len(feedback.suggestions) == 1
                and feedback.suggestions[0] == "Here is some dumb feedback!"
            )


def test_bh_debate_invoke(test_storage, test_feedback_storage, test_printer):
    num_members = 3
    num_rounds = 2

    def _debate_round_prompt(idx: int):
        # New prompt — this will be different depending on the member
        # number.
        template = "These are the solutions to the problem from other agents:\n\n<other_agent_responses>\n{other_agent_responses}\n</other_agent_responses>\n\nBased off the opinion of other agents, can you give an updated response to the task: <task>Example task</task>"
        if idx == 1:
            other = [2, 3]
        elif idx == 2:
            other = [1, 3]
        else:
            other = [1, 2]
        other_agent_responses = "\n\n".join(
            [f"Agent {j}:\nHello from our mocked class!" for j in other]
        )
        new_prompt = template.format(other_agent_responses=other_agent_responses)
        return new_prompt

    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        debate_team = BeehiveDebateTeam(
            name="BeehiveDebateTeam",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            judge_model=OpenAIModel(model="gpt-3.5-turbo"),
            num_members=num_members,
            num_rounds=num_rounds,
        )
        debate_team._db_storage = test_storage

        # Mock result of apply_async
        mock_result = mock.Mock()
        mock_result.ready.return_value = True
        mock_result.get.return_value = (
            [
                BHMessage(
                    role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
                )
            ],
            Feedback(confidence=5, suggestions=["Here is some dumb feedback."]),
        )

        # Mocked team pool
        with mock.patch("beehive.invokable.team.Pool") as mock_pool_init:
            mock_team_pool_instance = mock_pool_init.return_value.__enter__.return_value
            mock_team_pool_instance.apply_async.return_value = mock_result

            # Mocked debate pool
            with mock.patch("beehive.invokable.debate.Pool") as mock_debate_pool_init:
                mock_debate_pool_instance = (
                    mock_debate_pool_init.return_value.__enter__.return_value
                )
                mock_debate_pool_instance.apply_async.return_value = mock_result

                # Invoke
                output = debate_team.invoke(
                    task="Example task", stdout_printer=test_printer
                )

                # Check the arguments for `apply_async`
                team_pool_call_args = mock_team_pool_instance.apply_async.call_args_list
                debate_round_pool_call_args = (
                    mock_debate_pool_instance.apply_async.call_args_list
                )

                # The `_invoke_agents_in_process` function is defined in the team class.
                # For this, we expect three calls, one for each member.
                assert len(team_pool_call_args) == num_members

                # The actual debate rounds happen in the debate team class. For this we
                # expect num_members * num_rounds calls
                assert len(debate_round_pool_call_args) == num_members * num_rounds

                # Arguments. In the initial pool, the agent state should only contain
                # the system message.
                for i, call in enumerate(team_pool_call_args, start=1):
                    args = call.args
                    kwargs = call.kwargs

                    # There should only be one argument — the function
                    assert len(args) == 1
                    assert args[0] == _invoke_agent_in_process

                    # There should be two keyword args — args and callback
                    assert len(kwargs.keys()) == 2
                    assert "args" in kwargs
                    assert "callback" in kwargs
                    assert len(kwargs["args"]) == 13

                    # Check state
                    state = kwargs["args"][0]
                    assert len(state) == 1
                    assert state[0] == BHMessage(
                        role=MessageRole.SYSTEM,
                        content="You are a helpful AI assistant.",
                    )

                    # Name and backstory should match the original agent, though `name`
                    # will have the index appended onto it.
                    assert kwargs["args"][1] == f"{debate_team.name}-{i}"
                    assert kwargs["args"][2] == debate_team.backstory

                # Debate round arguments. Here, the state should contain the initial
                # response + the responses from the previous rounds of debate.
                first_round_debate_args = debate_round_pool_call_args[0:3]
                second_round_debate_args = debate_round_pool_call_args[3:]
                assert (
                    len(first_round_debate_args) == 3
                    and len(second_round_debate_args) == 3
                )

                # For the first round, the state should only contain the messages from
                # the initial (i.e., "round 0") response.
                for i, call in enumerate(first_round_debate_args, start=1):
                    args = call.args
                    kwargs = call.kwargs
                    state = kwargs["args"][0]
                    assert len(state) == 3
                    assert (
                        state[0]
                        == BHMessage(
                            role=MessageRole.SYSTEM,
                            content="You are a helpful AI assistant.",
                        )
                        and state[1]
                        == BHMessage(role=MessageRole.USER, content="Example task")
                        and state[2]
                        == BHMessage(
                            role=MessageRole.ASSISTANT,
                            content="Hello from our mocked class!",
                        )
                    )

                    # New prompt — this will be different depending on the member
                    # number.
                    new_prompt = _debate_round_prompt(i)
                    assert kwargs["args"][7] == new_prompt

                # For the second round, the state should contain the "round 0" response
                # and the "round 1" response.
                for j, call in enumerate(second_round_debate_args, start=1):
                    args = call.args
                    kwargs = call.kwargs
                    state = kwargs["args"][0]
                    assert len(state) == 5
                    assert (
                        state[0]
                        == BHMessage(
                            role=MessageRole.SYSTEM,
                            content="You are a helpful AI assistant.",
                        )
                        and state[1]
                        == BHMessage(role=MessageRole.USER, content="Example task")
                        and state[2]
                        == BHMessage(
                            role=MessageRole.ASSISTANT,
                            content="Hello from our mocked class!",
                        )
                        and state[3]
                        == BHMessage(
                            role=MessageRole.USER, content=_debate_round_prompt(j)
                        )
                        and state[4]
                        == BHMessage(
                            role=MessageRole.ASSISTANT,
                            content="Hello from our mocked class!",
                        )
                    )

                # Output should just be a single message
                assert len(output["messages"]) == 1
                assert output["messages"][0] == BHMessage(
                    role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
                )

            # The individual debater states should contain the messages from all rounds.
            # We already checked the contents of most of these messages above, so no
            # need to re-check.
            for inv in debate_team._agent_list:
                assert len(inv.state) == (
                    1  # system message
                    + 2  # round 0 user message + response
                    + 2  # round 1 user message + response
                    + 2  # round 2 user message + response
                    + 1  # summary message
                )

                # Check the last message
                last_message_content = inv.state[-1].content
                message = DebateJudgeSummaryPrompt(
                    num_agents=str(num_members),
                    num_rounds=str(num_rounds),
                    task="Example task",
                    final_answer="Hello from our mocked class!",
                ).render()
                assert last_message_content == message


def test_ensemble_final_answer_llm(test_storage, test_feedback_storage, test_printer):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        ensemble = BeehiveEnsemble(
            name="TestBeehiveEnsemble",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            num_members=3,
            final_answer_method="llm",
            synthesizer_model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        ensemble._db_storage = test_storage

        # Mock result of apply_async
        mock_result = mock.Mock()
        mock_result.ready.return_value = True
        mock_result.get.return_value = (
            [
                BHMessage(
                    role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
                )
            ],
            Feedback(confidence=5, suggestions=["Here is some dumb feedback."]),
        )

        # Mocked team pool
        with mock.patch("beehive.invokable.team.Pool") as mock_pool_init:
            mock_team_pool_instance = mock_pool_init.return_value.__enter__.return_value
            mock_team_pool_instance.apply_async.return_value = mock_result

            # Invoke
            output = ensemble.invoke(task="Example task", stdout_printer=test_printer)

            # Check the arguments for `apply_async`
            team_pool_call_args = mock_team_pool_instance.apply_async.call_args_list

            # Arguments. In the initial pool, the agent state should only contain
            # the system message.
            for i, call in enumerate(team_pool_call_args, start=1):
                args = call.args
                kwargs = call.kwargs

                # There should only be one argument — the function
                assert len(args) == 1
                assert args[0] == _invoke_agent_in_process

                # There should be two keyword args — args and callback
                assert len(kwargs.keys()) == 2
                assert "args" in kwargs
                assert "callback" in kwargs
                assert len(kwargs["args"]) == 13

                # Check state
                state = kwargs["args"][0]
                assert len(state) == 1
                assert state[0] == BHMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful AI assistant.",
                )

                # Name and backstory should match the original agent, though `name`
                # will have the index appended onto it.
                assert kwargs["args"][1] == f"{ensemble.name}-{i}"
                assert kwargs["args"][2] == ensemble.backstory

            # The synthesizer model is used to return the final output
            assert len(output["messages"]) == 1
            assert output["messages"][0] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Each agent in ensemble should have three messages in their state: the
            # system message, the task, the response, and then the summary message. We
            # check the contents of these messages above, so no need to re-check.
            for inv in ensemble._agent_list:
                assert len(inv.state) == 4

                # Check the last message
                last_message_content = inv.state[-1].content
                message = EnsembleLLMSummaryPrompt(
                    num_agents="3",
                    task="Example task",
                    final_answer="Hello from our mocked class!",
                ).render()
                assert last_message_content == message


def test_ensemble_final_answer_default_similarity(
    test_storage, test_feedback_storage, test_printer
):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        ensemble = BeehiveEnsemble(
            name="TestBeehiveEnsemble",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            num_members=3,
            final_answer_method="similarity",
        )
        ensemble._db_storage = test_storage

        # Mock result of apply_async
        mock_result = mock.Mock()
        mock_result.ready.return_value = True
        mock_result.get.return_value = (
            [
                BHMessage(
                    role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
                )
            ],
            Feedback(confidence=5, suggestions=["Here is some dumb feedback."]),
        )

        # Mocked team pool
        with mock.patch("beehive.invokable.team.Pool") as mock_pool_init:
            mock_team_pool_instance = mock_pool_init.return_value.__enter__.return_value
            mock_team_pool_instance.apply_async.return_value = mock_result

            # Invoke
            output = ensemble.invoke(task="Example task", stdout_printer=test_printer)

            # Check the arguments for `apply_async`
            team_pool_call_args = mock_team_pool_instance.apply_async.call_args_list

            # Arguments. In the initial pool, the agent state should only contain
            # the system message.
            for i, call in enumerate(team_pool_call_args, start=1):
                args = call.args
                kwargs = call.kwargs

                # There should only be one argument — the function
                assert len(args) == 1
                assert args[0] == _invoke_agent_in_process

                # There should be two keyword args — args and callback
                assert len(kwargs.keys()) == 2
                assert "args" in kwargs
                assert "callback" in kwargs
                assert len(kwargs["args"]) == 13

                # Check state
                state = kwargs["args"][0]
                assert len(state) == 1
                assert state[0] == BHMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful AI assistant.",
                )

                # Name and backstory should match the original agent, though `name`
                # will have the index appended onto it.
                assert kwargs["args"][1] == f"{ensemble.name}-{i}"
                assert kwargs["args"][2] == ensemble.backstory

            # The synthesizer model is used to return the final output
            assert len(output["messages"]) == 1
            assert output["messages"][0] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Each agent in ensemble should have three messages in their state: the
            # system message, the task, the response, and then the summary message. We
            # check the contents of these messages above, so no need to re-check.
            for inv in ensemble._agent_list:
                assert len(inv.state) == 4

                # Check the last message
                last_message_content = inv.state[-1].content
                message = EnsembleFuncSummaryPrompt(
                    num_agents="3",
                    task="Example task",
                    final_answer="Hello from our mocked class!",
                ).render()
                assert last_message_content == message


def test_ensemble_final_answer_custom_similarity(
    test_storage, test_feedback_storage, test_printer
):
    def custom_similarity(x1: str, x2: str) -> float:
        return random.random()

    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        ensemble = BeehiveEnsemble(
            name="TestBeehiveEnsemble",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            num_members=3,
            final_answer_method="similarity",
            similarity_score_func=custom_similarity,
        )
        ensemble._db_storage = test_storage

        # Mock result of apply_async
        mock_result = mock.Mock()
        mock_result.ready.return_value = True
        mock_result.get.return_value = (
            [
                BHMessage(
                    role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
                )
            ],
            Feedback(confidence=5, suggestions=["Here is some dumb feedback."]),
        )

        # Mocked team pool
        with mock.patch("beehive.invokable.team.Pool") as mock_pool_init:
            mock_team_pool_instance = mock_pool_init.return_value.__enter__.return_value
            mock_team_pool_instance.apply_async.return_value = mock_result

            # Invoke
            output = ensemble.invoke(task="Example task", stdout_printer=test_printer)

            # Check the arguments for `apply_async`
            team_pool_call_args = mock_team_pool_instance.apply_async.call_args_list

            # Arguments. In the initial pool, the agent state should only contain
            # the system message.
            for i, call in enumerate(team_pool_call_args, start=1):
                args = call.args
                kwargs = call.kwargs

                # There should only be one argument — the function
                assert len(args) == 1
                assert args[0] == _invoke_agent_in_process

                # There should be two keyword args — args and callback
                assert len(kwargs.keys()) == 2
                assert "args" in kwargs
                assert "callback" in kwargs
                assert len(kwargs["args"]) == 13

                # Check state
                state = kwargs["args"][0]
                assert len(state) == 1
                assert state[0] == BHMessage(
                    role=MessageRole.SYSTEM,
                    content="You are a helpful AI assistant.",
                )

                # Name and backstory should match the original agent, though `name`
                # will have the index appended onto it.
                assert kwargs["args"][1] == f"{ensemble.name}-{i}"
                assert kwargs["args"][2] == ensemble.backstory

            # The synthesizer model is used to return the final output
            assert len(output["messages"]) == 1
            assert output["messages"][0] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Each agent in ensemble should have three messages in their state: the
            # system message, the task, the response, and then the summary message. We
            # check the contents of these messages above, so no need to re-check.
            for inv in ensemble._agent_list:
                assert len(inv.state) == 4

                # Check the last message
                last_message_content = inv.state[-1].content
                message = EnsembleFuncSummaryPrompt(
                    num_agents="3",
                    task="Example task",
                    final_answer="Hello from our mocked class!",
                ).render()
                assert last_message_content == message


def test_team_with_response_model(test_storage, test_feedback_storage, test_printer):
    # Test output
    class TestResponseModel(BaseModel):
        title: str
        action: Literal["thought", "observation", "action", "final_answer"]
        next_action: Literal["thought", "observation", "action", "final_answer"] | None
        content: str

    with mock.patch("beehive.models.openai_model.OpenAIModel._client") as mocked_client:
        # Each worker will have chat loop = 3 but will terminate after the second loop.
        response_messages: list[str] = []
        for i in range(1, 4):
            response_messages.append(
                TestResponseModel(
                    title=f"Worker {i}'s thought",
                    action="thought",
                    next_action="action",
                    content=f"This is worker {i}'s first response.",
                ).model_dump_json(),
            )
            response_messages.append(
                TestResponseModel(
                    title=f"Worker {i}'s final answer",
                    action="final_answer",
                    next_action=None,
                    content=f"This is worker {i}'s final answer.",
                ).model_dump_json(),
            )
            response_messages.append(
                '{"confidence": 5, "suggestions": ["This is some silly feedback"]}',
            )
        chat_completion_messages = [MockChatCompletion([x]) for x in response_messages]
        mocked_client.chat.completions.create = mock.Mock(
            side_effect=chat_completion_messages
        )

        # Ensemble
        ensemble = BeehiveEnsemble(
            name="TestEnsembleWithResponseModel",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            num_members=3,
            final_answer_method="similarity",
            response_model=TestResponseModel,
            chat_loop=3,
            termination_condition=lambda x: x.action == "final_answer",
        )
        ensemble._db_storage = test_storage

        with mock.patch("beehive.invokable.team.Pool") as MockPool:
            mock_pool = MockPool.return_value.__enter__.return_value
            mock_pool.apply_async = mock.Mock(
                side_effect=lambda func, args, callback: MockAsyncResult(
                    result=func(*args)
                )
            )
            messages, feedback = ensemble._invoke_agents_in_process(
                task="This is a test task", printer=test_printer
            )
            assert len(messages.keys()) == 3
            for i in range(1, 4):
                assert messages.get(i, None)
                assert len(messages[i]) == 2
                assert messages[i][0] == BHMessage(
                    role=MessageRole.ASSISTANT, content=response_messages[3 * i - 3]
                )
                assert messages[i][1] == BHMessage(
                    role=MessageRole.ASSISTANT, content=response_messages[3 * i - 2]
                )

                assert feedback.get(i, None)
                assert feedback[i] == Feedback(
                    confidence=5, suggestions=["This is some silly feedback"]
                )
