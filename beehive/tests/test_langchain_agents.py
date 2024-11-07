from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI

from beehive.invokable.base import Feedback
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_model import OpenAIModel
from beehive.tests.mocks import MockOpenAIClient, MockPrinter


@pytest.fixture(scope="module")
def test_model():
    with mock.patch("langchain_openai.chat_models.base.openai.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        model = ChatOpenAI()
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


def test_lc_agent_invoke(test_model, test_printer):
    test_agent = BeehiveLangchainAgent(
        name="TestLangchainAgent",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )
    messages = test_agent._invoke(task="Tell me a joke!", stdout_printer=test_printer)

    # Confirm output is equivalent to the output of our mocked class
    assert len(messages) == 1
    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Hello from our mocked class!"

    # Check agent's state
    system_message = SystemMessage(
        content="You are a helpful AI assistant.",
    )
    user_message = HumanMessage(
        content="Tell me a joke!",
    )
    for msg in [system_message, user_message]:
        assert msg in test_agent.state

    # Last message is an AI message.
    last_message = test_agent.state[-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "Hello from our mocked class!"


def test_lc_agent_invoke_with_memory_no_feedback(
    test_model: OpenAIModel,
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    # Invoke the agent and check that appropriate functions were called.
    test_agent = BeehiveLangchainAgent(
        name="TestLangchainAgent",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )
    output = test_agent.invoke(task="Tell me a joke!", stdout_printer=test_printer)

    # Output asserts
    assert "task_id" in output
    assert "messages" in output
    assert "printer" in output
    assert output["task_id"] == "some_unique_task_id"
    assert len(output["messages"]) == 1
    output_message = output["messages"][0]
    assert isinstance(output_message, AIMessage)
    assert output_message.content == "Hello from our mocked class!"
    assert output["printer"] == test_printer

    # Database storage methods are called appropriately
    assert test_storage.add_invokable.called
    assert test_storage.add_task.called
    assert test_storage.get_model_objects.called
    assert test_storage.add_message.called
    assert test_storage.add_tool_calls.called

    # Feedback storage methods are NOT called, because the agent's feedback attribute is
    # False
    assert not test_feedback_storage.grab_feedback_from_similar_tasks.called
    assert not test_feedback_storage.embed_task_and_record_feedback.called

    # Printer methods are called appropriately
    assert test_printer.invokable_label_text.called
    assert test_printer.print_invokable_output.called
    assert test_printer.separation_rule.called


def test_lc_agent_invoke_with_memory_feedback(
    test_model: OpenAIModel,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    with mock.patch(
        "beehive.invokable.executor.InvokableExecutor._feedback_model"
    ) as mocked_fb_model:
        mocked_fb_model.chat.return_value = [
            BHMessage(
                role=MessageRole.ASSISTANT,
                content='{"confidence": 5, "suggestions": ["Here is some dumb feedback!"]}',
            )
        ]

        # Invoke the agent and check that appropriate functions were called.
        test_agent = BeehiveLangchainAgent(
            name="TestAgent",
            backstory="You are a helpful AI assistant.",
            model=test_model,
            feedback=True,
        )
        _ = test_agent.invoke(task="Tell me a joke!", stdout_printer=test_printer)

        # Feedback storage methods should now be called.
        assert test_feedback_storage.grab_feedback_from_similar_tasks.called
        assert test_feedback_storage.embed_task_and_record_feedback.called

        # Feedback should reflect our mocked feedback above.
        called_args = (
            test_feedback_storage.embed_task_and_record_feedback.call_args.args
        )
        assert called_args[0] == test_agent
        assert called_args[2] == Feedback(
            confidence=5, suggestions=["Here is some dumb feedback!"]
        )
