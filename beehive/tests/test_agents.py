import json
from typing import Literal
from unittest import mock

import pytest
from pydantic import BaseModel

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Context, Feedback, Invokable
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_model import OpenAIModel
from beehive.tests.mocks import MockChatCompletion, MockOpenAIClient, MockPrinter


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


def test_invokable_context(test_model):
    inv1 = Invokable(
        name="TestInvokable",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )
    inv2 = Invokable(
        name="TestInvokable2",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )

    # Add some messages to inv2's state
    inv2.state.append(
        BHMessage(
            role=MessageRole.USER,
            content="This is an example user task for invokable 2.",
        )
    )
    inv2.state.append(
        BHMessage(
            role=MessageRole.ASSISTANT,
            content="This is an example agent response from invokable 2.",
        )
    )

    # Construct context dictionary
    inv1_context_dictionary = inv1.construct_context_dictionary([inv2])
    assert "TestInvokable2" in inv1_context_dictionary
    assert isinstance(inv1_context_dictionary["TestInvokable2"], Context)
    expected_context = {
        "TestInvokable2": Context(
            agent_backstory=inv2.backstory,
            agent_messages=[
                "This is an example agent response from invokable 2.",
            ],
        )
    }
    assert inv1_context_dictionary == expected_context

    # Invokable 1 doesn't have any context so far. Calling
    # `remove_duplicate_context` should just return the same context dictionary.
    inv1_context_dictionary_no_dup = inv1.remove_duplicate_context(
        inv1_context_dictionary
    )
    assert inv1_context_dictionary_no_dup == inv1_context_dictionary
    assert inv1._context_messages == inv1_context_dictionary

    # Adding the same context dictionary shouldn't do anything.
    inv1_context_dictionary_no_dup_second_try = inv1.remove_duplicate_context(
        inv1_context_dictionary
    )

    # The output of `remove_duplicate_context` is the context for the current
    # invokation. Since we're not adding any new messages, the output should be empty.
    assert inv1_context_dictionary_no_dup_second_try == {}

    # _context_messages attribute should not have changed
    assert inv1._context_messages == inv1_context_dictionary

    # If we add some new messages, the context messages attribute should change
    new_context = {
        "TestInvokable2": Context(
            agent_backstory=inv2.backstory,
            agent_messages=[
                "This is an example agent response from invokable 2.",
                "This is a new message.",
            ],
        ),
        "TestInvokable3": Context(
            agent_backstory="Imaginary invokable backstory.",
            agent_messages=[
                "This is a new message from a new agent.",
            ],
        ),
    }
    inv1_context_dictionary_new = inv1.remove_duplicate_context(new_context)
    expected_context_curr_invokation = {
        "TestInvokable2": Context(
            agent_backstory=inv2.backstory,
            agent_messages=[
                "This is a new message.",
            ],
        ),
        "TestInvokable3": Context(
            agent_backstory="Imaginary invokable backstory.",
            agent_messages=[
                "This is a new message from a new agent.",
            ],
        ),
    }
    assert inv1_context_dictionary_new == expected_context_curr_invokation

    # Full context
    assert inv1._context_messages == new_context


def test_agent_invoke(test_model):
    test_agent = BeehiveAgent(
        name="TestAgent",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )
    messages = test_agent._invoke(task="Tell me a joke!")

    # Confirm output is equivalent to the output of our mocked class
    assert len(messages) == 1
    assert isinstance(messages[0], BHMessage)
    assert messages[0].content == "Hello from our mocked class!"

    # Check agent's state
    system_message = BHMessage(
        role=MessageRole.SYSTEM,
        content="You are a helpful AI assistant.",
        tool_calls=[],
    )
    user_message = BHMessage(
        role=MessageRole.USER, content="Tell me a joke!", tool_calls=[]
    )
    response_message = BHMessage(
        role=MessageRole.ASSISTANT,
        content="Hello from our mocked class!",
        tool_calls=[],
    )
    for msg in [system_message, user_message, response_message]:
        assert msg in test_agent.state


def test_agent_invoke_with_memory_no_feedback(
    test_model: OpenAIModel,
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    # Invoke the agent and check that appropriate functions were called.
    test_agent = BeehiveAgent(
        name="TestAgent",
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
    assert isinstance(output_message, BHMessage)
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


def test_agent_invoke_with_memory_feedback(
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
        test_agent = BeehiveAgent(
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


def test_agent_invoke_with_context(
    test_model: OpenAIModel,
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    test_agent1 = BeehiveAgent(
        name="TestAgent1",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )
    test_agent1.invoke(task="Tell me a joke!", stdout_printer=test_printer)

    # Invoke the agent and check that the context was added to agent2's state
    test_agent2 = BeehiveAgent(
        name="TestAgent",
        backstory="You are a helpful AI assistant.",
        model=test_model,
    )
    test_agent2.invoke(
        task="Tell me a second joke!",
        context=[test_agent1],
        stdout_printer=test_printer,
    )
    state = test_agent2.state
    assert len(state) == 4
    assert (
        isinstance(state[0], BHMessage)
        and state[0].role == MessageRole.SYSTEM
        and state[0].content == "You are a helpful AI assistant."
    )
    assert (
        isinstance(state[1], BHMessage)
        and state[1].role == MessageRole.CONTEXT
        and '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}'
        in state[1].content
    )
    assert (
        isinstance(state[2], BHMessage)
        and state[2].role == MessageRole.USER
        and state[2].content == "Tell me a second joke!"
    )
    assert (
        isinstance(state[3], BHMessage)
        and state[3].role == MessageRole.ASSISTANT
        and state[3].content == "Hello from our mocked class!"
    )


def test_agent_chat_loop(
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    # Test output
    class TestResponseModel(BaseModel):
        title: str
        action: Literal["thought", "observation", "action", "final_answer"]
        next_action: Literal["thought", "observation", "action", "final_answer"] | None
        content: str

    with mock.patch(
        "beehive.models.openai_model.OpenAIModel._client"
    ) as mocked_openai_client:
        chat_message_strings = [
            TestResponseModel(
                title="Title1",
                action="thought",
                next_action="observation",
                content="This is the first reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title2",
                action="observation",
                next_action="action",
                content="This is the second reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title3",
                action="action",
                next_action="final_answer",
                content="This is the third reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title4",
                action="final_answer",
                next_action=None,
                content="This is the final answer.",
            ).model_dump_json(),
        ]
        chat_messages = [MockChatCompletion([x]) for x in chat_message_strings]
        mocked_openai_client.chat.completions.create = mock.Mock(
            side_effect=chat_messages
        )

        # Create reasoning agent
        chat_loop = 4
        reasoning_agent = BeehiveAgent(
            name="ReasoningAgent",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            response_model=TestResponseModel,
            termination_condition=lambda x: x.action == "final_answer",
            chat_loop=chat_loop,
        )
        output = reasoning_agent.invoke(
            "This is a test task.", stdout_printer=test_printer
        )

        # There should be four messages in the output
        assert len(output["messages"]) == chat_loop
        for i in range(4):
            assert output["messages"][i] == BHMessage(
                role=MessageRole.ASSISTANT, content=chat_message_strings[i]
            )


def test_agent_chat_loop_early_termination(
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    # Test output
    class TestResponseModel(BaseModel):
        title: str
        action: Literal["thought", "observation", "action", "final_answer"]
        next_action: Literal["thought", "observation", "action", "final_answer"] | None
        content: str

    with mock.patch(
        "beehive.models.openai_model.OpenAIModel._client"
    ) as mocked_openai_client:
        chat_message_strings = [
            TestResponseModel(
                title="Title1",
                action="thought",
                next_action="observation",
                content="This is the first reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title2",
                action="observation",
                next_action="action",
                content="This is the second reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title3",
                action="final_answer",
                next_action=None,
                content="This is the final answer.",
            ).model_dump_json(),
        ]
        chat_messages = [MockChatCompletion([x]) for x in chat_message_strings]
        mocked_openai_client.chat.completions.create = mock.Mock(
            side_effect=chat_messages
        )

        # Create reasoning agent
        chat_loop = 4
        reasoning_agent = BeehiveAgent(
            name="ReasoningAgent",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            response_model=TestResponseModel,
            termination_condition=lambda x: x.action == "final_answer",
            chat_loop=chat_loop,
        )
        output = reasoning_agent.invoke(
            "This is a test task.", stdout_printer=test_printer
        )

        # There should be three messages in the output — we do not invoke the agent a
        # fourth time because of the termination condition.
        assert len(output["messages"]) == 3
        for i in range(3):
            assert output["messages"][i] == BHMessage(
                role=MessageRole.ASSISTANT, content=chat_message_strings[i]
            )


def test_agent_response_model_bad_output_no_pass_back_model_errors(
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    # Test output
    class TestResponseModel(BaseModel):
        title: str
        action: Literal["thought", "observation", "action", "final_answer"]
        next_action: Literal["thought", "observation", "action", "final_answer"] | None
        content: str

    with mock.patch(
        "beehive.models.openai_model.OpenAIModel._client"
    ) as mocked_openai_client:
        chat_message_strings = [
            TestResponseModel(
                title="Title1",
                action="thought",
                next_action="observation",
                content="This is the first reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title2",
                action="observation",
                next_action="action",
                content="This is the second reasoning message.",
            ).model_dump_json(),
            "This is a poorly formatted output.",
        ]
        chat_messages = [MockChatCompletion([x]) for x in chat_message_strings]
        mocked_openai_client.chat.completions.create = mock.Mock(
            side_effect=chat_messages
        )

        # Create reasoning agent
        chat_loop = 4
        reasoning_agent = BeehiveAgent(
            name="ReasoningAgent",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            response_model=TestResponseModel,
            termination_condition=lambda x: x.action == "final_answer",
            chat_loop=chat_loop,
        )

        # Invoke without passing back the model errors. This should raise a JSON error.
        with pytest.raises(json.decoder.JSONDecodeError) as _:
            reasoning_agent.invoke(
                "This is a test task.",
                pass_back_model_errors=False,
                stdout_printer=test_printer,
            )


def test_agent_response_model_bad_output_yes_pass_back_model_errors(
    test_storage: mock.MagicMock,
    test_feedback_storage: mock.MagicMock,
    test_printer: mock.MagicMock,
):
    # Test output
    class TestResponseModel(BaseModel):
        title: str
        action: Literal["thought", "observation", "action", "final_answer"]
        next_action: Literal["thought", "observation", "action", "final_answer"] | None
        content: str

    with mock.patch(
        "beehive.models.openai_model.OpenAIModel._client"
    ) as mocked_openai_client:
        chat_message_strings = [
            TestResponseModel(
                title="Title1",
                action="thought",
                next_action="observation",
                content="This is the first reasoning message.",
            ).model_dump_json(),
            TestResponseModel(
                title="Title2",
                action="observation",
                next_action="action",
                content="This is the second reasoning message.",
            ).model_dump_json(),
            "This is a poorly formatted output.",
            TestResponseModel(
                title="Title3",
                action="final_answer",
                next_action=None,
                content="This is the final answer.",
            ).model_dump_json(),
        ]
        chat_messages = [MockChatCompletion([x]) for x in chat_message_strings]
        mocked_openai_client.chat.completions.create = mock.Mock(
            side_effect=chat_messages
        )

        # Create reasoning agent
        chat_loop = 4
        reasoning_agent = BeehiveAgent(
            name="ReasoningAgent",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            response_model=TestResponseModel,
            termination_condition=lambda x: x.action == "final_answer",
            chat_loop=chat_loop,
        )

        # Invoke WITH passing back the model errors. This should add an extra
        # system message to our agent's state.
        output = reasoning_agent.invoke(
            "This is a test task.",
            pass_back_model_errors=True,
            stdout_printer=test_printer,
        )

        # There should be three messages in the output — we do not invoke the agent a
        # fourth time because of the termination condition.
        assert len(output["messages"]) == 3
        assert output["messages"][0] == BHMessage(
            role=MessageRole.ASSISTANT, content=chat_message_strings[0]
        )
        assert output["messages"][1] == BHMessage(
            role=MessageRole.ASSISTANT, content=chat_message_strings[1]
        )
        assert output["messages"][2] == BHMessage(
            role=MessageRole.ASSISTANT, content=chat_message_strings[3]
        )

        # In the state, we should see the following:
        #   System message
        #   User task
        #   First reasoning message
        #   Second reasoning message
        #   System message
        #   Third reasoning message
        assert len(reasoning_agent.state) == 6
        assert isinstance(reasoning_agent.state[4], BHMessage)
        assert reasoning_agent.state[4].role == MessageRole.SYSTEM
        expected_msg = "Encountered a `JSONDecodeError` / Pydantic `ValidationError` with the following content: <content>This is a poorly formatted output.</content>. **All output must be formatted according to the JSON schema described in the instructions**. Do not make this same mistake again."
        assert reasoning_agent.state[4].content == expected_msg
