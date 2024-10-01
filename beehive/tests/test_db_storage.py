from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from beehive.constants import INTERNAL_FOLDER_PATH
from beehive.invokable.agent import BeehiveAgent
from beehive.memory.db_storage import (
    BeehiveModel,
    DbStorage,
    InvokableModel,
    MessageModel,
    TaskModel,
    ToolCallModel,
)
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_model import OpenAIModel
from beehive.tools.base import BHTool, BHToolCall


@pytest.fixture(scope="module")
def test_db_storage() -> DbStorage:
    # Create a SQLLite database on the local machine.
    db_storage = DbStorage(
        db_uri=f"sqlite:///{Path(INTERNAL_FOLDER_PATH).resolve()}/test_beehive.db"
    )
    return db_storage


@pytest.fixture(scope="module")
def test_agent(test_db_storage: DbStorage):
    test_invokable = BeehiveAgent(
        name="TestAgent",
        backstory="You are a helpful AI assistant.",
        model=OpenAIModel(model="gpt-3.5-turbo"),
    )
    test_invokable._db_storage = test_db_storage
    test_db_storage.add_invokable(test_invokable)

    # History should be empty
    history = test_invokable.grab_history_for_invokable_execution()
    assert history == []
    return test_invokable


def test_get_model_objects(test_agent, test_db_storage):
    res = test_db_storage.get_model_objects(
        select(InvokableModel).where(InvokableModel.name == "TestAgent")
    )
    assert len(res) == 1
    invokable_obj = res[0]
    assert invokable_obj.type == "BeehiveAgent"
    assert invokable_obj.name == "TestAgent"
    assert invokable_obj.backstory == "You are a helpful AI assistant."

    # Adding the same invokable again will not duplicate results in our database.
    test_db_storage.add_invokable(test_agent)
    res = test_db_storage.get_model_objects(
        select(InvokableModel).where(InvokableModel.name == "TestAgent")
    )
    assert len(res) == 1

    # Try getting an object that doesn't exist
    res = test_db_storage.get_model_objects(
        select(InvokableModel).where(InvokableModel.name == "ThisDoesNotExist")
    )
    assert not res


def test_add_beehive(test_agent: BeehiveAgent, test_db_storage: DbStorage):
    bh_id = test_db_storage.add_beehive(
        invokable=test_agent, task="This is an example Beehive task."
    )
    res = test_db_storage.get_model_objects(
        select(BeehiveModel).where(BeehiveModel.id == bh_id)
    )
    assert len(res) == 1
    bh_obj = res[0]
    assert bh_obj.id == bh_id
    assert bh_obj.name == test_agent.name
    assert bh_obj.task == "This is an example Beehive task."


def test_add_task(test_agent: BeehiveAgent, test_db_storage: DbStorage):
    task_id = test_db_storage.add_task(
        task="This is an example task.", invokable=test_agent
    )

    # Task should now exist in the database
    res = test_db_storage.get_model_objects(
        select(TaskModel).where(TaskModel.id == task_id)
    )
    assert len(res) == 1
    task_obj = res[0]
    assert task_obj.content == "This is an example task."
    assert task_obj.invokable == "TestAgent"


def test_add_message(test_agent: BeehiveAgent, test_db_storage: DbStorage):
    task_id = test_db_storage.add_task(
        task="This is an example task.", invokable=test_agent
    )
    test_message = BHMessage(role=MessageRole.USER, content="This is a test message.")
    message_id = test_db_storage.add_message(task_id, test_message)

    # Message should now exist in the database
    res = test_db_storage.get_model_objects(
        select(MessageModel).where(MessageModel.id == message_id)
    )
    assert len(res) == 1
    message_obj = res[0]
    assert message_obj.role == "user"
    assert message_obj.content == "This is a test message."


def test_add_calls(test_agent: BeehiveAgent, test_db_storage: DbStorage):
    task_id = test_db_storage.add_task(
        task="This is an example task.", invokable=test_agent
    )

    def example_tool():
        """This is an example tool."""
        return "test tool"

    tool_call_id = str(uuid4())
    test_message = BHMessage(
        role=MessageRole.USER,
        content="This is a test message.",
        tool_calls=[
            BHToolCall(
                tool=BHTool(func=example_tool),
                tool_name="test_tool",
                tool_arguments={},
                tool_call_id=tool_call_id,
            )
        ],
    )
    message_id = test_db_storage.add_message(task_id, test_message)
    test_db_storage.add_tool_calls(test_message, message_id)

    # Tool call should now exist in the database
    res = test_db_storage.get_model_objects(
        select(ToolCallModel).where(ToolCallModel.tool_call_id == tool_call_id)
    )
    assert len(res) == 1
    tool_call_obj = res[0]
    assert tool_call_obj.message == message_id
    assert tool_call_obj.args == {}


def test_history(test_agent: BeehiveAgent, test_db_storage):
    # At this point, we've added a few different messages from a few different tasks.
    test_agent._db_storage = test_db_storage
    history = test_agent.grab_history_for_invokable_execution()
    assert len(history) == 2
    first_history_message = history[0]
    assert (
        isinstance(first_history_message, BHMessage)
        and first_history_message.role == MessageRole.USER
        and first_history_message.content == "This is a test message."
    )
    second_history_message = history[1]
    assert (
        isinstance(second_history_message, BHMessage)
        and second_history_message.role == MessageRole.USER
        and second_history_message.content == "This is a test message."
        and len(second_history_message.tool_calls) == 1
        and second_history_message.tool_calls[0].tool_name == "test_tool"
        and second_history_message.tool_calls[0].tool_arguments == {}
    )
