from typing import Literal
from unittest import mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai.chat_models import ChatOpenAI
from pydantic_core import ValidationError

from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Route
from beehive.invokable.beehive import Beehive, DynamicExecution, FixedExecution
from beehive.invokable.langchain_agent import BeehiveLangchainAgent
from beehive.invokable.types import BHStateElt
from beehive.message import BHMessage, MessageRole
from beehive.models.openai_model import OpenAIModel
from beehive.prompts import ConciseContextPrompt, FullContextPrompt
from beehive.tests.mocks import MockChatCompletion, MockOpenAIClient, MockPrinter


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


def test_dynamic_fixed_execution():
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )

        # Fixed execution
        fixed_execution = FixedExecution(route=(agent1 >> agent2))
        assert fixed_execution.route.this == agent1
        assert fixed_execution.route.other == agent2

        # Dynamic execution
        DynamicExecution(
            entrypoint=agent1,
            edges=[agent1 >> agent2],
            llm_router_additional_instructions={agent1: ["This is a test"]},
        )


def test_bh_setup():
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked:
        mocked.return_value = MockOpenAIClient()
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        bh_with_edges = Beehive(
            name="TestBeehive",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            execution_process=DynamicExecution(
                entrypoint=agent1,
                edges=[agent1 >> agent2],
                llm_router_additional_instructions={agent1: ["This is a test"]},
            ),
        )
        expected_invokable_map = {
            "TestAgent1": agent1,
            "TestAgent2": agent2,
        }

        # Invokables
        assert bh_with_edges._invokable_map == expected_invokable_map
        assert agent1 in bh_with_edges._invokables
        assert agent2 in bh_with_edges._invokables
        assert len(bh_with_edges._nodes) == len(bh_with_edges._invokables)
        assert (
            bh_with_edges._entrypoint_node
            and bh_with_edges._entrypoint_node.invokable == agent1
        )

        # Edges
        assert isinstance(bh_with_edges.execution_process, DynamicExecution)
        assert len(bh_with_edges.execution_process.edges) == 1
        bh_edge = bh_with_edges.execution_process.edges[0]
        assert bh_edge.this == agent1
        assert bh_edge.other == agent2

        # Router —— used to intelligently route conversations
        assert isinstance(bh_with_edges._router, BeehiveAgent)

        # Beehive with route
        bh_with_route = Beehive(
            name="TestBeehive",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            execution_process=FixedExecution(route=(agent1 >> agent2)),
        )

        # Invokables
        assert isinstance(bh_with_route.execution_process, FixedExecution)
        assert bh_with_route._invokable_map == expected_invokable_map
        assert agent1 in bh_with_route._invokables
        assert agent2 in bh_with_route._invokables
        assert not bh_with_route._entrypoint_node
        assert not bh_with_route._nodes

        # Router —— used to create prompts
        assert isinstance(bh_with_route._router, BeehiveAgent)

        # self-loops in routes are not permitted
        with pytest.raises(ValueError):
            Beehive(
                name="TestBeehive",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=FixedExecution(route=(agent1 >> agent1)),
            )
        # self-loops in edges are not permitted
        with pytest.raises(ValueError):
            Beehive(
                name="TestBeehive",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=DynamicExecution(
                    entrypoint=agent1,
                    edges=[agent1 >> agent1],
                ),
            )


def test_beehive_function_calls():
    with mock.patch(
        "beehive.invokable.beehive.Beehive._invoke_without_route"
    ) as mocked:
        mocked.return_value = ["Invoked Beehive without a route!"]
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        bh_with_edges = Beehive(
            name="TestBeehive",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            execution_process=DynamicExecution(
                entrypoint=agent1,
                edges=[agent1 >> agent2],
            ),
        )
        noroute_output = bh_with_edges._invoke(task="This is an example task")
        assert noroute_output == ["Invoked Beehive without a route!"]  # type: ignore

    # Invoke with a route
    with mock.patch("beehive.invokable.beehive.Beehive._invoke_with_route") as mocked:
        mocked.return_value = ["Invoked Beehive with a route!"]
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        bh_with_route = Beehive(
            name="TestBeehive",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
            execution_process=FixedExecution(route=(agent1 >> agent2)),
        )
        route_output = bh_with_route._invoke(task="This is an example task")
        assert route_output == ["Invoked Beehive with a route!"]  # type: ignore


def test_beehive_no_route(test_storage, test_feedback_storage, test_printer):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked_model:
        mocked_model.return_value = MockOpenAIClient()
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )

        with mock.patch("beehive.invokable.beehive.Beehive._router") as mocked_router:
            mocked_router.name = "Router"

            # Router return messages. This is for all of our tests
            mocked_router_return_messages = [
                # Testing `invoke_router_without_route` with a bad output
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task."}',
                    )
                ],
                # Testing `invoke_router_without_route` with a good output
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent2", "reason": "Agent 2 specializes in this next task.", "task": "This is agent 2\'s next task."}',
                    )
                ],
                # Testing full invokation
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent2", "reason": "Agent 2 specializes in this next task.", "task": "This is agent 2\'s next task."}',
                    )
                ],
            ]
            mocked_router._invoke = mock.Mock(side_effect=mocked_router_return_messages)

            # The router should start at Agent1, since that is the entrypoint. From
            # there, it should only be allowed to send the conversation to Agent 2.
            mocked_router._invoke.return_value = [
                BHMessage(
                    role=MessageRole.ASSISTANT,
                    content='{"task": "This is agent 1\'s first task."}',
                )
            ]
            bh = Beehive(
                name="TestBeehive",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=DynamicExecution(
                    entrypoint=agent1,
                    edges=[agent1 >> agent2],
                ),
            )
            bh._db_storage = test_storage

            # The starting node shoul be agent 1
            assert bh._entrypoint_node
            assert bh._entrypoint_node.invokable == agent1

            # There should only be one possible next agent
            _, next_agents = bh.prompt_router_for_next_agent(
                original_task="", node=bh._entrypoint_node, prev_invokables=[]
            )
            assert len(next_agents) == 1
            assert next_agents[0] == agent2

            # Pydantic class for next agent
            next_agent_actor = bh.create_next_agent_actor_pydantic_class(
                [x.name for x in next_agents]
            )
            assert (
                next_agent_actor.model_fields["agent"].annotation
                == Literal["TestAgent2", "FINISH"]
            )

            # Router invokation -- this should fail because of a bad output
            with pytest.raises(ValidationError):
                bh.invoke_router_without_route(
                    original_task="",
                    current_node=bh._entrypoint_node,
                    counter=0,
                    retry_limit=100,
                    pass_back_model_errors=False,
                    printer=test_printer,
                )

            # Now, set a valid output
            next_agent_name, task, counter = bh.invoke_router_without_route(
                original_task="",
                current_node=bh._entrypoint_node,
                counter=0,
                retry_limit=100,
                pass_back_model_errors=True,
                printer=test_printer,
            )
            assert next_agent_name == "TestAgent2"
            assert task == "This is agent 2's next task."
            assert counter == 1

            # Invoke — this should trigger agent 1, then agent 2. Then, since agent 2
            # cannot talk to anyone else, the Beehive should finish.
            output = bh.invoke("dumb task", stdout_printer=test_printer)
            messages = output["messages"]
            assert len(messages) == 2

            # First message should be from agent 1
            message1 = messages[0]
            assert isinstance(message1, BHStateElt)
            assert message1.task_id == "some_unique_task_id"
            assert message1.invokable == agent1
            assert len(message1.completion_messages) == 1
            assert message1.completion_messages[0] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Second message should be from agent 2
            message2 = messages[1]
            assert isinstance(message2, BHStateElt)
            assert message2.task_id == "some_unique_task_id"
            assert message2.invokable == agent2
            assert len(message2.completion_messages) == 1
            assert message2.completion_messages[0] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Beehive tasks are not directly stored in the database
            assert not output["task_id"]

            # Now, check the states of the invokables. The state of agent 1 should only
            # have the system message, the user task, and the response.
            assert len(agent1.state) == 3
            assert isinstance(agent1.state[0], BHMessage)
            assert agent1.state[0].role == MessageRole.SYSTEM
            assert agent1.state[0].content == "You are a helpful AI assistant."

            assert isinstance(agent1.state[1], BHMessage)
            assert agent1.state[1].role == MessageRole.USER
            assert agent1.state[1].content == "This is agent 1's first task."

            assert isinstance(agent1.state[2], BHMessage)
            assert agent1.state[2].role == MessageRole.ASSISTANT
            assert agent1.state[2].content == "Hello from our mocked class!"

            # Agent 2's state should have one more message than agent 1, since it
            # includes the context from agent 1.
            assert len(agent2.state) == 4

            assert isinstance(agent2.state[0], BHMessage)
            assert agent2.state[0].role == MessageRole.SYSTEM
            assert agent2.state[0].content == "You are a helpful AI assistant."

            assert isinstance(agent2.state[1], BHMessage)
            assert agent2.state[1].role == MessageRole.CONTEXT
            assert (
                agent2.state[1].content
                == FullContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}',
                ).render()
            )

            assert isinstance(agent2.state[2], BHMessage)
            assert agent2.state[2].role == MessageRole.USER
            assert agent2.state[2].content == "This is agent 2's next task."

            assert isinstance(agent2.state[3], BHMessage)
            assert agent2.state[3].role == MessageRole.ASSISTANT
            assert agent2.state[3].content == "Hello from our mocked class!"


def test_beehive_no_route_with_cycle(test_storage, test_feedback_storage, test_printer):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked_model:
        mocked_model.return_value = MockOpenAIClient()
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        with mock.patch("beehive.invokable.beehive.Beehive._router") as mocked_router:
            next_agents = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent2", "reason": "Agent 2 specializes in this next task.", "task": "This is agent 2\'s next task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent1", "reason": "Agent 1 specializes in this next task.", "task": "This is agent 1\'s final task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent2", "reason": "Agent 2 specializes in this next task.", "task": "This is agent 2\'s final task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "FINISH", "reason": "We are done with this test.", "task": "We are done with this task."}',
                    )
                ],
            ]
            mocked_router.name = "Router"
            mocked_router._invoke = mock.Mock(side_effect=next_agents)

            bh = Beehive(
                name="TestBeehive",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=DynamicExecution(
                    entrypoint=agent1,
                    edges=[
                        agent1 >> agent2,
                        agent2 >> agent1,
                    ],
                ),
            )
            bh._db_storage = test_storage
            output = bh.invoke("test", stdout_printer=test_printer)
            assert len(output["messages"]) == 4

            # Check outputs
            first_output = output["messages"][0]
            assert isinstance(first_output, BHStateElt)
            assert first_output.index == 1
            assert first_output.task == "This is agent 1's first task."
            assert first_output.invokable == agent1
            assert len(first_output.completion_messages) == 1
            assert (
                first_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            second_output = output["messages"][1]
            assert isinstance(second_output, BHStateElt)
            assert second_output.index == 2
            assert second_output.task == "This is agent 2's next task."
            assert second_output.invokable == agent2
            assert len(second_output.completion_messages) == 1
            assert (
                second_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            third_output = output["messages"][2]
            assert isinstance(third_output, BHStateElt)
            assert third_output.index == 3
            assert third_output.task == "This is agent 1's final task."
            assert third_output.invokable == agent1
            assert len(third_output.completion_messages) == 1
            assert (
                third_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            fourth_output = output["messages"][3]
            assert isinstance(fourth_output, BHStateElt)
            assert fourth_output.index == 4
            assert fourth_output.task == "This is agent 2's final task."
            assert fourth_output.invokable == agent2
            assert len(fourth_output.completion_messages) == 1
            assert (
                fourth_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            # Check agent states.
            # For agent 1, uur expectation is:
            #   First message is the system message
            #   Second message is the first user task
            #   Third message is the agent's response
            #   Fourth message is the context from the second task (since the conversation moves from second agent back to the first agent)
            #   Fifth message is the agent's second task
            #   Sixth message is the agent's second response
            assert len(agent1.state) == 6
            first_message = agent1.state[0]
            assert isinstance(first_message, BHMessage)
            assert first_message.role == MessageRole.SYSTEM
            assert first_message.content == "You are a helpful AI assistant."

            second_message = agent1.state[1]
            assert isinstance(second_message, BHMessage)
            assert second_message.role == MessageRole.USER
            assert second_message.content == "This is agent 1's first task."

            third_message = agent1.state[2]
            assert isinstance(third_message, BHMessage)
            assert third_message.role == MessageRole.ASSISTANT
            assert third_message.content == "Hello from our mocked class!"

            fourth_message = agent1.state[3]
            assert isinstance(fourth_message, BHMessage)
            assert fourth_message.role == MessageRole.CONTEXT
            assert (
                fourth_message.content
                == FullContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent2"}',
                ).render()
            )

            fifth_message = agent1.state[4]
            assert isinstance(fifth_message, BHMessage)
            assert fifth_message.role == MessageRole.USER
            assert fifth_message.content == "This is agent 1's final task."

            sixth_message = agent1.state[5]
            assert isinstance(sixth_message, BHMessage)
            assert sixth_message.role == MessageRole.ASSISTANT
            assert sixth_message.content == "Hello from our mocked class!"

            # For agent 2, uur expectation is:
            #   First message is the system message
            #   Second message is the first context from task 1
            #   Third message is the first user task
            #   Fourth message is the agent's response
            #   Fifth message is the agent's second task. Note that there should not
            #     be any context, because we only create a context if prevous invokables
            #     create messages that are not already in the agent's context. Since our
            #     mocked class always returns the same message, this condition is not
            #     satisfied.
            #   Sixth message is the agent's second response
            assert len(agent2.state) == 6
            first_message2 = agent2.state[0]
            assert isinstance(first_message2, BHMessage)
            assert first_message2.role == MessageRole.SYSTEM
            assert first_message2.content == "You are a helpful AI assistant."

            second_message2 = agent2.state[1]
            assert isinstance(second_message2, BHMessage)
            assert second_message2.role == MessageRole.CONTEXT
            assert (
                second_message2.content
                == FullContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}',
                ).render()
            )

            third_message2 = agent2.state[2]
            assert isinstance(third_message2, BHMessage)
            assert third_message2.role == MessageRole.USER
            assert third_message2.content == "This is agent 2's next task."

            fourth_message2 = agent2.state[3]
            assert isinstance(fourth_message2, BHMessage)
            assert fourth_message2.role == MessageRole.ASSISTANT
            assert fourth_message2.content == "Hello from our mocked class!"

            fifth_message2 = agent2.state[4]
            assert isinstance(fifth_message2, BHMessage)
            assert fifth_message2.role == MessageRole.USER
            assert fifth_message2.content == "This is agent 2's final task."

            sixth_message2 = agent2.state[5]
            assert isinstance(sixth_message2, BHMessage)
            assert sixth_message2.role == MessageRole.ASSISTANT
            assert sixth_message2.content == "Hello from our mocked class!"


def test_beehive_route(test_storage, test_feedback_storage, test_printer):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked_model:
        mocked_model.return_value = MockOpenAIClient()
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )

        with mock.patch("beehive.invokable.beehive.Beehive._router") as mocked_router:
            next_tasks = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 2\'s next task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s final task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 2\'s final task."}',
                    )
                ],
            ]
            mocked_router.name = "Router"
            mocked_router._invoke = mock.Mock(side_effect=next_tasks)

            bh = Beehive(
                name="TestBeehive",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=FixedExecution(
                    route=(agent1 >> agent2 >> agent1 >> agent2)
                ),
            )
            assert isinstance(bh.execution_process, FixedExecution)
            assert isinstance(bh.execution_process.route, Route)
            assert bh.execution_process.route._invokable_order == [
                agent1,
                agent2,
                agent1,
                agent2,
            ]
            bh._db_storage = test_storage
            output = bh.invoke("test", stdout_printer=test_printer)

            # Check messages
            output_elts = output["messages"]
            assert len(output_elts) == 4

            # Check outputs
            first_output = output["messages"][0]
            assert isinstance(first_output, BHStateElt)
            assert first_output.index == 1
            assert first_output.task == "This is agent 1's first task."
            assert first_output.invokable == agent1
            assert len(first_output.completion_messages) == 1
            assert (
                first_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            second_output = output["messages"][1]
            assert isinstance(second_output, BHStateElt)
            assert second_output.index == 2
            assert second_output.task == "This is agent 2's next task."
            assert second_output.invokable == agent2
            assert len(second_output.completion_messages) == 1
            assert (
                second_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            third_output = output["messages"][2]
            assert isinstance(third_output, BHStateElt)
            assert third_output.index == 3
            assert third_output.task == "This is agent 1's final task."
            assert third_output.invokable == agent1
            assert len(third_output.completion_messages) == 1
            assert (
                third_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            fourth_output = output["messages"][3]
            assert isinstance(fourth_output, BHStateElt)
            assert fourth_output.index == 4
            assert fourth_output.task == "This is agent 2's final task."
            assert fourth_output.invokable == agent2
            assert len(fourth_output.completion_messages) == 1
            assert (
                fourth_output.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            # Check agent states.
            # For agent 1, uur expectation is:
            #   First message is the system message
            #   Second message is the first user task
            #   Third message is the agent's response
            #   Fourth message is the context from the second task (since the conversation moves from second agent back to the first agent)
            #   Fifth message is the agent's second task
            #   Sixth message is the agent's second response
            assert len(agent1.state) == 6
            first_message = agent1.state[0]
            assert isinstance(first_message, BHMessage)
            assert first_message.role == MessageRole.SYSTEM
            assert first_message.content == "You are a helpful AI assistant."

            second_message = agent1.state[1]
            assert isinstance(second_message, BHMessage)
            assert second_message.role == MessageRole.USER
            assert second_message.content == "This is agent 1's first task."

            third_message = agent1.state[2]
            assert isinstance(third_message, BHMessage)
            assert third_message.role == MessageRole.ASSISTANT
            assert third_message.content == "Hello from our mocked class!"

            fourth_message = agent1.state[3]
            assert isinstance(fourth_message, BHMessage)
            assert fourth_message.role == MessageRole.CONTEXT
            assert (
                fourth_message.content
                == FullContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent2"}',
                ).render()
            )

            fifth_message = agent1.state[4]
            assert isinstance(fifth_message, BHMessage)
            assert fifth_message.role == MessageRole.USER
            assert fifth_message.content == "This is agent 1's final task."

            sixth_message = agent1.state[5]
            assert isinstance(sixth_message, BHMessage)
            assert sixth_message.role == MessageRole.ASSISTANT
            assert sixth_message.content == "Hello from our mocked class!"

            # For agent 2, uur expectation is:
            #   First message is the system message
            #   Second message is the first context from task 1
            #   Third message is the first user task
            #   Fourth message is the agent's response
            #   Fifth message is the agent's second task. Note that there should not
            #     be any context, because we only create a context if prevous invokables
            #     create messages that are not already in the agent's context. Since our
            #     mocked class always returns the same message, this condition is not
            #     satisfied.
            #   Sixth message is the agent's second response
            assert len(agent2.state) == 6
            first_message2 = agent2.state[0]
            assert isinstance(first_message2, BHMessage)
            assert first_message2.role == MessageRole.SYSTEM
            assert first_message2.content == "You are a helpful AI assistant."

            second_message2 = agent2.state[1]
            assert isinstance(second_message2, BHMessage)
            assert second_message2.role == MessageRole.CONTEXT
            assert (
                second_message2.content
                == FullContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}',
                ).render()
            )

            third_message2 = agent2.state[2]
            assert isinstance(third_message2, BHMessage)
            assert third_message2.role == MessageRole.USER
            assert third_message2.content == "This is agent 2's next task."

            fourth_message2 = agent2.state[3]
            assert isinstance(fourth_message2, BHMessage)
            assert fourth_message2.role == MessageRole.ASSISTANT
            assert fourth_message2.content == "Hello from our mocked class!"

            fifth_message2 = agent2.state[4]
            assert isinstance(fifth_message2, BHMessage)
            assert fifth_message2.role == MessageRole.USER
            assert fifth_message2.content == "This is agent 2's final task."

            sixth_message2 = agent2.state[5]
            assert isinstance(sixth_message2, BHMessage)
            assert sixth_message2.role == MessageRole.ASSISTANT
            assert sixth_message2.content == "Hello from our mocked class!"


def test_beehive_no_route_with_nested_beehive(
    test_storage, test_feedback_storage, test_printer
):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked_model:
        mocked_model.return_value = MockOpenAIClient()
        agent0 = BeehiveAgent(
            name="TestAgent0",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        with mock.patch(
            "beehive.invokable.beehive.Beehive._create_router"
        ) as mocked_router:
            mocked_router_instances = [
                mock.Mock(name="RouterInner"),
                mock.Mock(name="RouterOuter"),
            ]

            # Next tasks for inner router (the nested Beehive will use a pre-defined
            # route).
            inner_router_next_tasks = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first test task in the inner beehive."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 2\'s next task."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s final task."}',
                    )
                ],
            ]
            mocked_router_instances[0]._invoke.side_effect = inner_router_next_tasks

            # Next agents for outer router
            outer_router_next_agents = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is the TestAgent0\'s task."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "InnerBeehive", "reason": "The inner Beehive can accomplish this task.", "task": "This is the inner Beehive\'s task."}',
                    )
                ],
            ]
            mocked_router_instances[1]._invoke.side_effect = outer_router_next_agents

            mocked_router.side_effect = mocked_router_instances

            # Inner Beehive
            inner_bh = Beehive(
                name="InnerBeehive",
                backstory="You are the inner Beehive.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=FixedExecution(route=(agent1 >> agent2 >> agent1)),
            )
            inner_bh._db_storage = test_storage
            outer_bh = Beehive(
                name="InnerBeehive",
                backstory="You are the inner Beehive.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=DynamicExecution(
                    entrypoint=agent0,
                    edges=[agent0 >> inner_bh],
                ),
            )
            outer_bh._db_storage = test_storage
            output = outer_bh.invoke("test task", stdout_printer=test_printer)

            # Output messages
            output_elts = output["messages"]
            assert len(output_elts) == 4

            # Check element contents
            first_elt = output_elts[0]
            assert isinstance(first_elt, BHStateElt)
            assert first_elt.index == 1
            assert first_elt.task == "This is the TestAgent0's task."
            assert first_elt.invokable == agent0
            assert len(first_elt.completion_messages) == 1
            assert (
                first_elt.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            second_elt = output_elts[1]
            assert isinstance(second_elt, BHStateElt)
            assert second_elt.index == 1
            assert (
                second_elt.task
                == "This is agent 1's first test task in the inner beehive."
            )
            assert second_elt.invokable == agent1
            assert len(second_elt.completion_messages) == 1
            assert (
                second_elt.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            third_elt = output_elts[2]
            assert isinstance(third_elt, BHStateElt)
            assert third_elt.index == 2
            assert third_elt.task == "This is agent 2's next task."
            assert third_elt.invokable == agent2
            assert len(third_elt.completion_messages) == 1
            assert (
                third_elt.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            fourth_elt = output_elts[3]
            assert isinstance(fourth_elt, BHStateElt)
            assert fourth_elt.index == 3
            assert fourth_elt.task == "This is agent 1's final task."
            assert fourth_elt.invokable == agent1
            assert len(fourth_elt.completion_messages) == 1
            assert (
                fourth_elt.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            # Check invokable states. For Agent 0, we expect:
            #  The system message
            #  The user task
            #  The agent's response
            assert len(agent0.state) == 3
            first_message0 = agent0.state[0]
            assert isinstance(first_message0, BHMessage)
            assert first_message0.role == MessageRole.SYSTEM
            assert first_message0.content == "You are a helpful AI assistant."

            second_message0 = agent0.state[1]
            assert isinstance(second_message0, BHMessage)
            assert second_message0.role == MessageRole.USER
            assert second_message0.content == "This is the TestAgent0's task."

            third_message0 = agent0.state[2]
            assert isinstance(third_message0, BHMessage)
            assert third_message0.role == MessageRole.ASSISTANT
            assert third_message0.content == "Hello from our mocked class!"

            # In the inner beehive, there are two invokables: agent 1 and agent 2.
            assert len(inner_bh.state) == 3

            # For agent 1, uur expectation is:
            #   First message is the system message
            #   Second message is context from agent 0
            #   Third message is the first user task
            #   Fourth message is the agent's response
            #   Fifth message is the context from the second task
            #     Context from agent 0 is already in the agent's state, so it doesn't
            #     get re-added here.
            #   Sixth message is the agent's second task
            #   Seventh message is the agent's second response
            assert len(agent1.state) == 7
            first_message1 = agent1.state[0]
            assert isinstance(first_message1, BHMessage)
            assert first_message1.role == MessageRole.SYSTEM
            assert first_message1.content == "You are a helpful AI assistant."

            second_message1 = agent1.state[1]
            assert isinstance(second_message1, BHMessage)
            assert second_message1.role == MessageRole.CONTEXT
            assert (
                second_message1.content
                == FullContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent0"}',
                ).render()
            )

            third_message1 = agent1.state[2]
            assert isinstance(third_message1, BHMessage)
            assert third_message1.role == MessageRole.USER
            assert (
                third_message1.content
                == "This is agent 1's first test task in the inner beehive."
            )

            fourth_message1 = agent1.state[3]
            assert isinstance(fourth_message1, BHMessage)
            assert (
                fourth_message1.role == MessageRole.ASSISTANT
                and fourth_message1.content == "Hello from our mocked class!"
            )

            fifth_message1 = agent1.state[4]
            assert isinstance(fifth_message1, BHMessage)
            assert fifth_message1.role == MessageRole.CONTEXT
            assert (
                fifth_message1.content
                == ConciseContextPrompt(
                    context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent2"}',
                ).render()
            )

            sixth_message1 = agent1.state[5]
            assert isinstance(sixth_message1, BHMessage)
            assert sixth_message1.role == MessageRole.USER
            assert sixth_message1.content == "This is agent 1's final task."

            seventh_message1 = agent1.state[6]
            assert isinstance(seventh_message1, BHMessage)
            assert seventh_message1.role == MessageRole.ASSISTANT
            assert seventh_message1.content == "Hello from our mocked class!"

            # For agent 2, uur expectation is:
            #   First message is the system message
            #   Second message is the first context from task 1
            #   Third message is the first user task
            #   Fourth message is the agent's response
            assert len(agent2.state) == 4
            first_message2 = agent2.state[0]
            assert isinstance(first_message2, BHMessage)
            assert first_message2.role == MessageRole.SYSTEM
            assert first_message2.content == "You are a helpful AI assistant."

            second_message2 = agent2.state[1]
            assert isinstance(second_message2, BHMessage)
            assert second_message2.role == MessageRole.CONTEXT
            assert (
                second_message2.content
                == FullContextPrompt(
                    context="\n".join(
                        [
                            '- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent0"}',
                            '- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}',
                        ]
                    ),
                ).render()
            )

            third_message2 = agent2.state[2]
            assert isinstance(third_message2, BHMessage)
            assert third_message2.role == MessageRole.USER
            assert third_message2.content == "This is agent 2's next task."

            fourth_message2 = agent2.state[3]
            assert isinstance(fourth_message2, BHMessage)
            assert fourth_message2.role == MessageRole.ASSISTANT
            assert fourth_message2.content == "Hello from our mocked class!"


def test_beehive_with_mix_of_agents(test_storage, test_feedback_storage, test_printer):
    with mock.patch("beehive.models.openai_model.OpenAI") as mocked_bh_model:
        mocked_bh_model.return_value = MockOpenAIClient()

        with mock.patch(
            "langchain_openai.chat_models.base.openai.OpenAI"
        ) as mocked_lc_model:
            mocked_lc_model.return_value = MockOpenAIClient()

            agent1 = BeehiveLangchainAgent(
                name="TestLangchainAgent1",
                backstory="You are a helpful AI assistant.",
                model=ChatOpenAI(model="gpt-3.5-turbo"),
            )
            agent2 = BeehiveAgent(
                name="TestAgent2",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
            )

            with mock.patch(
                "beehive.invokable.beehive.Beehive._router"
            ) as mocked_router:
                mocked_router.name = "Router"

                # The router should start at Agent1, since that is the entrypoint. From
                # there, it should only be allowed to send the conversation to Agent 2.
                next_agents = [
                    [
                        BHMessage(
                            role=MessageRole.ASSISTANT,
                            content='{"task": "This is the first task for agent 1."}',
                        )
                    ],
                    [
                        BHMessage(
                            role=MessageRole.ASSISTANT,
                            content='{"agent": "TestAgent2", "reason": "Agent 2 specializes in this next task.", "task": "This is agent 2\'s next task."}',
                        )
                    ],
                    [
                        BHMessage(
                            role=MessageRole.ASSISTANT,
                            content='{"agent": "TestLangchainAgent1", "reason": "Agent 1 specializes in this final task.", "task": "This is agent 1\'s final task."}',
                        )
                    ],
                    [
                        BHMessage(
                            role=MessageRole.ASSISTANT,
                            content='{"agent": "FINISH", "reason": "We are done with this test.", "task": "We are done with this task."}',
                        )
                    ],
                ]
                mocked_router._invoke = mock.Mock(side_effect=next_agents)
                bh = Beehive(
                    name="TestBeehive",
                    backstory="You are a helpful AI assistant.",
                    model=OpenAIModel(model="gpt-3.5-turbo"),
                    execution_process=DynamicExecution(
                        entrypoint=agent1,
                        edges=[
                            agent1 >> agent2,
                            agent2 >> agent1,
                        ],
                    ),
                )
                bh._db_storage = test_storage
                output = bh.invoke("Test task", stdout_printer=test_printer)
                output_messages = output["messages"]
                assert len(output_messages) == 3

                # Message contents
                first_elt = output_messages[0]
                assert isinstance(first_elt, BHStateElt)
                assert first_elt.task == "This is the first task for agent 1."
                assert len(first_elt.completion_messages) == 1
                assert isinstance(first_elt.completion_messages[0], BHMessage)
                assert (
                    first_elt.completion_messages[0].content
                    == "Hello from our mocked class!"
                )

                second_elt = output_messages[1]
                assert isinstance(second_elt, BHStateElt)
                assert second_elt.task == "This is agent 2's next task."
                assert len(second_elt.completion_messages) == 1
                assert isinstance(second_elt.completion_messages[0], BHMessage)
                assert (
                    second_elt.completion_messages[0].content
                    == "Hello from our mocked class!"
                )

                third_elt = output_messages[2]
                assert isinstance(third_elt, BHStateElt)
                assert third_elt.task == "This is agent 1's final task."
                assert len(third_elt.completion_messages) == 1
                assert isinstance(third_elt.completion_messages[0], BHMessage)
                assert (
                    third_elt.completion_messages[0].content
                    == "Hello from our mocked class!"
                )

                # For agent 1, uur expectation is:
                #   First message is the system message
                #   Second message is the first user task
                #   Third message is the agent's response
                #   Fourth message is the context from the second task
                #   Fifth message is the agent's second task
                #   Sixth message is the agent's second response
                assert len(agent1.state) == 6
                first_message1 = agent1.state[0]
                assert isinstance(first_message1, SystemMessage)
                assert first_message1.content == "You are a helpful AI assistant."

                second_message1 = agent1.state[1]
                assert isinstance(second_message1, HumanMessage)
                assert second_message1.content == "This is the first task for agent 1."

                third_message1 = agent1.state[2]
                assert isinstance(third_message1, AIMessage)
                assert third_message1.content == "Hello from our mocked class!"

                fourth_message1 = agent1.state[3]
                assert isinstance(fourth_message1, AIMessage)
                assert (
                    fourth_message1.content
                    == FullContextPrompt(
                        context='- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent2"}',
                    ).render()
                )

                fifth_message1 = agent1.state[4]
                assert isinstance(fifth_message1, HumanMessage)
                assert fifth_message1.content == "This is agent 1's final task."

                sixth_message1 = agent1.state[5]
                assert isinstance(sixth_message1, AIMessage)
                assert sixth_message1.content == "Hello from our mocked class!"

                # For agent 2. We've tested this extensively in our previous tests, so
                # skip.


def test_beehive_with_questions_between_agents(
    test_storage, test_feedback_storage, test_printer
):
    with mock.patch("beehive.models.openai_model.OpenAIModel._client") as mocked_model:
        chat_messages = [
            MockChatCompletion(["Hello from our mocked class!"]),
            MockChatCompletion(
                [
                    '{"question": "Can you clarify something for me?", "invokable": "TestAgent1", "reason": "I need this clarification please"}'
                ]
            ),
            MockChatCompletion(["Hello from our mocked class (clarified)!"]),
            MockChatCompletion(
                ["After receiving clarification, hello from our mocked class!"]
            ),
        ]
        mocked_model.chat.completions.create = mock.Mock(side_effect=chat_messages)

        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        with mock.patch("beehive.invokable.beehive.Beehive._router") as mocked_router:
            next_agents = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent2", "reason": "Agent 2 specializes in this next task.", "task": "This is agent 2\'s next task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "FINISH", "reason": "We are done with this test.", "task": "We are done with this task."}',
                    )
                ],
            ]
            mocked_router.name = "Router"
            mocked_router._invoke = mock.Mock(side_effect=next_agents)

            bh = Beehive(
                name="TestBeehive",
                backstory="You are a helpful AI assistant.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=DynamicExecution(
                    entrypoint=agent1,
                    edges=[
                        agent1 >> agent2,
                        agent2 >> agent1,
                    ],
                ),
                enable_questioning=True,
            )
            bh._db_storage = test_storage
            output = bh.invoke("test", stdout_printer=test_printer)

            # Check elements -- there should be four. One for the agent1's first task,
            # one for agent2's question, one for agent1's response, and one for agent2's
            # final message.
            output_elements = output["messages"]
            assert len(output_elements) == 4

            # Element-by-element checks
            first_element = output_elements[0]
            assert isinstance(first_element, BHStateElt)
            assert first_element.invokable == agent1
            assert len(first_element.completion_messages) == 1
            assert isinstance(first_element.completion_messages[0], BHMessage)
            assert (
                first_element.completion_messages[0].content
                == "Hello from our mocked class!"
            )

            second_element = output_elements[1]
            assert isinstance(second_element, BHStateElt)
            assert second_element.invokable == agent2
            assert len(second_element.completion_messages) == 1
            assert isinstance(second_element.completion_messages[0], BHMessage)
            assert (
                second_element.completion_messages[0].content
                == '{"question": "Can you clarify something for me?", "invokable": "TestAgent1", "reason": "I need this clarification please"}'
            )

            third_element = output_elements[2]
            assert isinstance(third_element, BHStateElt)
            assert third_element.invokable == agent1
            assert len(third_element.completion_messages) == 1
            assert isinstance(third_element.completion_messages[0], BHMessage)
            assert (
                third_element.completion_messages[0].content
                == "Hello from our mocked class (clarified)!"
            )

            fourth_element = output_elements[3]
            assert isinstance(fourth_element, BHStateElt)
            assert fourth_element.invokable == agent2
            assert len(fourth_element.completion_messages) == 1
            assert isinstance(fourth_element.completion_messages[0], BHMessage)
            assert (
                fourth_element.completion_messages[0].content
                == "After receiving clarification, hello from our mocked class!"
            )

            # Agent states. We expect agent 1's state to have 5 messages:
            #   System message
            #   First user task
            #   First response
            #   Clarification question from agent 2
            #   Response to clarification question from agent 2
            assert len(agent1.state) == 5
            assert agent1.state[0] == BHMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful AI assistant.",
                tool_calls=[],
            )
            assert agent1.state[1] == BHMessage(
                role=MessageRole.USER,
                content="This is agent 1's first task.",
                tool_calls=[],
            )
            assert agent1.state[2] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="Hello from our mocked class!",
                tool_calls=[],
            )
            assert agent1.state[3] == BHMessage(
                role=MessageRole.USER,
                content="Can you clarify something for me?",
                tool_calls=[],
            )
            assert agent1.state[4] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="Hello from our mocked class (clarified)!",
                tool_calls=[],
            )

            # We expect agent 2's state to have 7 messages:
            #   System message
            #   Context message
            #   First user task
            #   Clarification question to agent 1
            #   Context message containing agent 1's answer
            #   First user task (again)
            #   First response
            assert len(agent2.state) == 7
            assert agent2.state[0] == BHMessage(
                role=MessageRole.SYSTEM,
                content="You are a helpful AI assistant.",
                tool_calls=[],
            )
            assert isinstance(agent2.state[1], BHMessage)
            assert agent2.state[1].role == MessageRole.CONTEXT
            assert (
                '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}'
                in agent2.state[1].content
            )

            assert isinstance(agent2.state[2], BHMessage)
            assert agent2.state[2].role == MessageRole.USER
            assert (
                "This is agent 2's next task.\n\nAsk clarifying questions if you are unsure of how to complete the task or need any clarification."
                in agent2.state[2].content
            )

            assert agent2.state[3] == BHMessage(
                role=MessageRole.QUESTION,
                content='{"question": "Can you clarify something for me?", "invokable": "TestAgent1", "reason": "I need this clarification please"}',
                tool_calls=[],
            )

            assert isinstance(agent2.state[4], BHMessage)
            assert agent2.state[4].role == MessageRole.CONTEXT
            assert (
                "The format for each item in the list is the same as before."
                in agent2.state[4].content
            )
            assert (
                '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class (clarified)!"], "agent_name": "TestAgent1"}'
                in agent2.state[4].content
            )

            assert isinstance(agent2.state[5], BHMessage)
            assert agent2.state[5].role == MessageRole.USER
            assert agent2.state[5].content == agent2.state[2].content

            assert agent2.state[6] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="After receiving clarification, hello from our mocked class!",
                tool_calls=[],
            )


def test_agent_asking_beehive_question(
    test_storage, test_feedback_storage, test_printer
):
    expected_msg_contents = [
        "Hello from our mocked class!",
        "Hello from our mocked class!",
        '{"question": "Can you clarify something for me?", "invokable": "InnerBeehive", "reason": "I need this clarification please"}',
        "Hello from our mocked class (clarified)!",
        "After receiving clarification, hello from our mocked class!",
    ]
    with mock.patch("beehive.models.openai_model.OpenAIModel._client") as mocked_model:
        chat_messages = [MockChatCompletion([x]) for x in expected_msg_contents]
        mocked_model.chat.completions.create = mock.Mock(side_effect=chat_messages)

        agent0 = BeehiveAgent(
            name="TestAgent0",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        with mock.patch(
            "beehive.invokable.beehive.Beehive._create_router"
        ) as mocked_router:
            mocked_router_instances = [
                mock.Mock(name="RouterInner"),
                mock.Mock(name="RouterOuter"),
            ]

            # Next tasks for inner router (the nested Beehive will use a pre-defined
            # route).
            inner_router_next_tasks = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task in the inner beehive."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 2\'s first task."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"invokable": "TestAgent2"}',
                    ),
                ],
            ]
            mocked_router_instances[0]._invoke.side_effect = inner_router_next_tasks

            # Next agents for outer router
            outer_router_next_agents = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is the inner Beehive\'s task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent0", "reason": "TestAgent0 can accomplish this task.", "task": "This is the TestAgent0\'s task."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "FINISH", "reason": "We are done with this task.", "task": ""}',
                    ),
                ],
            ]
            mocked_router_instances[1]._invoke.side_effect = outer_router_next_agents

            mocked_router.side_effect = mocked_router_instances

            inner_bh = Beehive(
                name="InnerBeehive",
                backstory="You are the inner Beehive.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=FixedExecution(route=(agent1 >> agent2)),
            )
            inner_bh._db_storage = test_storage

            # We tested questioning in a DynamicExecution earlier. Test FixedExecution
            # here.
            outer_bh = Beehive(
                name="OuterBeehive",
                backstory="You are the inner Beehive.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=FixedExecution(route=(inner_bh >> agent0)),
                enable_questioning=True,
            )
            outer_bh._db_storage = test_storage
            output = outer_bh.invoke("test task", stdout_printer=test_printer)

            # We expect there to be 5 output messages: two messages from the inner_bh
            # responses, one clarification question from agent0 --> inner_bh, inner_bh's
            # response to the clarification question, and then finally agent0's
            # response.
            output_elts = output["messages"]
            assert len(output_elts) == 5

            # First element
            first_elt = output_elts[0]
            assert isinstance(first_elt, BHStateElt)
            assert first_elt.invokable == agent1
            assert (
                first_elt.task == "This is agent 1's first task in the inner beehive."
            )
            assert len(first_elt.completion_messages) == 1
            assert isinstance(first_elt.completion_messages[0], BHMessage)
            assert first_elt.completion_messages[0].content == expected_msg_contents[0]

            second_elt = output_elts[1]
            assert isinstance(second_elt, BHStateElt)
            assert second_elt.invokable == agent2
            assert second_elt.task == "This is agent 2's first task."
            assert len(second_elt.completion_messages) == 1
            assert isinstance(second_elt.completion_messages[0], BHMessage)
            assert second_elt.completion_messages[0].content == expected_msg_contents[1]

            third_elt = output_elts[2]
            assert isinstance(third_elt, BHStateElt)
            assert third_elt.invokable == agent0
            assert "This is the TestAgent0's task." in third_elt.task
            assert (
                "IF YOU WANT TO ASK A QUESTION, format the output as a JSON instance that conforms to the JSON schema below."
                in third_elt.task
            )
            assert len(third_elt.completion_messages) == 1
            assert isinstance(third_elt.completion_messages[0], BHMessage)
            assert third_elt.completion_messages[0].content == expected_msg_contents[2]

            fourth_elt = output_elts[3]
            assert isinstance(fourth_elt, BHStateElt)
            assert fourth_elt.invokable == agent2
            assert fourth_elt.task == "Can you clarify something for me?"
            assert len(fourth_elt.completion_messages) == 1
            assert isinstance(fourth_elt.completion_messages[0], BHMessage)
            assert fourth_elt.completion_messages[0].content == expected_msg_contents[3]

            fifth_elt = output_elts[4]
            assert isinstance(fifth_elt, BHStateElt)

            # The original element isn't techni
            assert fifth_elt.invokable == agent0
            assert fifth_elt.task == third_elt.task
            assert len(fifth_elt.completion_messages) == 1
            assert isinstance(fifth_elt.completion_messages[0], BHMessage)
            assert fifth_elt.completion_messages[0].content == expected_msg_contents[4]

            # Agent states. Agent 1's state should be pretty simple — it should just
            # contain the system message, the task, and the response.
            assert len(agent1.state) == 3
            assert agent1.state[0] == BHMessage(
                role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
            )
            assert agent1.state[1] == BHMessage(
                role=MessageRole.USER,
                content="This is agent 1's first task in the inner beehive.",
            )
            assert agent1.state[2] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Agent 2's state. This will include:
            #   System message
            #   Context message from agent 1
            #   User query
            #   First response
            #   Clarifying question from agent 0 -- no context here, because nothing
            #     *new* was invoked between agent 2's response and the clarifiying
            #     question.
            #   Response to clarifying question
            assert len(agent2.state) == 6
            assert agent2.state[0] == BHMessage(
                role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
            )
            assert isinstance(agent2.state[1], BHMessage)
            assert agent2.state[1].role == MessageRole.CONTEXT
            assert (
                '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}'
                in agent2.state[1].content
            )
            assert agent2.state[2] == BHMessage(
                role=MessageRole.USER, content="This is agent 2's first task."
            )
            assert agent2.state[3] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )
            assert agent2.state[4] == BHMessage(
                role=MessageRole.USER, content="Can you clarify something for me?"
            )
            assert agent2.state[5] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="Hello from our mocked class (clarified)!",
            )

            # Agent 0's state. This will include
            # System message
            # Context message from agents 1 and 2
            # User query
            # Clarifying question to InnerBeehive
            # Context message with additional clarification
            # User query
            # Response to use query
            assert len(agent0.state) == 7
            assert agent0.state[0] == BHMessage(
                role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
            )

            assert isinstance(agent0.state[1], BHMessage)
            assert agent0.state[1].role == MessageRole.CONTEXT
            assert (
                '- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}'
                in agent0.state[1].content
            )
            assert (
                '\n- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent2"}'
                in agent0.state[1].content
            )

            assert isinstance(agent0.state[2], BHMessage)
            assert agent0.state[2].role == MessageRole.USER
            assert "This is the TestAgent0's task" in agent0.state[2].content
            assert (
                "IF YOU WANT TO ASK A QUESTION, format the output as a JSON instance that conforms to the JSON schema below."
                in agent0.state[2].content
            )

            assert agent0.state[3] == BHMessage(
                role=MessageRole.QUESTION,
                content='{"question": "Can you clarify something for me?", "invokable": "InnerBeehive", "reason": "I need this clarification please"}',
            )

            assert isinstance(agent0.state[4], BHMessage)
            assert agent0.state[4].role == MessageRole.CONTEXT
            assert (
                '- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class (clarified)!"], "agent_name": "TestAgent2"}'
                in agent0.state[4].content
            )

            assert agent0.state[5] == agent0.state[2]

            assert agent0.state[6] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="After receiving clarification, hello from our mocked class!",
            )


def test_beehive_asking_agent_question(
    test_storage, test_feedback_storage, test_printer
):
    expected_msg_contents = [
        "Hello from our mocked class!",
        "Hello from our mocked class!",
        '{"question": "Can you clarify something for me?", "invokable": "TestAgent0", "reason": "I need this clarification please"}',
        "Hello from our mocked class (clarified)!",
        "After receiving clarification, hello from our mocked class!",
    ]
    with mock.patch("beehive.models.openai_model.OpenAIModel._client") as mocked_model:
        chat_messages = [MockChatCompletion([x]) for x in expected_msg_contents]
        mocked_model.chat.completions.create = mock.Mock(side_effect=chat_messages)

        agent0 = BeehiveAgent(
            name="TestAgent0",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent1 = BeehiveAgent(
            name="TestAgent1",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        agent2 = BeehiveAgent(
            name="TestAgent2",
            backstory="You are a helpful AI assistant.",
            model=OpenAIModel(model="gpt-3.5-turbo"),
        )
        with mock.patch(
            "beehive.invokable.beehive.Beehive._create_router"
        ) as mocked_router:
            mocked_router_instances = [
                mock.Mock(name="RouterInner"),
                mock.Mock(name="RouterOuter"),
            ]

            # Next tasks for inner router (the nested Beehive will use a pre-defined
            # route).
            inner_router_next_tasks = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is agent 1\'s first task in the inner beehive."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent2", "reason": "Agent2 can complete this task.", "task": "This is agent 2\'s first task in the inner beehive."}',
                    ),
                ],
            ]
            mocked_router_instances[0]._invoke.side_effect = inner_router_next_tasks

            # Next agents for outer router
            outer_router_next_agents = [
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "TestAgent0", "reason": "TestAgent0 can accomplish this task.", "task": "This is the TestAgent0\'s task."}',
                    ),
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"task": "This is the inner Beehive\'s task."}',
                    )
                ],
                [
                    BHMessage(
                        role=MessageRole.ASSISTANT,
                        content='{"agent": "FINISH", "reason": "We are done with this task.", "task": ""}',
                    ),
                ],
            ]
            mocked_router_instances[1]._invoke.side_effect = outer_router_next_agents

            mocked_router.side_effect = mocked_router_instances

            inner_bh = Beehive(
                name="InnerBeehive",
                backstory="You are the inner Beehive.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=DynamicExecution(
                    entrypoint=agent1, edges=[(agent1 >> agent2)]
                ),
                enable_questioning=True,
            )
            inner_bh._db_storage = test_storage

            # We tested questioning in a DynamicExecution earlier. Test FixedExecution
            # here.
            outer_bh = Beehive(
                name="OuterBeehive",
                backstory="You are the inner Beehive.",
                model=OpenAIModel(model="gpt-3.5-turbo"),
                execution_process=FixedExecution(route=(agent0 >> inner_bh)),
                enable_questioning=True,
            )
            outer_bh._db_storage = test_storage
            output = outer_bh.invoke("test task", stdout_printer=test_printer)

            # We expect there to be 5 output messages: first message from agent 0, first
            # message from agent 1, clarification question from agent 2, response by
            # agent 0, and final response by agent 2.
            output_elts = output["messages"]
            assert len(output_elts) == 5

            # # First element
            first_elt = output_elts[0]
            assert isinstance(first_elt, BHStateElt)
            assert first_elt.invokable == agent0
            assert first_elt.task == "This is the TestAgent0's task."
            assert len(first_elt.completion_messages) == 1
            assert isinstance(first_elt.completion_messages[0], BHMessage)
            assert first_elt.completion_messages[0].content == expected_msg_contents[0]

            second_elt = output_elts[1]
            assert isinstance(second_elt, BHStateElt)
            assert second_elt.invokable == agent1
            assert (
                second_elt.task == "This is agent 1's first task in the inner beehive."
            )
            assert len(second_elt.completion_messages) == 1
            assert isinstance(second_elt.completion_messages[0], BHMessage)
            assert second_elt.completion_messages[0].content == expected_msg_contents[1]

            third_elt = output_elts[2]
            assert isinstance(third_elt, BHStateElt)
            assert third_elt.invokable == agent2
            assert (
                "This is agent 2's first task in the inner beehive." in third_elt.task
            )
            assert (
                "IF YOU WANT TO ASK A QUESTION, format the output as a JSON instance that conforms to the JSON schema below."
                in third_elt.task
            )
            assert len(third_elt.completion_messages) == 1
            assert isinstance(third_elt.completion_messages[0], BHMessage)
            assert third_elt.completion_messages[0].content == expected_msg_contents[2]

            fourth_elt = output_elts[3]
            assert isinstance(fourth_elt, BHStateElt)
            assert fourth_elt.invokable == agent0
            assert fourth_elt.task == "Can you clarify something for me?"
            assert len(fourth_elt.completion_messages) == 1
            assert isinstance(fourth_elt.completion_messages[0], BHMessage)
            assert fourth_elt.completion_messages[0].content == expected_msg_contents[3]

            fifth_elt = output_elts[4]
            assert isinstance(fifth_elt, BHStateElt)
            assert fifth_elt.invokable == agent2
            assert fifth_elt.task == third_elt.task
            assert len(fifth_elt.completion_messages) == 1
            assert isinstance(fifth_elt.completion_messages[0], BHMessage)
            assert fifth_elt.completion_messages[0].content == expected_msg_contents[4]

            # Agent 0's state. This will include
            #   System message
            #   User query
            #   First response
            #   Context message with context from agent 1
            #   Clarifying question from InnerBeehive
            #   Response to clarifying question
            assert len(agent0.state) == 6
            assert agent0.state[0] == BHMessage(
                role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
            )
            assert agent0.state[1] == BHMessage(
                role=MessageRole.USER, content="This is the TestAgent0's task."
            )
            assert agent0.state[2] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )
            assert isinstance(agent0.state[3], BHMessage)
            assert agent0.state[3].role == MessageRole.CONTEXT
            assert (
                '- {"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}'
                in agent0.state[3].content
            )
            assert agent0.state[4] == BHMessage(
                role=MessageRole.USER, content="Can you clarify something for me?"
            )
            assert agent0.state[5] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="Hello from our mocked class (clarified)!",
            )

            # Agent states. Agent 1's state should be pretty simple — it should just
            # contain the system message, context message, the task, and the response.
            assert len(agent1.state) == 4
            assert agent1.state[0] == BHMessage(
                role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
            )
            assert isinstance(agent1.state[1], BHMessage)
            assert agent1.state[1].role == MessageRole.CONTEXT
            assert agent1.state[2] == BHMessage(
                role=MessageRole.USER,
                content="This is agent 1's first task in the inner beehive.",
            )
            assert agent1.state[3] == BHMessage(
                role=MessageRole.ASSISTANT, content="Hello from our mocked class!"
            )

            # Agent 2's state. This will include:
            #   System message
            #   Context message from agent 0 & 1
            #   User query
            #   Clarifying question to agent 0
            #   Context message with clarification
            #   User query
            #   Response
            assert len(agent2.state) == 7
            assert agent2.state[0] == BHMessage(
                role=MessageRole.SYSTEM, content="You are a helpful AI assistant."
            )
            assert isinstance(agent2.state[1], BHMessage)
            assert agent2.state[1].role == MessageRole.CONTEXT
            assert (
                '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent0"}'
                in agent2.state[1].content
            )
            assert (
                '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class!"], "agent_name": "TestAgent1"}'
                in agent2.state[1].content
            )
            assert isinstance(agent2.state[2], BHMessage)
            assert agent2.state[2].role == MessageRole.USER
            assert (
                "This is agent 2's first task in the inner beehive."
                in agent2.state[2].content
            )
            assert agent2.state[3] == BHMessage(
                role=MessageRole.QUESTION,
                content='{"question": "Can you clarify something for me?", "invokable": "TestAgent0", "reason": "I need this clarification please"}',
            )
            assert isinstance(agent2.state[4], BHMessage)
            assert agent2.state[4].role == MessageRole.CONTEXT
            assert (
                '{"agent_backstory": "You are a helpful AI assistant.", "agent_messages": ["Hello from our mocked class (clarified)!"], "agent_name": "TestAgent0"}'
                in agent2.state[4].content
            )
            assert agent2.state[5] == agent2.state[2]
            assert agent2.state[6] == BHMessage(
                role=MessageRole.ASSISTANT,
                content="After receiving clarification, hello from our mocked class!",
            )
