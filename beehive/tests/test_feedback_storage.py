from pathlib import Path

import pytest
from sqlalchemy import select

from beehive.constants import INTERNAL_FOLDER_PATH
from beehive.invokable.agent import BeehiveAgent
from beehive.invokable.base import Feedback
from beehive.memory.db_storage import DbStorage, TaskModel
from beehive.memory.feedback_storage import FeedbackStorage
from beehive.models.openai_model import OpenAIModel
from beehive.tests.mocks import MockEmbeddingModel


@pytest.fixture(scope="module")
def test_agent():
    test_invokable = BeehiveAgent(
        name="TestAgent",
        backstory="You are a helpful AI assistant.",
        model=OpenAIModel(model="gpt-3.5-turbo"),
    )
    return test_invokable


@pytest.fixture(scope="module")
def test_db_storage():
    # Create a SQLLite database on the local machine.
    db_storage = DbStorage(
        db_uri=f"sqlite:///{Path(INTERNAL_FOLDER_PATH).resolve()}/test_beehive.db"
    )
    return db_storage


@pytest.fixture(scope="module")
def test_feedback_storage():
    # Create a ChromaDB database on the local machine.
    fb_storage = FeedbackStorage(
        client_path=f"{Path(INTERNAL_FOLDER_PATH).resolve()}/test_feedback"
    )
    return fb_storage


def test_embed_and_record_feedback(
    test_agent: BeehiveAgent,
    test_db_storage: DbStorage,
    test_feedback_storage: FeedbackStorage,
):
    task_id = test_db_storage.add_task(
        task="This is an example task.",
        invokable=test_agent,
    )
    task_obj = test_db_storage.get_model_objects(
        select(TaskModel).where(TaskModel.id == task_id)
    )[0]
    test_feedback_storage.embed_task_and_record_feedback(
        invokable=test_agent,
        task=task_obj,
        feedback=Feedback(confidence=5, suggestions=["Suggestion 1", "Suggestion 2"]),
        embedder=MockEmbeddingModel(),  # type: ignore
    )

    # Tasks collection should contain the task
    task_output = test_feedback_storage.client.get_collection("tasks").query(
        query_embeddings=[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        where={
            "invokable": test_agent.name,
        },
    )
    task_documents = task_output["documents"]
    task_metadatas = task_output["metadatas"]
    assert len(task_documents) > 0
    assert len(task_metadatas) > 0
    assert "This is an example task." in task_documents[0]

    # Metadatas is a list of dictionaries. We only supply a single dictionary.
    metadata = task_metadatas[0][0]
    assert "feedback_ids" in metadata
    assert "invokable" in metadata
    assert metadata["invokable"] == "TestAgent"

    # Feedback collection
    feedback_output = test_feedback_storage.client.get_collection("feedback").query(
        query_embeddings=[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        where={
            "task": task_id,
        },
    )
    fb_documents = feedback_output["documents"]
    assert len(fb_documents) > 0
    assert "Suggestion 1" in fb_documents[0]
    assert "Suggestion 2" in fb_documents[0]

    # Similar feedback
    similar_results = test_feedback_storage.grab_feedback_from_similar_tasks(
        invokable=test_agent,
        task="This is another example task.",
        embedder=MockEmbeddingModel(),  # type: ignore
    )
    assert similar_results
    similar_fb_documents = similar_results["documents"]
    assert len(similar_fb_documents) > 0
    assert "Suggestion 1" in similar_fb_documents
    assert "Suggestion 2" in similar_fb_documents
