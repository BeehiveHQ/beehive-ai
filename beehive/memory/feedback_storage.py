from pathlib import Path
from uuid import uuid4

from chromadb import PersistentClient
from chromadb.api.types import GetResult

from beehive.constants import INTERNAL_FOLDER_PATH
from beehive.invokable.base import Feedback, Invokable
from beehive.invokable.types import EmbeddingDistance
from beehive.memory.db_storage import TaskModel
from beehive.models.base import BHEmbeddingModel

# Constants
FEEDBACK_DB_PATH = f"{Path(INTERNAL_FOLDER_PATH).resolve()}/feedback"


# Storage class
class FeedbackStorage:
    client_path: str
    embedding_distance: EmbeddingDistance

    def __init__(
        self,
        client_path: str | None = None,
        embedding_distance: EmbeddingDistance = EmbeddingDistance.L2,
    ):
        self.client_path = client_path if client_path else FEEDBACK_DB_PATH
        self.embedding_distance = embedding_distance
        self.client = PersistentClient(path=self.client_path)
        self.client.get_or_create_collection(
            name="tasks", metadata={"hnsw:space": self.embedding_distance.value}
        )
        self.client.get_or_create_collection(
            name="feedback", metadata={"hnsw:space": self.embedding_distance.value}
        )

    def embed_task_and_record_feedback(
        self,
        invokable: Invokable,
        task: TaskModel,
        feedback: Feedback | None,
        embedder: BHEmbeddingModel | None,
    ):
        if feedback:
            # The embedding model must be specified
            assert embedder

            feedback_suggestions = feedback.suggestions
            feedback_confidence = feedback.confidence

            # Record feedback
            feedback_ids: list[str] = []
            feedback_collection = self.client.get_collection("feedback")
            for x in feedback_suggestions:
                # Unique UUID for feedback
                feedback_id = str(uuid4())
                feedback_ids.append(feedback_id)

                # Store feedback in collection.
                x_embeddings = embedder.get_embeddings(x)
                feedback_collection.add(
                    documents=[x],
                    embeddings=[x_embeddings],
                    ids=[feedback_id],
                    metadatas=[
                        {
                            "invokable": invokable.name,
                            "task": task.id,
                            "confidence": feedback_confidence,
                        }
                    ],
                )

            # Embed task
            task_embeddings = embedder.get_embeddings(task.content)
            tasks_collection = self.client.get_collection("tasks")
            tasks_collection.add(
                documents=[task.content],
                embeddings=[task_embeddings],
                ids=[task.id],
                metadatas=[
                    {
                        "invokable": invokable.name,
                        "feedback_ids": " || ".join(feedback_ids),
                    }
                ],
            )

    def grab_feedback_from_similar_tasks(
        self,
        invokable: Invokable,
        task: str,
        embedder: BHEmbeddingModel,
        n_results: int = 10,
    ) -> GetResult | None:
        # Grab similar tasks
        embeddings = embedder.get_embeddings(task)
        tasks_collection = self.client.get_collection("tasks")
        feedback_collection = self.client.get_collection("feedback")
        try:
            similar_tasks_results = tasks_collection.query(
                query_embeddings=[embeddings],
                n_results=n_results,
                where={"invokable": invokable.name},
            )
            # Grab feedback
            ids: list[str] = []
            for metadatas in similar_tasks_results["metadatas"]:
                for feedback_dict in metadatas:
                    ids.extend(feedback_dict["feedback_ids"].split(" || "))
            res = feedback_collection.get(
                ids=ids,
                where={
                    "invokable": invokable.name,
                },
            )
            return res

        # TODO I'm seeing a `StopIteration` error. I think it's because the collections
        # are empty, but investigate further.
        except Exception:
            return None

    def grab_feedback_for_task(self, task_id: str) -> GetResult:
        feedback_collection = self.client.get_collection("feedback")
        res = feedback_collection.get(
            where={"task": task_id},
        )
        return res
