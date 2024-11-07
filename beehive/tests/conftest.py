import shutil
from pathlib import Path

from beehive.constants import INTERNAL_FOLDER_PATH
from beehive.memory.db_storage import DbStorage
from beehive.memory.feedback_storage import FeedbackStorage


def pytest_sessionstart():
    DbStorage(
        db_uri=f"sqlite:///{Path(INTERNAL_FOLDER_PATH).resolve()}/test_beehive.db"
    )
    FeedbackStorage(client_path=f"{Path(INTERNAL_FOLDER_PATH).resolve()}/test_feedback")

    # Reset the prompts
    shutil.rmtree(f"{Path(INTERNAL_FOLDER_PATH).resolve()}/prompts")


def pytest_sessionfinish():
    Path.unlink(Path(INTERNAL_FOLDER_PATH).resolve() / "test_beehive.db")
    shutil.rmtree(f"{Path(INTERNAL_FOLDER_PATH).resolve()}/test_feedback")
