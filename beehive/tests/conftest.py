import shutil
from pathlib import Path

from constants import INTERNAL_FOLDER_PATH
from memory.db_storage import DbStorage
from memory.feedback_storage import FeedbackStorage


def pytest_sessionstart():
    DbStorage(
        db_uri=f"sqlite:///{Path(INTERNAL_FOLDER_PATH).resolve()}/test_beehive.db"
    )
    FeedbackStorage(client_path=f"{Path(INTERNAL_FOLDER_PATH).resolve()}/test_feedback")


def pytest_sessionfinish():
    Path.unlink(Path(INTERNAL_FOLDER_PATH).resolve() / "test_beehive.db")
    shutil.rmtree(f"{Path(INTERNAL_FOLDER_PATH).resolve()}/test_feedback")
