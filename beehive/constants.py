import os
from pathlib import Path

# Folder for databases, artifacts, and other such things
INTERNAL_FOLDER_PATH = Path(os.path.expanduser("~/.beehive"))
if not INTERNAL_FOLDER_PATH.is_dir():
    INTERNAL_FOLDER_PATH.mkdir()
