import logging

from rich.logging import RichHandler

logging.basicConfig(
    level="WARN",
    format="%(message)s",
    handlers=[RichHandler()],
)
logger = logging.getLogger(__file__)
