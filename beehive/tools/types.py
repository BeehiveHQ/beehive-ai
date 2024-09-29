from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from pydantic import BaseModel


class DocstringFormat(str, Enum):
    SPHINX: Final = "sphinx"
    GOOGLE: Final = "google"


@dataclass
class FunctionSpec:
    name: str
    description: str
    params: type[BaseModel] | None = None

    def serialize(self) -> dict[str, str | dict[str, Any]]:
        if self.params:
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": self.params.model_json_schema(),
                },
            }
        else:
            raise ValueError("BaseModel encapsulating function params not specified!")
