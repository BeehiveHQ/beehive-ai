from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Gender(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    not_given = "not_given"


class TestModel(BaseModel):
    name: Gender = Field(default=Gender.male, description="test")
    test_object: dict[str, Any]
