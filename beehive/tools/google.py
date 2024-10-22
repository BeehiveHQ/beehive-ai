"""
Assumes docstrings follow the Google style guide: https://github.com/google/styleguide
and that functions are type-hinted.
"""

from pydantic import Field

from beehive.tools.parser import SchemaParser


class GoogleParser(SchemaParser):
    start_pattern: str = Field(
        default="^Args:$",
        description="Pattern indicating the beginning of the argument section.",
    )
    arg_pattern: str = Field(
        default=r"^([A-Za-z0-9\_]+)\:(.*)$",
        description=(
            "Pattern for argument name and description. This regex should have two"
            " capture groups — the first one should match the argument name, and the"
            " second one should match the argument description."
        ),
    )
    stop_pattern: str = Field(
        default="^Returns:$",
        description="Pattern indicating the end of the argument section.",
    )
    ignore_patterns: list[str] = Field(
        default_factory=list,
        description="List of patterns to ignore in the argument section.",
    )
