from pydantic import Field

from beehive.tools.parser import SchemaParser


class SphinxParser(SchemaParser):
    start_pattern: str = Field(
        default="^:param.*$",
        description="Pattern indicating the beginning of the argument section.",
    )
    arg_pattern: str = Field(
        default=r"^:param\s([a-zA-Z0-9_]+):(.+)?$",
        description=(
            "Pattern for argument name and description. This regex should have two"
            " capture groups — the first one should match the argument name, and the"
            " second one should match the argument description."
        ),
    )
    start_pattern_is_arg: bool = Field(
        default=True,
        description="Whether `start_pattern` is also an argument. Default is False.",
    )
    stop_pattern: str = Field(
        default="^:return:.+$",
        description="Pattern indicating the end of the argument section.",
    )
    ignore_patterns: list[str] = Field(
        default=[r"^:type\s([a-zA-Z0-9_]+):(.+)$"],
        description="List of patterns to ignore in the argument section.",
    )
