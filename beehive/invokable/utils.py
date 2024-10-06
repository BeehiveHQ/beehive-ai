"""Util functions for invokables.
"""
import re
from typing import Any, Callable, Tuple

from beehive.invokable.types import AnyMessageSequence
from beehive.tools.base import BHTool
from beehive.tools.types import DocstringFormat


def _construct_bh_tools_map(
    tools: list[Callable[..., Any]],
    docstring_format: DocstringFormat | None,
) -> Tuple[dict[str, BHTool], dict[str, dict[str, str | dict[str, Any]]]]:
    if tools:
        _tools_map = {
            x.__name__: BHTool(func=x, docstring_format=docstring_format) for x in tools
        }
        _tools_serialized = {
            _name: _tool.derive_json_specification()
            for _name, _tool in _tools_map.items()
        }
        return _tools_map, _tools_serialized
    return {}, {}


def _convert_messages_to_string(messages: AnyMessageSequence, delim: str = " ") -> str:
    return delim.join(filter(None, [str(m.content) for m in messages]))


def _process_json_output(content: str) -> str:
    # Sometimes, the router wraps the JSON return in ```json...```. Remove these.
    content = re.sub(r"\n", "", content)
    content = re.sub(r"^(\`{3}json)(.*)(\`{3})$", r"\2", content)
    return content
