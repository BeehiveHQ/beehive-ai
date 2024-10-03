import pytest

from beehive.tests.parser_google_fns import google_fetch_smalltable_rows
from beehive.tests.parser_sphinx_fns import sphinx_fetch_smalltable_rows
from beehive.tools.base import create_parser
from beehive.tools.google import GoogleParser
from beehive.tools.sphinx import SphinxParser


def test_docstring_detection():
    parser = create_parser(sphinx_fetch_smalltable_rows)
    assert isinstance(parser, SphinxParser)

    parser = create_parser(google_fetch_smalltable_rows)
    assert isinstance(parser, GoogleParser)

    # Unknown docstring
    def function_with_unknown_docstring(x: str) -> None:
        """Docstring does not follow Sphinx or Google conventions.

        args:
            - x (str): variable 1
        returns:
            None
        """

    with pytest.raises(ValueError) as cm:
        create_parser(function_with_unknown_docstring)
    expected_message = "Could not detect docstring format!"
    assert expected_message in str(cm)
