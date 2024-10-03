import pytest
from pydantic import BaseModel, Field, ValidationError

from beehive.tests.parser_common import TestModel
from beehive.tests.parser_google_fns import (
    bad_google_fetch_smalltable_rows_bad_format,
    bad_google_fetch_smalltable_rows_extra_arg,
    google_fetch_smalltable_rows,
)
from beehive.tests.parser_sphinx_fns import (
    bad_sphinx_fetch_smalltable_rows_bad_format,
    bad_sphinx_fetch_smalltable_rows_extra_arg,
    sphinx_fetch_smalltable_rows,
)
from beehive.tools.google import GoogleParser
from beehive.tools.sphinx import SphinxParser


def test_sphinx_parser():
    parser = SphinxParser(
        function=sphinx_fetch_smalltable_rows,
        function_name="sphinx_fetch_smalltable_rows",
        docstring=sphinx_fetch_smalltable_rows.__doc__,
    )

    # Snakecase
    assert parser._function_name_snakecase == "SphinxFetchSmalltableRows"

    # Defaults
    expected_defaults = {"require_all_keys": False}
    assert parser._defaults == expected_defaults

    # Parse
    spec = parser.parse()
    assert spec.name == "sphinx_fetch_smalltable_rows"
    assert (
        spec.description
        == "Fetches rows from a Smalltable. Retrieves rows pertaining to the given keys from the Table instance represented by table_handle. String keys will be UTF-8 encoded."
    )

    # Pydantic model
    class SphinxFetchSmalltableRows(BaseModel):
        table_handle: TestModel = Field(description="An TestModel instance.")
        keys: list[TestModel] = Field(
            description="A sequence of strings representing the key of each table row to fetch. String keys will be UTF-8 encoded."
        )
        require_all_keys: bool = Field(
            description="If True only rows with values set for all keys will be returned.",
            default=False,
        )

    assert (
        parser._model.model_json_schema()
        == SphinxFetchSmalltableRows.model_json_schema()
    )


def test_google_parser():
    parser = GoogleParser(
        function=google_fetch_smalltable_rows,
        function_name="google_fetch_smalltable_rows",
        docstring=google_fetch_smalltable_rows.__doc__,
    )

    # Snakecase
    assert parser._function_name_snakecase == "GoogleFetchSmalltableRows"

    # Defaults
    expected_defaults = {"require_all_keys": False}
    assert parser._defaults == expected_defaults

    # Parse
    spec = parser.parse()
    assert spec.name == "google_fetch_smalltable_rows"
    assert (
        spec.description
        == "Fetches rows from a Smalltable. Retrieves rows pertaining to the given keys from the Table instance represented by table_handle. String keys will be UTF-8 encoded."
    )

    # Pydantic model
    class GoogleFetchSmalltableRows(BaseModel):
        table_handle: TestModel = Field(description="An TestModel instance.")
        keys: list[TestModel] = Field(
            description="A sequence of strings representing the key of each table row to fetch. String keys will be UTF-8 encoded."
        )
        require_all_keys: bool = Field(
            description="If True only rows with values set for all keys will be returned.",
            default=False,
        )

    assert (
        parser._model.model_json_schema()
        == GoogleFetchSmalltableRows.model_json_schema()
    )


def test_validation():
    def test_function_with_vargs(*args):
        """Dumb docstring"""
        pass

    def test_function_with_varkw(**kwargs):
        """Dumb docstring"""
        pass

    def test_function_no_type_annotations(a, b):
        """Dumb docstring"""
        pass

    def test_function_no_docstring(a: str):
        pass

    # *args is not allowed
    with pytest.raises(ValidationError):
        SphinxParser(
            function=test_function_with_vargs,
            function_name="test_function_with_vargs",
            docstring=test_function_with_vargs.__doc__,
        )

    # **kwargs is not allowed
    with pytest.raises(ValidationError):
        SphinxParser(
            function=test_function_with_varkw,
            function_name="test_function_with_varkw",
            docstring=test_function_with_varkw.__doc__,
        )

    # Function needs type annotations
    with pytest.raises(ValidationError):
        SphinxParser(
            function=test_function_no_type_annotations,
            function_name="test_function_no_type_annotations",
            docstring=test_function_no_type_annotations.__doc__,
        )

    # Docstring is required
    with pytest.raises(ValidationError):
        SphinxParser(
            function=test_function_no_docstring,
            function_name="test_function_no_docstring",
            docstring=test_function_no_docstring.__doc__,
        )


def test_bad_sphinx_docstring_extra_arg():
    parser = SphinxParser(
        function=bad_sphinx_fetch_smalltable_rows_extra_arg,
        function_name="bad_sphinx_fetch_smalltable_rows_extra_arg",
        docstring=bad_sphinx_fetch_smalltable_rows_extra_arg.__doc__,
    )
    with pytest.raises(ValueError) as cm:
        parser.parse_descriptions_from_docstring()
    expected_msg = "Unrecognized parameter `this_should_not_exist`"
    assert expected_msg in str(cm)


def test_bad_sphinx_docstring_bad_format():
    parser = SphinxParser(
        function=bad_sphinx_fetch_smalltable_rows_bad_format,
        function_name="bad_sphinx_fetch_smalltable_rows_bad_format",
        docstring=bad_sphinx_fetch_smalltable_rows_bad_format.__doc__,
    )
    with pytest.raises(ValueError) as cm:
        parser.parse()
    expected_msg = "Field `keys` is missing a description!"
    assert expected_msg in str(cm)


def test_bad_google_docstring_extra_arg():
    parser = GoogleParser(
        function=bad_google_fetch_smalltable_rows_extra_arg,
        function_name="bad_google_fetch_smalltable_rows_extra_arg",
        docstring=bad_google_fetch_smalltable_rows_extra_arg.__doc__,
    )
    with pytest.raises(ValueError) as cm:
        parser.parse_descriptions_from_docstring()
    expected_msg = "Unrecognized parameter `this_should_not_exist`"
    assert expected_msg in str(cm)


def test_bad_google_docstring_bad_format():
    parser = GoogleParser(
        function=bad_google_fetch_smalltable_rows_bad_format,
        function_name="bad_google_fetch_smalltable_rows_bad_format",
        docstring=bad_google_fetch_smalltable_rows_bad_format.__doc__,
    )
    with pytest.raises(ValueError) as cm:
        parser.parse()
    expected_msg = "Error in formatting in line 7 of docstring: `table_handle`"
    assert expected_msg in str(cm)
