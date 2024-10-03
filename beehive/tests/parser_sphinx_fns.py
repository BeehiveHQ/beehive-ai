from typing import Mapping

from beehive.tests.parser_common import TestModel


def sphinx_fetch_smalltable_rows(
    table_handle: TestModel,
    keys: list[TestModel],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    :param table_handle: An TestModel instance.
    :type table_handle: An TestModel instance.
    :param keys: A sequence of strings representing the key of each table row to
        fetch. String keys will be UTF-8 encoded.
    :type keys: list[TestModel]
    :param require_all_keys: If True only rows with values set for all keys will be returned.
    :type require_all_keys: bool
    :return: a dict mapping keys to the corresponding table row data fetched. Each
        row is represented as a tuple of strings. For example:
        {b'Serak': ('Rigel VII', 'Preparer'),
        b'Zim': ('Irk', 'Invader'),
        b'Lrrr': ('Omicron Persei 8', 'Emperor')}
    """
    return {}


def bad_sphinx_fetch_smalltable_rows_extra_arg(
    table_handle: TestModel,
    keys: list[TestModel],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    :param table_handle: An TestModel instance.
    :type table_handle: An TestModel instance.
    :param this_should_not_exist: An extra argument that should not exist!
    :type this_should_not_exist: None
    :param keys: A sequence of strings representing the key of each table row to
        fetch. String keys will be UTF-8 encoded.
    :type keys: list[TestModel]
    :param require_all_keys: If True only rows with values set for all keys will be returned.
    :type require_all_keys: bool
    :return: a dict mapping keys to the corresponding table row data fetched. Each
        row is represented as a tuple of strings. For example:
        {b'Serak': ('Rigel VII', 'Preparer'),
        b'Zim': ('Irk', 'Invader'),
        b'Lrrr': ('Omicron Persei 8', 'Emperor')}
    """
    return {}


def bad_sphinx_fetch_smalltable_rows_bad_format(
    table_handle: TestModel,
    keys: list[TestModel],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    :param table_handle: An TestModel instance.
    :type table_handle: An TestModel instance.
    :params keys: A sequence of strings representing the key of each table row to
        fetch. String keys will be UTF-8 encoded.
    :type keys: list[TestModel]
    :param require_all_keys: If True only rows with values set for all keys will be returned.
    :type require_all_keys: bool
    :return: a dict mapping keys to the corresponding table row data fetched. Each
        row is represented as a tuple of strings. For example:
        {b'Serak': ('Rigel VII', 'Preparer'),
        b'Zim': ('Irk', 'Invader'),
        b'Lrrr': ('Omicron Persei 8', 'Emperor')}
    """
    return {}
