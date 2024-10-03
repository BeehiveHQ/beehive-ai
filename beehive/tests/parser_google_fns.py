from typing import Mapping

from beehive.tests.parser_common import TestModel


def google_fetch_smalltable_rows(
    table_handle: TestModel,
    keys: list[TestModel],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    Args:
    table_handle:
        An TestModel instance.
    keys:
        A sequence of strings representing the key of each table row to
        fetch. String keys will be UTF-8 encoded.
    require_all_keys:
        If True only rows with values set for all keys will be returned.

    Returns:
    A dict mapping keys to the corresponding table row data
    fetched. Each row is represented as a tuple of strings. For
    example:

    {b'Serak': ('Rigel VII', 'Preparer'),
    b'Zim': ('Irk', 'Invader'),
    b'Lrrr': ('Omicron Persei 8', 'Emperor')}

    Returned keys are always bytes.  If a key from the keys argument is
    missing from the dictionary, then that row was not found in the
    table (and require_all_keys must have been False).

    Raises:
    IOError: An error occurred accessing the smalltable.
    """
    return {}


def bad_google_fetch_smalltable_rows_extra_arg(
    table_handle: TestModel,
    keys: list[TestModel],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    Args:
    table_handle:
        An TestModel instance.
    this_should_not_exist:
        An extra argument that should not exist.
    keys:
        A sequence of strings representing the key of each table row to
        fetch. String keys will be UTF-8 encoded.
    require_all_keys:
        If True only rows with values set for all keys will be returned.

    Returns:
    A dict mapping keys to the corresponding table row data
    fetched. Each row is represented as a tuple of strings. For
    example:

    {b'Serak': ('Rigel VII', 'Preparer'),
    b'Zim': ('Irk', 'Invader'),
    b'Lrrr': ('Omicron Persei 8', 'Emperor')}

    Returned keys are always bytes.  If a key from the keys argument is
    missing from the dictionary, then that row was not found in the
    table (and require_all_keys must have been False).

    Raises:
    IOError: An error occurred accessing the smalltable.
    """
    return {}


def bad_google_fetch_smalltable_rows_bad_format(
    table_handle: TestModel,
    keys: list[TestModel],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle. String keys will be UTF-8 encoded.

    Args:
    table_handle
        An TestModel instance.
    keys:
        A sequence of strings representing the key of each table row to
        fetch. String keys will be UTF-8 encoded.
    require_all_keys:
        If True only rows with values set for all keys will be returned.

    Returns:
    A dict mapping keys to the corresponding table row data
    fetched. Each row is represented as a tuple of strings. For
    example:

    {b'Serak': ('Rigel VII', 'Preparer'),
    b'Zim': ('Irk', 'Invader'),
    b'Lrrr': ('Omicron Persei 8', 'Emperor')}

    Returned keys are always bytes.  If a key from the keys argument is
    missing from the dictionary, then that row was not found in the
    table (and require_all_keys must have been False).

    Raises:
    IOError: An error occurred accessing the smalltable.
    """
    return {}
