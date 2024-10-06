# Tools

Tools give language models the ability to interact with the outside world. This enables agents browse the web, make requests against an API, or execute any arbitrary function.

In Beehive, tools are simply functions with a docstring and type-hints.

!!! warning
    If a function doesn't have both a docstring and type-hints, Beehive will throw an error!

Docstrings are used to grab the tool description as well as the description for each of arguments.

Beehive currently requires that docstring follow the sphinx or google standards. You can find the specifications here:

- Sphinx: [https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)
- Google: [https://google.github.io/styleguide/pyguide.html#383-functions-and-methods](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)

Under the hood, Beehive parses the docstring and type hint and constructs a Pydantic `BaseModel` representing the function. The serialized `BaseModel` is then passed to the `BHChatModel` powering the `Invokable`, which then expresses the intent to call a specific tool in their response. Beehive handles interpreting this intent and actually calling the function.

Here are some examples of tools with different docstrings.


??? example "Sphinx docstring"

    ```python
    class SearchDepth(str, Enum):
        BASIC = "basic"
        ADVANCED = "advanced"


    def tavily_search_tool(
        query: str,
        search_depth: SearchDepth = SearchDepth.BASIC,
        include_images: bool = False,
        include_answer: bool = True,
        include_raw_content: bool = False,
        max_results: int = 5,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
    ):
        """
        Use this as a search engine optimized for comprehensive, accurate, and trusted
        results. Very useful for when you need to answer questions about current events, or
        if you need search the web for information.

        :param query: search query
        :type query: str
        :param search_depth: depth of the search; basic should be used for quick results,
            and advanced for indepth high quality results but longer response time, defaults
            to basic
        :type search_depth: class:`test.SearchDepth`
        :param include_images: include a list of query related images in the response,
            defaults to False
        :type include_images: bool
        :param include_answer: include answers in the search results, defaults to True
        :type include_answer: bool
        :param include_raw_content: include raw content in the search results, defaults to
            False
        :type include_raw_content: bool
        :param max_results: number of maximum search results to return, defaults to 5.
        :type max_results: int
        :param include_domains: list of domains to specifically include in the search
            results, defaults to None
        :type include_domains: list[str], optional
        :param exclude_domains: list of domains to specifically exclude from the search
            results, defaults to None
        :type exclude_domains: list[str], optional
        """
        base_url = "https://api.tavily.com/"
        endpoint = "search"
        resp = requests.post(
            f"{base_url}{endpoint}",
            json={
                "api_key": "<tavily_api_key>",
                "query": query,
                "search_depth": search_depth,
                "include_images": include_images,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
                "max_results": max_results,
                "include_domains": include_domains,
                "exclude_domains": exclude_domains,
            },
        )
        try:
            return resp.json()["answer"]
        except json.JSONDecodeError as e:
            logger.error(e)
            return "Could not execute the Tavily search...Try again!"
    ```

??? example "Google docstring"

    ```python
    class Gender(str, Enum):
        male = "male"
        female = "female"
        other = "other"
        not_given = "not_given"


    class TestModel(BaseModel):
        name: Gender = Field(default=Gender.male, description="test")
        test_object: dict[str, Any]


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
    ```
