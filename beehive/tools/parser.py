import inspect
import re
from collections import defaultdict
from typing import Any, Callable, Tuple

from pydantic import BaseModel, Field, PrivateAttr, create_model, model_validator

from beehive.tools.types import FunctionSpec


class SchemaParser(BaseModel):
    function: Callable[..., Any] = Field(description="Function to parse.")
    function_name: str = Field(description="Name of the function.")
    docstring: str | None = Field(
        description="Docstring, retrieved via the __doc__ attribute."
    )
    start_pattern: str = Field(
        description="Pattern indicating the beginning of the argument section.",
    )
    arg_pattern: str = Field(
        description=(
            "Pattern for argument name and description. This regex should have two"
            " capture groups — the first one should match the argument name, and the"
            " second one should match the argument description."
        ),
    )
    start_pattern_is_arg: bool = Field(
        default=False,
        description="Whether `start_pattern` is also an argument. Default is False.",
    )
    stop_pattern: str = Field(
        description="Pattern indicating the end of the argument section."
    )
    ignore_patterns: list[str] = Field(
        default_factory=list,
        description="List of patterns to ignore in the argument section.",
    )

    _start_regex: re.Pattern[str] = PrivateAttr()
    _arg_regex: re.Pattern[str] = PrivateAttr()
    _stop_regex: re.Pattern[str] = PrivateAttr()
    _ignore_regexes: list[re.Pattern[str]] = PrivateAttr()
    _function_name_snakecase: str = PrivateAttr()
    _spec: inspect.FullArgSpec = PrivateAttr()
    _args: list[str] = PrivateAttr()
    _defaults: dict[str, Any] = PrivateAttr()
    _model: type[BaseModel] = PrivateAttr()

    @model_validator(mode="after")
    def define_private_attr(self) -> "SchemaParser":
        self._function_name_snakecase = "".join(
            [x.title() for x in self.function_name.split("_")]
        )
        self._start_regex = re.compile(self.start_pattern)
        self._arg_regex = re.compile(self.arg_pattern)
        self._stop_regex = re.compile(self.stop_pattern)
        self._ignore_regexes: list[re.Pattern[str]] = []
        for x in self.ignore_patterns:
            self._ignore_regexes.append(re.compile(x))
        return self

    @model_validator(mode="after")
    def define_arg_spec(self) -> "SchemaParser":
        if not self.docstring:
            raise ValueError("Function must have a docstring!")

        self._spec = inspect.getfullargspec(self.function)
        if self._spec.varargs:
            raise ValueError(
                f"Cannot have *{self._spec.varargs} in function definition."
            )
        if self._spec.varkw:
            raise ValueError(
                f"Cannot have **{self._spec.varkw} in function definition."
            )

        # If the above validations have passed, there either all of the arguments should
        # be keyword-only (e.g., if the user passes in * as the first argument in the
        # function), or none of them should be.
        spec_args = self._spec.args
        spec_kwonlyargs = self._spec.kwonlyargs
        assert not (spec_args and spec_kwonlyargs)
        if spec_args:
            self._args = self._spec.args
        elif spec_kwonlyargs:
            self._args = self._spec.kwonlyargs

        # Defaults
        _defaults: dict[str, Any] = {}
        if self._spec.defaults:
            for i, x in enumerate(self._spec.defaults, 1):
                _defaults[self._spec.args[-i]] = x
        self._defaults = _defaults

        # Defaults if all arguments are keyword-only
        if self._spec.kwonlydefaults is not None:
            self._defaults.update(self._spec.kwonlydefaults)

        # All arguments must be annotated
        type_annotated_vars = set(self._spec.annotations.keys())
        type_annotated_vars.discard("return")
        missing_type_annotations = set(self._args) - type_annotated_vars
        if missing_type_annotations:
            raise ValueError(
                (
                    f"Error in function {self.function_name}. All variables must be"
                    " type-hinted! Function is missing type annotations for the"
                    f" following: {missing_type_annotations}"
                )
            )

        return self

    def parse_descriptions_from_docstring(self) -> Tuple[str, dict[str, str]]:
        # For mypy
        if not self.docstring:
            raise ValueError("Function must have a docstring!")

        # If the user uses a Pydantic model for their function arguments, then they
        # should specify descriptions in each Field's `description` argument. Otherwise,
        # they should specify the description for each field in the docstring.
        tool_description: list[str] = []

        # Instantiate flags
        flag_in_args_section: bool = False
        flag_ignore = False

        # Keep track of current parameter and the descriptions
        current_param: str = ""
        all_params_content: dict[str, list[str]] = defaultdict(list)

        # Iterate
        for idx, line in enumerate(self.docstring.split("\n"), start=1):
            line = line.strip()

            # If we encounter a pattern that we need to ignore, turn the ignore flag on.
            # We ignore all lines until we encounter the next argument.
            for pattern in self._ignore_regexes:
                if re.findall(pattern, line):
                    flag_ignore = True
                    continue

            # Check if we have passed the `start` pattern.
            if re.findall(self._start_regex, line):
                flag_in_args_section = True

                # If the `start_pattern` could also be an argument, then continue on
                # with the logic. Otherwise, continue to the next line.
                if not self.start_pattern_is_arg:
                    continue

            # Check if we've encountered an argument (only if we've passed the `start`
            # pattern).
            if flag_in_args_section:
                matches_iterable = re.findall(self._arg_regex, line)
                if matches_iterable:
                    # Make sure we set flag_ignore to False
                    flag_ignore = False

                    # If we've hit the stop pattern, break
                    if re.findall(self._stop_regex, line):
                        break

                    # Add to all_params_content dictionary
                    match = matches_iterable[0]

                    # Should be two capture groups
                    if len(match) != 2:
                        raise ValueError(
                            "`arg_pattern` should have two capture groups!"
                        )

                    # Make sure the current parameter exists in the function's signature
                    param_name = match[0].strip()
                    if param_name not in self._args:
                        raise ValueError(
                            (
                                f"Unrecognized parameter `{param_name}`. Expected one of"
                                f" {self._args}"
                            )
                        )
                    param_description = match[1].strip()
                    if param_description:
                        all_params_content[param_name].append(param_description)

                    # Keep track of the current parameter
                    current_param = param_name
                else:
                    if not current_param:
                        raise ValueError(
                            f"Error in formatting in line {idx} of docstring: `{line}`"
                        )
                    if not flag_ignore:
                        all_params_content[current_param].append(line)

            # Any line before the start of the argument section is considered to be part
            # of the function description.
            else:
                if line:
                    tool_description.append(line)

        tool_description_final = " ".join(filter(None, tool_description))
        all_params_descriptions_final = {
            k: " ".join(filter(None, v)) for k, v in all_params_content.items()
        }
        return tool_description_final, all_params_descriptions_final

    def create_pydantic_model(
        self, params_descriptions: dict[str, str]
    ) -> type[BaseModel]:
        annotations = self._spec.annotations

        # Create Pydantic model for the function
        fields: dict[str, Any] = {
            param_name: (
                param_type,
                Field(
                    default=self._defaults[param_name]
                    if param_name in self._defaults
                    else ...,
                    description=params_descriptions.get(param_name, None),
                ),
            )
            for param_name, param_type in annotations.items()
            if param_name != "return"
        }
        pydantic_model: type[BaseModel] = create_model(
            self._function_name_snakecase,
            __base__=BaseModel,
            **fields,
        )
        assert issubclass(pydantic_model, BaseModel)

        # Make sure all field descriptions are specified
        for k, v in pydantic_model.model_fields.items():
            if not v.description:
                raise ValueError(
                    (
                        f"Field `{k}` is missing a description! Descriptions are necessary in"
                        " order for the LLM to properly invoke the function as a tool. Make sure"
                        " your docstring is formatted properly!"
                    )
                )
        return pydantic_model

    def parse(self) -> FunctionSpec:
        tool_description, params_descriptions = self.parse_descriptions_from_docstring()
        self._model = self.create_pydantic_model(params_descriptions)

        # Create FunctionSpec
        spec = FunctionSpec(
            name=self.function_name, description=tool_description, params=self._model
        )
        return spec
