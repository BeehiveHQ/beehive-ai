from beehive.prompts import CONTEXT_PROMPT_FULL, prompt_generator


def test_prompt():
    # Rendered prompts
    expected_prompt_no_variables = prompt_generator(CONTEXT_PROMPT_FULL)
    assert (
        expected_prompt_no_variables
        == CONTEXT_PROMPT_FULL.replace("{{context}}", "")[:-1]
    )  # remove the last "\n" char

    # Rendered prompt with bad variables
    expected_prompt_bad_variables = prompt_generator(
        CONTEXT_PROMPT_FULL, does_not_exist="this_does_not_exist"
    )
    assert (
        expected_prompt_bad_variables
        == CONTEXT_PROMPT_FULL.replace("{{context}}", "")[:-1]
    )  # remove the last "\n" char

    # Rendered prompt with good variables
    expected_prompt = prompt_generator(CONTEXT_PROMPT_FULL, context="test")
    assert (
        expected_prompt == CONTEXT_PROMPT_FULL.replace("{{context}}", "test")[:-1]
    )  # remove the last "\n" char
