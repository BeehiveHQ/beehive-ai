from typing import Any


class MockPrinter:
    def __init__(self, *args, **kwargs):
        self._current_beehive_panel = None

    def create_console(self, *args, **kwargs):
        return None

    def register_beehive(self, *args, **kwargs):
        return None

    def unregister_beehive(self, *args, **kwargs):
        return None

    def print_standard(self, *args, **kwargs):
        return None

    def beehive_label(self, *args, **kwargs):
        return None

    def print_router_text(self, *args, **kwargs):
        return None

    def invokable_label_text(self, *args, **kwargs):
        return None

    def separation_rule(self, *args, **kwargs):
        return None


class MockMessage:
    content: str | None
    tool_calls: list[Any]

    def __init__(self, content: str | None) -> None:
        self.content = content
        self.tool_calls = []


class MockChoice:
    message: MockMessage

    def __init__(self, message: MockMessage) -> None:
        self.message = message


class MockChatCompletion:
    choices: list[MockChoice]

    def __init__(self, inputs: list[str | None]) -> None:
        self.choices = [MockChoice(MockMessage(inp)) for inp in inputs]

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        """For Langchain"""
        return {
            "choices": [
                {"message": {"role": "assistant", "content": c.message.content}}
            ]
            for c in self.choices
        }


class MockCompletions:
    def __init__(self) -> None:
        pass

    def create(self, *args, **kwargs) -> MockChatCompletion:
        return MockChatCompletion(inputs=["Hello from our mocked class!"])


class MockChat:
    completions: MockCompletions

    def __init__(self, *args, **kwargs) -> None:
        self.completions = MockCompletions()


class MockOpenAIClient:
    chat: MockChat

    def __init__(self, *args, **kwargs) -> None:
        self.chat = MockChat()


class MockEmbeddingModel:
    def __init__(self) -> None:
        pass

    def get_embeddings(self, *args, **kwargs):
        return [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


class MockAsyncResult:
    def __init__(self, result: Any):
        self.result = result

    def ready(self):
        return True

    def get(self):
        return self.result
