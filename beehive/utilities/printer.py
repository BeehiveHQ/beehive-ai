from pydantic import BaseModel, Field, PrivateAttr, model_validator
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.text import Text

from beehive.invokable.types import AnyMessageSequence


class BeehivePanel(Panel):
    """Additional class for distinguishing Beehive panels from Invokable panels. Used
    for `isinstance` checks.
    """

    pass


class Printer(BaseModel):
    """
    Printer for printing pretty output to the user. This is called when the user sets
    `verbose=True` in their Beehive or invokable.
    """

    size: int | None = Field(
        default=120,
        description="The width of the terminal. Set to `None` to auto-detect width.",
    )
    _all_beehives: list[str] = PrivateAttr(default_factory=list)
    _console: Console = PrivateAttr()
    _live: Live | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def create_console(self) -> "Printer":
        self._console = Console(width=120, highlight=False)
        return self

    def register_beehive(self, beehive_name: str):
        if not self._all_beehives:
            self._console.print(self.separation_rule())
        self._all_beehives.append(beehive_name)

    def unregister_beehive(self, beehive_name: str):
        self._all_beehives.remove(beehive_name)

    def print_standard(self, message: str, style: str | None = None) -> None:
        """
        Thin wrapper around `Console.print`. We do this so that other classes can print
        using the Console without having to access the `_console` private attribute.
        """
        if style:
            self._console.print(message, style)
        else:
            self._console.print(message)

    def beehive_label(self) -> Text:
        rule_label = Text()
        for bh in self._all_beehives:
            rule_label.append(bh, style=Style(color="gold1"))
            rule_label.append(" / ", style=Style(color="grey100", dim=True))
        return rule_label

    def print_router_text(self, next_invokable: str) -> None:
        rule_label = self.beehive_label()
        rule_label.append("Router", style=Style(color="purple"))
        self._console.print(Padding(rule_label, pad=(0, 0, 1, 0)))
        self._console.print(
            f"Sending conversation to [bold]{next_invokable}[/bold] next!"
        )
        self._console.print(self.separation_rule())
        return None

    def invokable_label_text(
        self,
        invokable_name: str,
        invokable_color: str,
        task: str,
    ) -> Group:
        # The invokable text consists of the label (which indicates at which point in
        # the Beehive we are), and the task.
        rule_label = self.beehive_label()
        rule_label.append(invokable_name, style=Style(color=invokable_color))

        # Create some text for the task
        task_text = Text(task, style=Style(color="grey100", dim=True, italic=True))
        return Group(
            Padding(rule_label, pad=(0, 0, 1, 0)), Padding(task_text, pad=(0, 0, 1, 0))
        )

    def separation_rule(self) -> Rule:
        rule = Rule(style=Style(color="grey100", dim=True), characters="-")
        return rule

    def print_invokable_output(
        self,
        completion_messages: AnyMessageSequence | None = None,
    ):
        if not completion_messages:
            raise ValueError("Must specify `completion_messages` for printing!")
        self._console.print(
            Text(
                "\n\n".join(filter(None, [str(x.content) for x in completion_messages]))
            ),
        )
