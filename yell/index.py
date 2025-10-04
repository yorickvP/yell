import sys
from datetime import datetime
from functools import partial
from typing import Annotated, Any, Callable, Iterable, cast

import llm
import llm.cli
import llm.migrations
import rich
import rich.table
import sqlite_utils
import textual
import textual.app
import textual.command
import textual.events
import textual.widgets
import typer
from click.types import ParamType
from llm.models import AsyncChainResponse, AsyncModel
from textual.actions import SkipAction
from textual.app import ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.messages import TerminalColorTheme
from textual.screen import Screen

from .session import ChatSession

__all__ = ["app"]

from textual._ansi_sequences import ANSI_SEQUENCES_KEYS


class ShEnter:
    value = "shift+enter"


ANSI_SEQUENCES_KEYS["\x1b\r"] = (ShEnter(),)  # type: ignore[reportIndexIssue]


def models_list():
    for model_with_aliases in llm.get_models_with_aliases():
        yield model_with_aliases.model.model_id
        yield from model_with_aliases.aliases


app = typer.Typer()


def complete_models(incomplete: str):
    for name in models_list():
        if name.startswith(incomplete):
            yield name


def complete_option_names(ctx: typer.Context, incomplete: str):
    model_id = ctx.params.get("model_id") or llm.get_default_model()
    model = llm.get_model(model_id)
    for k, v in model.Options.model_fields.items():
        if k.startswith(incomplete):
            yield (k, (v.description or "").split(".")[0])


@app.command(help="🦙💬 YELL | a slightly better LLM terminal interface")
def main(
    *,
    model_id: Annotated[
        str | None, typer.Option("--model", "-m", autocompletion=complete_models)
    ] = None,
    _continue: Annotated[bool, typer.Option("--continue", "-c")] = False,
    system: Annotated[str | None, typer.Option("--system", "-s")] = None,
    conversation_id: Annotated[
        str | None, typer.Option("--conversation", "--cid")
    ] = None,
    fragments_arg: Annotated[
        list[str], typer.Option("--fragment", "-f", default_factory=list)
    ],
    system_fragments: Annotated[
        list[str] | None,
        typer.Option("--system-fragment", "--sf", default_factory=list),
    ],
    options: Annotated[
        list[Any] | None,
        typer.Option(
            "--option",
            "-o",
            click_type=cast(ParamType, (str, str)),
            autocompletion=complete_option_names,
        ),
    ] = None,
):
    # init code from cli.py
    log_path = llm.cli.logs_db_path()
    (log_path.parent).mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(log_path)
    llm.migrations.migrate(db)
    conversation = None
    if conversation_id or _continue:
        conversation = cast(
            llm.AsyncConversation | None,
            llm.cli.load_conversation(conversation_id, async_=True),
        )
        # if conversation:
        #     for resp in conversation.responses:
        #         if resp.prompt._prompt:
        #             prompt_session.history.append_string(resp.prompt._prompt)
        # todo: insert conversation history
    if model_id is None:
        if conversation:
            model_id = conversation.model.model_id
        else:
            model_id = llm.get_default_model()
    model = llm.get_async_model(model_id)
    session = ChatSession(model, conversation, db)
    validated_options = llm.cli.get_model_options(model.model_id)
    if options:
        merged_options = {**validated_options, **dict(options)}
        validated_options = model.Options.model_validate_strings(
            merged_options
        ).model_dump(exclude_none=True, exclude_unset=True)
    session.set_model(model)
    session.set_options(validated_options)

    if system:
        session.system = system
    session.attach_fragments(system_fragments or [], system=True)
    session.attach_fragments(fragments_arg, allow_attachments=True)

    app = YellApp(session)
    app.run()
    if session.conversation and session.conversation.responses:
        rich.print(f"Conversation saved as [green not bold]{session.conversation.id}")
    sys.exit(0)


class YellInput(textual.widgets.TextArea):
    app: "YellApp"

    BINDINGS = [
        Binding("enter", "accept", "Send", priority=True),
        Binding("shift+backspace", "delete_left", "Delete character left", show=False),
    ]

    def __init__(self, sc, *args, **kwargs):
        self.sc = sc
        super().__init__(*args, **kwargs)

    async def _on_key(self, event: textual.events.Key) -> None:
        if event.key == "shift+enter":
            new_event = textual.events.Key("enter", character="\r")
        elif event.key == "shift+space":
            new_event = textual.events.Key("space", character=" ")
        else:
            new_event = event
        await super()._on_key(new_event)

    async def action_accept(self):
        await self.sc.action_accept()

    def on_resize(self, message: textual.events.Resize) -> None:
        # this actually calls it twice before it stabilizes, hope that's fine
        if self.size != message.size:
            self.call_next(self.app.resize_textarea, self)


class ModelProvider(textual.command.Provider):
    """A provider for themes."""

    app: "YellApp"  # type: ignore[reportIncompatibleMethodOverride]

    @property
    def commands(self) -> list[tuple[str, Callable[[], None]]]:
        models = models_list()

        def set_model(name: str) -> None:
            self.app.session.set_model(llm.get_async_model(name))

            self.app.title = (
                f"Conversation with {self.app.session.conversation.model.model_id}"
            )

        # todo: filter discovery with no aliases
        return [(model, partial(set_model, model)) for model in models]

    async def discover(self) -> textual.command.Hits:
        for command in self.commands:
            yield textual.command.DiscoveryHit(*command)

    async def search(self, query: str) -> textual.command.Hits:
        matcher = self.matcher(query)

        for name, callback in self.commands:
            if (match := matcher.match(name)) > 0:
                yield textual.command.Hit(
                    match,
                    matcher.highlight(name),
                    callback,
                )


class YellApp(textual.app.App):
    CSS_PATH = "yell.tcss"
    BINDINGS = [
        Binding("ctrl+c", "ctrl_c", "Exit, maybe", priority=True, show=False),
        Binding("ctrl+d", "exit", "Exit", priority=True),
        Binding("ctrl+m", "pick_model", "Select model"),
    ]

    def __init__(self, session: ChatSession):
        super().__init__(watch_css=True)
        self.session = session

    def compose(self) -> ComposeResult:
        yield textual.widgets.Header()
        self.container = VerticalScroll(can_focus=False, can_focus_children=True)
        with self.container:
            for resp in self.session.conversation.responses:
                if resp.prompt._prompt:
                    umd = textual.widgets.Static(
                        resp.prompt._prompt, classes="yell_user_prompt"
                    )
                    umd.border_title = "User"
                    yield umd
                    # prompt_session.history.append_string(resp.prompt._prompt)
                if isinstance(resp, llm.AsyncResponse):
                    md = textual.widgets.Markdown(
                        resp.text_or_raise(), classes="yell_response"
                    )
                    md.border_title = resp.resolved_model or resp.model.model_id
                    yield md

            self.ta = YellInput(self, "", id="yell-input")
            yield self.ta
        yield textual.widgets.Footer()
        self.c_c_time = None

    def on_mount(self):
        self.title = f"Conversation with {self.session.conversation.model.model_id}"
        self.container.anchor()
        self.log(str(self.ansi_theme))

    def on_terminal_color_theme(self, message: TerminalColorTheme):
        match message.theme:
            case "light":
                self.theme = "catppuccin-latte"  # "textual-light"
            case "dark":
                self.theme = "catppuccin-mocha"  # "textual-dark"

    async def accept_text(self, text):
        if not (text.strip() or self.session.attachments or self.session.fragments):
            return

        if text[0] == "/":
            if text[1] == "/":
                text = text[1:]
            else:
                words = text.strip().split()
                match words[0]:
                    case "/quit" | "/exit":
                        self.exit()
                    case "/model":
                        if len(words) > 1:
                            self.session.set_model(llm.get_async_model(words[1]))
                        else:
                            self.session.set_model(
                                cast(AsyncModel, self.session.conversation.model)
                            )
                    case "/fragment":
                        self.session.attach_fragments(words[1:], allow_attachments=True)
                    case "/options":
                        self.action_show_options()
                    case _:
                        self.notify(f"Unknown command {words[0]}", severity="error")
                return

        umd = textual.widgets.Static(text, classes="yell_user_prompt")
        umd.border_title = "User"
        new_md = textual.widgets.Markdown("...", classes="yell_response")
        new_md.loading = True
        self.ta.loading = True
        new_md.border_title = self.session.conversation.model.model_id
        self.container.anchor()
        await self.container.mount_all([umd, new_md], before=self.ta)
        resp = self.session.run(text)
        self.run_worker(self.run_llm_response(new_md, resp), exclusive=True)

    async def action_accept(self):
        new_text = self.ta.text
        self.ta.text = ""
        await self.accept_text(new_text)

    async def run_llm_response(self, new_md, resp: AsyncChainResponse):
        full_resp = ""
        first = True
        try:
            async for r in resp:
                full_resp += r
                if first:
                    new_md.loading = False
                    await new_md.update(r)
                    first = False
                else:
                    await new_md.append(r)
            await new_md.update(full_resp)
        finally:
            self.ta.loading = False
            # this is needed when the terminal is not focussed while the completion finished
            # todo: manipulate the focus stack instead
            self.ta.focus()
        for r in resp._responses:
            (await r.to_sync_response()).log_to_db(self.session.db)

    async def action_ctrl_c(self) -> None:
        # todo: bubble back so copying works
        now = datetime.now()
        if self.c_c_time and (now - self.c_c_time).seconds < 0.5:
            self.exit()
        else:
            self.c_c_time = now
            try:
                # actually copy after 0.5s
                self.screen.action_copy_text()
            except SkipAction:
                self.notify("Press ctrl-c twice to exit", timeout=0.5)

    async def action_exit(self) -> None:
        self.exit()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Model", "Pick a model", self.action_pick_model)
        yield SystemCommand("Options", "Show options", self.action_show_options)

    def action_pick_model(self) -> None:
        self.push_screen(
            textual.command.CommandPalette(
                providers=[ModelProvider], placeholder="select a model"
            )
        )

    def on_text_area_changed(self, message: textual.widgets.TextArea.Changed) -> None:
        self.resize_textarea(self.ta)

    def resize_textarea(self, textarea: YellInput):
        cur_h = min(8, textarea.wrapped_document.height + 2)
        self.container.anchor()
        textarea.styles.height = cur_h

    def action_show_options(self) -> None:
        self.container.anchor()

        t = rich.table.Table()
        t.add_column("Name")
        t.add_column("Value")
        t.add_column("Description")
        for k, v in self.session.conversation.model.Options.model_fields.items():
            t.add_row(k, str(self.session.options.get(k)), v.description)
        self.container.mount(textual.widgets.Static(t), before=self.ta)


if __name__ == "__main__":
    app()
