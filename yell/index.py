import functools
import sys
from datetime import datetime
from functools import partial
from typing import Annotated, Any, Callable, Iterable, Literal, cast

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
import textual.widgets.option_list
import typer
from click.types import ParamType
from llm.models import AsyncChainResponse, AsyncModel
from rich.console import Group
from rich.text import Text
from textual.actions import SkipAction
from textual.app import ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import (
    Horizontal,
    VerticalGroup,
    VerticalScroll,
)
from textual.css.query import NoMatches
from textual.messages import TerminalColorTheme
from textual.reactive import Initialize, Reactive, reactive
from textual.screen import Screen
from textual.types import OptionDoesNotExist
from textual.widget import Widget
from textual.worker import WorkerState

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
    session = load_or_create_session(_continue, conversation_id, model_id)
    model = session.conversation.model

    validated_options = llm.cli.get_model_options(model.model_id)
    if options:
        merged_options = {**validated_options, **dict(options)}
        validated_options = model.Options.model_validate_strings(
            merged_options
        ).model_dump(exclude_none=True, exclude_unset=True)
    session.set_options(validated_options)

    if system:
        session.system = system
    session.attach_fragments(system_fragments or [], system=True)
    session.attach_fragments(fragments_arg, allow_attachments=True)

    app = YellApp(session)
    app.run()
    if app.session.conversation and app.session.conversation.responses:
        rich.print(
            f"Conversation saved as [green not bold]{app.session.conversation.id}"
        )
    sys.exit(0)


@functools.cache
def get_db() -> sqlite_utils.Database:
    # init code from cli.py
    log_path = llm.cli.logs_db_path()
    (log_path.parent).mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(log_path)
    llm.migrations.migrate(db)
    return db


def load_or_create_session(
    _continue: bool, conversation_id: str | None, model_id: str | None
):
    db = get_db()
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
    session.set_model(model)
    return session


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

            self.app.mutate_reactive(YellApp.session)

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


class YellHistoryOption(textual.widgets.option_list.Option):
    def __init__(self, c: dict, *args, **kwargs):
        t = Group(Text(c["name"], no_wrap=True))
        id = c["id"]
        self.text = c["name"]
        super().__init__(t, id=id, *args, **kwargs)


class YellHistory(Widget):
    can_focus = False
    BINDINGS = [Binding("escape", "app.action_history(False)")]
    session: Reactive[ChatSession] = reactive(
        Initialize(lambda self: cast(YellHistory, self)._session)
    )

    def __init__(self, session: ChatSession, transient: bool, *args, **kwargs):
        self._session = session
        self.transient = transient
        super().__init__(*args, **kwargs)
        db = get_db()
        convs = list(db["conversations"].rows_where(order_by="id desc", limit=1000))
        seen_ids = {c["id"] for c in convs}
        if self._session.conversation.id not in seen_ids:
            self.options = [
                YellHistoryOption(
                    {
                        "id": self._session.conversation.id,
                        "name": self._session.conversation.name or "New Chat",
                    }
                )
            ]
        else:
            self.options = []
        self.options += [YellHistoryOption(c) for c in convs]

    def compose(self) -> ComposeResult:
        yield textual.widgets.OptionList(*self.options, compact=True)

    def watch_session(self, prev: ChatSession, cur: ChatSession):
        olist = self.query_one(textual.widgets.OptionList)
        try:
            opt_id = olist.get_option_index(cur.conversation.id)
        except OptionDoesNotExist:
            return
        olist.highlighted = opt_id

    def on_mount(self):
        def adjust_hover(val):
            if val:
                self.tooltip = self.options[val].text
            else:
                self.tooltip = None

        self.watch(
            self.query_one(textual.widgets.OptionList),
            "_mouse_hovering_over",
            adjust_hover,
        )


class YellPrompt(textual.widgets.Static):
    BORDER_TITLE = "User"


class YellResponse(textual.widgets.Markdown):
    def __init__(self, model: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = model


class YellChats(VerticalGroup):
    session: Reactive[ChatSession] = reactive(
        Initialize(lambda self: cast(YellChats, self)._session)
    )

    def __init__(self, session: ChatSession, *args, **kwargs):
        self._session = session
        super().__init__(*args, **kwargs)

    def compose(self) -> ComposeResult:
        for resp in self.session.conversation.responses:
            if resp.prompt._prompt:
                yield YellPrompt(resp.prompt._prompt, markup=False)
                # prompt_session.history.append_string(resp.prompt._prompt)
            if isinstance(resp, llm.AsyncResponse):
                yield YellResponse(
                    model=resp.resolved_model or resp.model.model_id,
                    markdown=resp.text_or_raise(),
                )

    async def watch_session(self, old: ChatSession, curr: ChatSession):
        if old.conversation.id != curr.conversation.id:
            await self.recompose()


class YellApp(textual.app.App):
    CSS_PATH = "yell.tcss"
    BINDINGS = [
        Binding("ctrl+c", "ctrl_c", "Exit, maybe", priority=True, show=False),
        Binding("ctrl+d", "exit", "Exit", priority=True, show=True),
        Binding("ctrl+m", "pick_model", "Model"),
        Binding("ctrl+o", "history", "History"),
        Binding(
            "alt+down",
            "navigate('down')",
            "Next conv",
            group=Binding.Group(description="Navigate Chats"),
        ),
        Binding(
            "alt+up",
            "navigate('up')",
            "Previous conv",
            group=Binding.Group(description="Navigate Chats"),
        ),
        Binding("escape", "escape"),
        Binding("ctrl+n", "new_chat", "New Chat"),
        Binding("pageup", "page_up", priority=True),
        Binding("pagedown", "page_down", priority=True),
    ]

    session: Reactive[ChatSession] = reactive(
        Initialize(lambda self: cast(YellApp, self)._session)
    )

    def action_page_up(self):
        self.query_one("#chatarea").scroll_page_up()

    def action_page_down(self):
        self.query_one("#chatarea").scroll_page_down()

    def __init__(self, session: ChatSession):
        self._session = session
        super().__init__()
        self.theme = "catppuccin-latte"
        self._alt_selection = False

    def compose(self) -> ComposeResult:
        yield textual.widgets.Header()
        self.hor = Horizontal()
        with self.hor:
            yield YellHistory(self.session, transient=True)
            self.ta = YellInput(self, "", id="yell-input")
            self.container = VerticalScroll(
                YellChats(self.session).data_bind(YellApp.session),
                self.ta,
                can_focus=False,
                can_focus_children=True,
                id="chatarea",
            )
            yield self.container
        yield textual.widgets.Footer()
        self.c_c_time = None

    def on_mount(self):
        self.query_one("#chatarea").anchor()

    def watch_session(self, old_session: ChatSession, new_session: ChatSession):
        self.title = f"Conversation with {new_session.conversation.model.model_id}"
        self.sub_title = f"id {new_session.conversation.id}"

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
                        self.mutate_reactive(YellApp.session)
                    case "/fragment":
                        self.session.attach_fragments(words[1:], allow_attachments=True)
                    case "/options":
                        self.action_show_options()
                    case _:
                        self.notify(f"Unknown command {words[0]}", severity="error")
                return

        umd = YellPrompt(text, markup=False)
        new_md = YellResponse(self.session.conversation.model.model_id, "...")
        new_md.loading = True
        text_area = self.query_one(YellInput)
        text_area.loading = True
        self.query_one("#chatarea").anchor()
        await self.query_one(YellChats).mount_all([umd, new_md])
        resp = self.session.run(text)
        self.worker = self.run_worker(
            self.run_llm_response(new_md, resp), exclusive=True
        )

    async def action_accept(self):
        text_area = self.query_one(YellInput)
        new_text = text_area.text
        text_area.text = ""
        await self.accept_text(new_text)

    async def run_llm_response(self, new_md, resp: AsyncChainResponse):
        full_resp = ""
        first = True
        session = self.session
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
            text_area = self.query_one(YellInput)
            text_area.loading = False
            # this is needed when the terminal is not focussed while the completion finished
            # todo: manipulate the focus stack instead
            text_area.focus()
        for r in resp._responses:
            (await r.to_sync_response()).log_to_db(session.db)

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

    async def action_history(self, show_hide: bool | None = None) -> None:
        try:
            if show_hide in (False, None):
                self.query_one(YellHistory).remove()
        except NoMatches:
            if show_hide in (True, None):
                await self.hor.mount(
                    YellHistory(self.session, transient=False), before=0
                )
                self.query_one(YellHistory).query_one(
                    textual.widgets.OptionList
                ).focus()

    def get_system_commands(self, screen: Screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Model", "Pick a model", self.action_pick_model)
        yield SystemCommand("Options", "Show options", self.action_show_options)
        yield SystemCommand("Toggle history", "Toggle history", self.action_history)

    def action_pick_model(self) -> None:
        self.push_screen(
            textual.command.CommandPalette(
                providers=[ModelProvider], placeholder="select a model"
            )
        )

    async def on_text_area_changed(
        self, message: textual.widgets.TextArea.Changed
    ) -> None:
        self.resize_textarea(self.query_one(YellInput))
        try:
            if self.query_one(YellHistory).transient:
                await self.query_one(YellHistory).remove()
        except NoMatches:
            pass

    def resize_textarea(self, textarea: YellInput):
        cur_h = min(8, textarea.wrapped_document.height + 2)
        self.query_one("#chatarea").anchor()
        textarea.styles.height = cur_h

    def action_show_options(self) -> None:
        self.query_one("#chatarea").anchor()

        t = rich.table.Table()
        t.add_column("Name")
        t.add_column("Value")
        t.add_column("Description")
        for k, v in self.session.conversation.model.Options.model_fields.items():
            t.add_row(k, str(self.session.options.get(k)), v.description)
        self.query_one(YellChats).mount(textual.widgets.Static(t))

    async def on_option_list_option_selected(
        self, message: textual.widgets.OptionList.OptionSelected
    ):
        id_ = message.option.id
        if not id_:
            return
        if not self._alt_selection and self.query_one(YellHistory).transient:
            await self.query_one(YellHistory).remove()
        self._alt_selection = False
        if self.session.conversation.id == id_:
            return
        self.query_one("#chatarea").anchor()
        self.session = load_or_create_session(
            _continue=False, conversation_id=id_, model_id=None
        )
        self.ta.focus()

    async def action_new_chat(self):
        self.session = load_or_create_session(
            _continue=False, conversation_id=None, model_id=None
        )

    async def action_navigate(self, direction: Literal["down", "up"]):
        try:
            optionlist = self.query_one("YellHistory > OptionList")
        except NoMatches:
            return
        self._alt_selection = True
        with self.app.batch_update():
            await optionlist.run_action("cursor_" + direction)
            await optionlist.run_action("select")

    async def action_escape(self):
        if self.worker and self.worker.state == WorkerState.RUNNING:
            self.worker.cancel()
        try:
            if x := self.query_one(YellHistory):
                if x.has_focus_within:
                    await self.action_history(False)
        except NoMatches:
            return


if __name__ == "__main__":
    app()
