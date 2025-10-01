from datetime import datetime
from functools import partial
from typing import Annotated, Any, Callable, Iterable, cast
from click.types import ParamType
import llm
import llm.cli
import llm.migrations
from llm.models import AsyncChainResponse, AsyncModel
from textual.actions import SkipAction
from textual.app import ComposeResult, SystemCommand
import textual.app
from textual.binding import Binding
import textual.events
from textual.screen import Screen
import typer
import sqlite_utils
import rich
import rich.table
import textual
import textual.widgets
import textual.command
import sys
from .input import run as run_prompt
from .input import models_list
from .output import Format

__all__ = ["app"]

from textual._ansi_sequences import ANSI_SEQUENCES_KEYS
class ShEnter:
    value = "shift+enter"
ANSI_SEQUENCES_KEYS["\x1b\r"] = (ShEnter(),)

class ChatSession:
    def __init__(self, model: llm.Model, conversation: llm.AsyncConversation | None):
        self.console = rich.get_console()
        self.fragments: list[llm.Fragment] = []
        self.attachments: list[llm.Attachment] = []
        self.system_fragments: list[llm.Fragment | llm.Attachment] = []
        self.system: str | None = None

        if conversation is None:
            self.conversation = llm.AsyncConversation(model)
            # self.info(f"Starting conversation [green not bold]{self.conversation.id}")
        else:
            self.conversation = conversation
            self.info(
                f"Resuming conversation [green not bold]{conversation.id}[/] ({len(conversation.responses)} responses)"
            )

    def attach(self, frag: llm.Fragment | llm.Attachment, *, system: bool = False):
        if isinstance(frag, llm.Fragment):
            self.info(f"Added {'system ' if system else ''}fragment [b]{frag.source}")
            if system:
                self.system_fragments.append(frag)
            else:
                self.fragments.append(frag)
        elif isinstance(frag, llm.Attachment):
            self.info(
                f"Added {'system ' if system else ''}attachment [b]{frag.type} {frag.path} {frag.url}"
            )
            if system:
                self.system_fragments.append(frag)
            else:
                self.attachments.append(frag)

    def info(self, *args):
        # self.console.print(*args, style="cyan italic")
        pass

    def set_model(self, model: AsyncModel):
        self.conversation.model = model
        self.info(f"Chatting with [b black]{model.model_id}")

    def run(self, text, options) -> llm.models.AsyncChainResponse:
        response = self.conversation.chain(
            text,
            fragments=[str(f) for f in self.fragments],
            system_fragments=[str(f) for f in self.system_fragments],
            attachments=self.attachments,
            system=self.system,
            options=options,
        )

        self.system = None
        self.system_fragments = []
        self.fragments = []
        self.attachments = []
        return response


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
    format: Annotated[Format, None] = Format.markdown,
):
    # prompt_session = PromptSession()
    # init code from cli.py
    log_path = llm.cli.logs_db_path()
    (log_path.parent).mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(log_path)
    llm.migrations.migrate(db)
    conversation = None
    if conversation_id or _continue:
        conversation = cast(
            llm.AsyncConversation | None, llm.cli.load_conversation(conversation_id, async_=True)
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
    session = ChatSession(model, conversation)
    validated_options = llm.cli.get_model_options(model.model_id)
    if options:
        merged_options = {**validated_options, **dict(options)}
        validated_options = model.Options.model_validate_strings(
            merged_options
        ).model_dump(exclude_none=True, exclude_unset=True)
    kwargs = {}
    if validated_options:
        kwargs["options"] = validated_options
    session.set_model(model)

    if system:
        session.system = system
    for frag in llm.cli.resolve_fragments(db, system_fragments or []):
        session.attach(frag, system=True)
    for frag in llm.cli.resolve_fragments(db, fragments_arg, allow_attachments=True):
        session.attach(frag)

    app = YellApp(session, validated_options, db)
    app.run()
    if session.conversation and session.conversation.responses:
        rich.print(f"Conversation saved as [green not bold]{session.conversation.id}")
    sys.exit(0)

    # renderer = format.to_renderer()

class YellInput(textual.widgets.TextArea):
    BINDINGS = [
        Binding("enter", "accept", "Accept", priority=True),
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

    @property
    def commands(self) -> list[tuple[str, Callable[[], None]]]:
        models = models_list()

        def set_model(name: str) -> None:
            cast(YellApp, self.app).session.set_model(llm.get_async_model(name))
            # todo: update header

        # todo: filter discovery with no aliases
        return [
            (model, partial(set_model, model))
            for model in models
        ]

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
        Binding("ctrl+c", "ctrl_c", "Exit, maybe", priority=True),
        Binding("ctrl+d", "exit", "Exit", priority=True),
    ]
    def __init__(self, session: ChatSession, options: dict, db: sqlite_utils.Database):
        super().__init__(watch_css=True)
        self.session = session
        self.options = options
        self.db = db
    def compose(self) -> ComposeResult:
        yield textual.widgets.Header()
        for resp in self.session.conversation.responses:
            if resp.prompt._prompt:
                umd = textual.widgets.Markdown(resp.prompt._prompt, classes="yell_user_prompt")
                umd.border_title = "User"
                yield umd
                # prompt_session.history.append_string(resp.prompt._prompt)
            if isinstance(resp, llm.AsyncResponse):
                md = textual.widgets.Markdown(resp.text_or_raise(), classes="yell_response")
                md.border_title = resp.resolved_model or resp.model.model_id
                yield md

        self.ta = YellInput(self, "", id="yell-input")
        yield self.ta
        # yield textual.widgets.Footer()
        self.c_c_time = None

    def on_mount(self):
        self.theme = "textual-light"
        self.title = f"Conversation with {self.session.conversation.model.model_id}"
        self.screen.anchor()

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
                        for frag in llm.cli.resolve_fragments(
                            self.db, words[1:], allow_attachments=True
                        ):
                            self.session.attach(frag)
                    case "/options":
                        self.action_show_options()
                    case _:
                        self.notify(f"Unknown command {words[0]}", severity="error")
                return

        umd = textual.widgets.Markdown(text, classes="yell_user_prompt")
        umd.border_title = "User"
        new_md = textual.widgets.Markdown("...", classes="yell_response")
        new_md.loading = True
        self.ta.loading = True
        new_md.border_title = self.session.conversation.model.model_id
        self.screen.anchor()
        await self.mount_all([umd, new_md], before=self.ta)
        resp = self.session.run(text, self.options)
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
        for r in resp._responses:
            (await r.to_sync_response()).log_to_db(self.db)

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
        self.push_screen(textual.command.CommandPalette(providers=[ModelProvider], placeholder="select a model"))

    def on_text_area_changed(self, message: textual.widgets.TextArea.Changed) -> None:
        self.resize_textarea(self.ta)

    def resize_textarea(self, textarea: YellInput):
        cur_h = min(8, textarea.wrapped_document.height + 2)
        self.screen.anchor()
        textarea.styles.height = cur_h

    def action_show_options(self) -> None:
        self.screen.anchor()

        t = rich.table.Table()
        t.add_column("Name")
        t.add_column("Value")
        t.add_column("Description")
        for k, v in self.session.conversation.model.Options.model_fields.items():
            t.add_row(k, str(self.options.get(k)), v.description)
        self.mount(textual.widgets.Static(t), before=self.ta)

if __name__ == "__main__":
    app()
