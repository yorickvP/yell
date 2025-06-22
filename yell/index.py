from typing import Annotated, Any, cast
from click.types import ParamType
from prompt_toolkit import PromptSession
import llm
import llm.cli
import llm.migrations
import typer
import sqlite_utils
import rich
from .input import run as run_prompt
from .input import models_list
from .output import Format

__all__ = ["app"]


class ChatSession:
    def __init__(self, model: llm.Model, conversation: llm.Conversation | None):
        self.console = rich.get_console()
        self.fragments: list[llm.Fragment] = []
        self.attachments: list[llm.Attachment] = []
        self.system_fragments: list[llm.Fragment | llm.Attachment] = []
        self.system: str | None = None

        if conversation is None:
            self.conversation = llm.Conversation(model)
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
        self.console.print(*args, style="cyan italic")

    def set_model(self, model: llm.Model):
        self.conversation.model = model
        self.info(f"Chatting with [b black]{model.model_id}")

    def run(self, text, options) -> llm.models.ChainResponse:
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
    prompt_session = PromptSession()
    # init code from cli.py
    log_path = llm.cli.logs_db_path()
    (log_path.parent).mkdir(parents=True, exist_ok=True)
    db = sqlite_utils.Database(log_path)
    llm.migrations.migrate(db)
    conversation = None
    if conversation_id or _continue:
        conversation = cast(
            llm.Conversation | None, llm.cli.load_conversation(conversation_id)
        )
        if conversation:
            for resp in conversation.responses:
                if resp.prompt._prompt:
                    prompt_session.history.append_string(resp.prompt._prompt)
    if model_id is None:
        if conversation:
            model_id = conversation.model.model_id
        else:
            model_id = llm.get_default_model()
    model = llm.get_model(model_id)
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

    renderer = format.to_renderer()

    while True:
        try:
            text = run_prompt(prompt_session)
            if not (text.strip() or session.attachments or session.fragments):
                continue
            if text[0] == "/":
                if text[1] == "/":
                    text = text[1:]
                else:
                    words = text.strip().split()
                    match words[0]:
                        case "/quit" | "/exit":
                            break
                        case "/model":
                            if len(words) > 1:
                                session.set_model(llm.get_model(words[1]))
                            else:
                                session.set_model(
                                    cast(llm.Model, session.conversation.model)
                                )
                            continue
                        case "/fragment":
                            for frag in llm.cli.resolve_fragments(
                                db, words[1:], allow_attachments=True
                            ):
                                session.attach(frag)
                            continue
                        case "/options":
                            session.info(validated_options)
                            session.info(
                                {
                                    k: v.description
                                    for k, v in session.conversation.model.Options.model_fields.items()
                                }
                            )
                            continue
                        case "/format":
                            renderer = Format(words[1]).to_renderer()
                            session.info(f"Set output format to [green]{words[1]}")
                            continue
                        case "/last":
                            if session.conversation.responses:
                                last_resp = session.conversation.responses[-1]
                                session.console.print(
                                    ">",
                                    last_resp.prompt._prompt,
                                    style="yellow",
                                    markup=False,
                                )
                                renderer([str(last_resp)])
                            else:
                                session.info("No previous output")
                            continue
                        case _:
                            print("Unknown command", text)
                            continue
            response = session.run(text, validated_options)
            renderer(response)
            response.log_to_db(db)
            # usage = response._responses[-1].token_usage()  # todo: sum
        except (KeyboardInterrupt, EOFError):
            break
    if session.conversation and session.conversation.responses:
        session.info(f"Conversation saved as [green not bold]{session.conversation.id}")


if __name__ == "__main__":
    app()
