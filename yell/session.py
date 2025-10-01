
from typing import Iterable

import llm
import llm.cli
import llm.migrations
import rich
import rich.table
import sqlite_utils
from llm.models import AsyncModel


class ChatSession:
    def __init__(self, model: AsyncModel, conversation: llm.AsyncConversation | None, db: sqlite_utils.Database):
        self.console = rich.get_console()
        self.db = db
        self.fragments: list[llm.Fragment] = []
        self.attachments: list[llm.Attachment] = []
        self.system_fragments: list[llm.Fragment | llm.Attachment] = []
        self.system: str | None = None
        self.options: dict = {}

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

    def run(self, text) -> llm.models.AsyncChainResponse:
        response = self.conversation.chain(
            text,
            fragments=[str(f) for f in self.fragments],
            system_fragments=[str(f) for f in self.system_fragments],
            attachments=self.attachments,
            system=self.system,
            options=self.options,
        )

        self.system = None
        self.system_fragments = []
        self.fragments = []
        self.attachments = []
        return response

    def attach_fragments(self, fragments: Iterable[str], *, allow_attachments: bool = False, system: bool = False) -> None:
        for frag in llm.cli.resolve_fragments(self.db, fragments, allow_attachments=allow_attachments):
            self.attach(frag, system=system)
    def set_options(self, options: dict):
        self.options = options

