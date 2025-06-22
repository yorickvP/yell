from prompt_toolkit import filters
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import (
    ConditionalCompleter,
    FuzzyWordCompleter,
    NestedCompleter,
    PathCompleter,
)
from prompt_toolkit.cursor_shapes import ModalCursorShapeConfig
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.key_binding.bindings import emacs, named_commands
import llm

bindings = KeyBindings()

# swap enter and shift-enter
insert_mode = filters.vi_insert_mode | filters.emacs_insert_mode


@bindings.add("escape", "enter", filter=insert_mode & filters.is_multiline)
def _(event: KeyPressEvent) -> None:
    """
    Newline (in case of multiline input.
    """
    event.current_buffer.newline(copy_margin=not filters.in_paste_mode())


bindings.add("enter", filter=insert_mode & emacs.is_returnable)(
    named_commands.get_by_name("accept-line")
)


@filters.Condition
def start_slash() -> bool:
    return get_app().current_buffer.document.text.startswith("/")


def models_list():
    for model_with_aliases in llm.get_models_with_aliases():
        yield model_with_aliases.model.model_id
        yield from model_with_aliases.aliases


command_completer = NestedCompleter.from_nested_dict(
    {
        "/model": FuzzyWordCompleter(models_list, WORD=True),
        "/quit": None,
        "/last": None,
        "/options": None,
        "/fragment": PathCompleter(),
        "/format": {"plain", "markdown", "streamdown"},
    }
)
conditional_command_completer = ConditionalCompleter(command_completer, start_slash)


def run(prompt_session):
    return prompt_session.prompt(
        "> ",
        multiline=True,
        key_bindings=bindings,
        completer=conditional_command_completer,
        cursor=ModalCursorShapeConfig(),
    )
