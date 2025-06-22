from enum import Enum
from typing import Iterable
import sys
import subprocess
import rich
import rich.live
import rich.markdown
import rich.markup
from .better_live import BetterLiveRender


def render_text(response: Iterable[str]):
    for chunk in response:
        print(chunk, end="")
        sys.stdout.flush()
    print("")


def render_sd(response: Iterable[str]):
    subp = subprocess.Popen(
        ["streamdown", "-b", "30,0,50", "--config", "features.Clipboard=false"],
        stdin=subprocess.PIPE,
    )
    if not subp or not subp.stdin:
        raise RuntimeError("failed to start streamdown")
    for chunk in response:
        subp.stdin.write(chunk.encode("utf8"))
    subp.communicate()


def render_rich(response: Iterable[str]):
    r = ""
    with rich.live.Live(None, auto_refresh=False, vertical_overflow="crop") as live:
        live._live_render.__class__ = BetterLiveRender
        for chunk in response:
            r += chunk
            live.update(rich.markdown.Markdown(r), refresh=True)
    print()


class Format(str, Enum):
    plain = "plain"
    markdown = "markdown"
    streamdown = "streamdown"

    def to_renderer(self):
        match self:
            case "plain":
                return render_text
            case "streamdown":
                return render_sd
            case "markdown" | _:
                return render_rich
