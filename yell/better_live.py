import rich.live_render
from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment
from rich.text import Text
from rich._loop import loop_last


class BetterLiveRender(rich.live_render.LiveRender):
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        renderable = self.renderable
        style = console.get_style(self.style)
        lines = console.render_lines(renderable, options, style=style, pad=False)
        shape = Segment.get_shape(lines)

        _, height = shape
        if height > options.size.height:
            if self.vertical_overflow == "crop":
                lines = lines[(-options.size.height) :]
                shape = Segment.get_shape(lines)
            elif self.vertical_overflow == "ellipsis":
                lines = lines[: (options.size.height - 1)]
                overflow_text = Text(
                    "...",
                    overflow="crop",
                    justify="center",
                    end="",
                    style="live.ellipsis",
                )
                lines.append(list(console.render(overflow_text)))
                shape = Segment.get_shape(lines)
        self._shape = shape

        new_line = Segment.line()
        for last, line in loop_last(lines):
            yield from line
            if not last:
                yield new_line
