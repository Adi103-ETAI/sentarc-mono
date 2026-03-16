"""
Markdown renderer component.
"""
from __future__ import annotations

from textual.widgets import Markdown


class MarkdownComponent(Markdown):
    """
    Rich Markdown renderer built on textual's :class:`~textual.widgets.Markdown`.

    Parameters
    ----------
    markdown:
        Initial Markdown source to display.
    """

    DEFAULT_CSS = """
    MarkdownComponent {
        margin: 0 0 1 0;
    }
    """

    def __init__(
        self,
        markdown: str = "",
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(markdown=markdown or None, id=id, classes=classes)
        self._source = markdown

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """The current Markdown source."""
        return self._source

    async def set_markdown(self, markdown: str) -> None:
        """Update the displayed Markdown asynchronously."""
        self._source = markdown
        await self.update(markdown)

    def set_markdown_sync(self, markdown: str) -> None:
        """Schedule a Markdown update from synchronous code."""
        self._source = markdown
        self.call_after_refresh(lambda: self.update(markdown))

    # ------------------------------------------------------------------
    # Component protocol shim
    # ------------------------------------------------------------------

    def render(self, width: int = 0) -> list[str]:  # type: ignore[override]
        return self._source.splitlines() or [""]

    def handle_input(self, data: str) -> None:
        pass

    def invalidate(self) -> None:
        self.refresh()
