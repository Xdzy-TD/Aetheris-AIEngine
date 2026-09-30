"""
Textual-based Operator Console for Aetheris.

A terminal UI providing:
- Query input panel (question + real image path(s)/modality for single,
  cross-modal, or bi-temporal analysis)
- Full-width analysis report panel (rendered markdown of LLM synthesis)
- Live execution trace viewer (streams the hash-chained audit log in real
  time — including entries written by other processes, e.g. `serve`)
- Ollama-aware status footer

Launch with:
    python run.py tui
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def _check_textual() -> bool:
    """Check if Textual is installed."""
    try:
        import textual  # noqa: F401
        return True
    except ImportError:
        return False


if _check_textual():
    from textual.app import App, ComposeResult
    from textual.containers import Container, Horizontal
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        DirectoryTree,
        Footer,
        Header,
        Input,
        Label,
        RichLog,
        Select,
        Static,
    )

    class _FilePicker(ModalScreen[str | None]):
        """Minimal browse-for-image modal — built on Textual's DirectoryTree.

        Enter/click a file to pick it, Escape to cancel.
        """

        CSS = """
        _FilePicker { align: center middle; }
        #picker-box { width: 80%; height: 80%; border: heavy $accent; background: $surface; padding: 1 2; }
        """
        BINDINGS = [("escape", "dismiss(None)", "Cancel")]

        def compose(self) -> ComposeResult:
            with Container(id="picker-box"):
                yield Label("📁 Pick an image (Esc to cancel)", classes="panel-title")
                yield DirectoryTree(str(_Path.home()))

        def on_directory_tree_file_selected(
            self, event: DirectoryTree.FileSelected
        ) -> None:
            self.dismiss(str(event.path))

    # event_type -> Rich markup colour, for the live trace feed
    _EVENT_COLOR = {
        "plan_created": "dodger_blue1",
        "tool_called": "gold1",
        "tool_returned": "spring_green1",
        "answer_emitted": "orchid1",
    }
    _MODALITY_OPTIONS = [(m, m) for m in ("optical", "sar", "multispectral", "unknown")]

    class AetherisConsole(App):
        """Aetheris Operator Console — Textual TUI."""

        CSS = """
        Screen {
            layout: grid;
            grid-size: 3 1;
            grid-columns: 1fr 2fr 1fr;
            grid-gutter: 1;
            background: $surface;
        }

        /* --- Left column: query input --- */
        #query-panel {
            border: heavy $accent;
            padding: 1 2;
        }

        #query-panel .panel-title {
            text-style: bold;
            color: $text;
            margin-bottom: 1;
        }

        #query-input {
            margin-bottom: 1;
        }

        .image-row {
            height: 3;
            margin-bottom: 1;
        }

        .image-row Input {
            width: 3fr;
        }

        .image-row Select {
            width: 1fr;
        }

        .image-row Button {
            width: 5;
            min-width: 5;
        }

        #submit-btn {
            margin-top: 1;
            width: 100%;
        }

        #clear-btn {
            margin-top: 1;
            width: 100%;
        }

        /* --- Center column: analysis report --- */
        #report-panel {
            border: heavy $success;
            padding: 1 2;
        }

        #report-panel .panel-title {
            text-style: bold;
            color: $text;
            margin-bottom: 1;
        }

        #report-log {
            height: 1fr;
        }

        /* --- Right column: execution trace --- */
        #trace-panel {
            border: heavy $warning;
            padding: 1 2;
        }

        #trace-panel .panel-title {
            text-style: bold;
            color: $text;
            margin-bottom: 1;
        }

        #trace-log {
            height: 1fr;
        }

        /* --- Status bar --- */
        #status-bar {
            dock: bottom;
            height: 3;
            padding: 0 2;
            background: $panel;
            border-top: heavy $accent;
        }

        #status-bar Static {
            width: 1fr;
            content-align: center middle;
        }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("ctrl+l", "clear_log", "Clear Log"),
            ("escape", "cancel_query", "Cancel"),
            ("slash", "focus_query", "Focus Query"),
        ]

        TITLE = "AETHERIS"
        SUB_TITLE = "Agentic Remote-Sensing Intelligence System"

        # How often to poll the audit log for new entries (seconds). The log
        # is a plain file, so this also picks up activity from `run.py serve`
        # or any other process writing to the same log — a genuine live feed.
        _POLL_INTERVAL_S = 2.0

        def __init__(self) -> None:
            super().__init__()
            self._planner: Any = None
            self._tool_registry: Any = None
            self._memory: Any = None
            self._audit: Any = None
            self._ollama: Any = None
            self._seen_entry_ids: set[str] = set()
            self._processing = False

        def compose(self) -> ComposeResult:
            yield Header()

            # Left column — Query input
            with Container(id="query-panel"):
                yield Label("📡 QUERY INPUT", classes="panel-title")
                yield Input(
                    placeholder="Ask about the satellite image...",
                    id="query-input",
                )
                yield Horizontal(
                    Input(placeholder="Image path (optional)", id="image1-path"),
                    Select(_MODALITY_OPTIONS, value="optical", id="image1-modality",
                           allow_blank=False),
                    Button("📁", id="image1-browse"),
                    classes="image-row",
                )
                yield Horizontal(
                    Input(placeholder="Image 2 — change/cross-modal (optional)",
                          id="image2-path"),
                    Select(_MODALITY_OPTIONS, value="sar", id="image2-modality",
                           allow_blank=False),
                    Button("📁", id="image2-browse"),
                    classes="image-row",
                )
                yield Button("▶ Submit", id="submit-btn", variant="primary")
                yield Button("✕ Clear", id="clear-btn", variant="default")

            # Center column — Analysis report
            with Container(id="report-panel"):
                yield Label("📋 ANALYSIS REPORT", classes="panel-title")
                yield RichLog(id="report-log", highlight=True, markup=True, max_lines=500)

            # Right column — Execution trace
            with Container(id="trace-panel"):
                yield Label("🔗 EXECUTION TRACE", classes="panel-title")
                yield RichLog(id="trace-log", highlight=True, markup=True, max_lines=300)

            # Status bar
            with Horizontal(id="status-bar"):
                yield Static("", id="status-tools")
                yield Static("", id="status-ollama")
                yield Static("", id="status-memory")

            yield Footer()

        async def on_mount(self) -> None:
            """Initialise the controller, populate panels, and start the live feed."""
            self._init_controller()
            await self._update_status_bar()
            self.set_interval(self._POLL_INTERVAL_S, self._poll_audit_log)

        def _init_controller(self) -> None:
            """Initialise controller components."""
            try:
                from interfaces.api.deps import (
                    get_audit,
                    get_memory,
                    get_ollama_client,
                    get_planner,
                    get_registry,
                )

                self._tool_registry = get_registry()
                self._memory = get_memory()
                self._audit = get_audit()
                self._planner = get_planner()
                self._ollama = get_ollama_client()

                self._log_trace("[bold spring_green1]✓ Controller initialised[/]")
                tools = ", ".join(self._tool_registry.list_tool_names())
                self._log_trace(f"Tools: {tools}")
            except Exception as exc:
                self._log_trace(f"[bold red]⚠ Controller init failed: {exc}[/]")

        async def _update_status_bar(self) -> None:
            """Refresh the bottom status bar."""
            # Tools
            if self._tool_registry:
                names = self._tool_registry.list_tool_names()
                self.query_one("#status-tools", Static).update(
                    f"🔧 Tools: {len(names)} loaded"
                )

            # Ollama
            if self._ollama:
                try:
                    ok = await self._ollama.health_check()
                    model = self._ollama.model or "none"
                    icon = "✅" if ok else "❌"
                    self.query_one("#status-ollama", Static).update(
                        f"🤖 Ollama: {model} {icon}"
                    )
                except Exception:
                    self.query_one("#status-ollama", Static).update("🤖 Ollama: offline ❌")
            else:
                self.query_one("#status-ollama", Static).update("🤖 Ollama: disabled")

            # Memory
            if self._memory:
                stats = self._memory.stats()
                self.query_one("#status-memory", Static).update(
                    f"💾 Memory: {stats['total_entries']}/{stats['max_entries']}"
                )

        def _log_trace(self, message: str) -> None:
            """Append a (Rich-markup) message to the trace panel."""
            try:
                self.query_one("#trace-log", RichLog).write(message)
            except Exception:
                pass

        def _log_report(self, message: str) -> None:
            """Append content to the report panel."""
            try:
                self.query_one("#report-log", RichLog).write(message)
            except Exception:
                pass

        def _poll_audit_log(self) -> None:
            """Stream any new audit-log entries into the trace panel.

            Re-reads the log file each tick, so entries written by *any*
            process (this console, `run.py serve`, another console) show up
            here in near real time, not just the queries this session sent.
            """
            if not self._audit:
                return
            try:
                for entry in self._audit.get_last_n(50):
                    if entry.entry_id in self._seen_entry_ids:
                        continue
                    self._seen_entry_ids.add(entry.entry_id)
                    colour = _EVENT_COLOR.get(entry.event_type, "white")
                    tool = f" [{entry.tool_name}]" if entry.tool_name else ""
                    self._log_trace(
                        f"[{colour}]● {entry.event_type}{tool}[/] "
                        f"q={entry.query_id[:8]} @ {entry.timestamp:%H:%M:%S}"
                    )
            except Exception as exc:
                logger.warning("audit_poll_failed", error=str(exc))

        async def on_button_pressed(self, event: Button.Pressed) -> None:
            """Handle button clicks."""
            if event.button.id == "submit-btn":
                await self._submit_query()
            elif event.button.id == "clear-btn":
                self._clear_inputs()
            elif event.button.id in ("image1-browse", "image2-browse"):
                target = event.button.id.replace("-browse", "-path")
                self.push_screen(_FilePicker(), lambda p, t=target: self._set_image_path(t, p))

        def _set_image_path(self, input_id: str, path: str | None) -> None:
            if path:
                self.query_one(f"#{input_id}", Input).value = path

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            """Handle Enter key in the query input."""
            if event.input.id == "query-input":
                await self._submit_query()

        def _clear_inputs(self) -> None:
            """Reset all input fields and the report panel."""
            self.query_one("#query-input", Input).value = ""
            self.query_one("#image1-path", Input).value = ""
            self.query_one("#image2-path", Input).value = ""
            self.query_one("#report-log", RichLog).clear()

        def _collect_images(self) -> tuple[list, bytes | None] | None:
            """Validate and build ImageMetadata from the path/modality fields.

            Returns ``(images, image_bytes)`` or ``None`` (and logs the
            problem) if a non-empty path was given that doesn't exist —
            paths are never silently dropped, matching the API's own
            upload-then-validate behaviour.
            """
            from controller.schemas import ImageMetadata

            images = []
            image_bytes: bytes | None = None
            for path_id, modality_id in (
                ("image1-path", "image1-modality"),
                ("image2-path", "image2-modality"),
            ):
                raw_path = self.query_one(f"#{path_id}", Input).value.strip()
                if not raw_path:
                    continue
                path = _Path(raw_path).expanduser()
                if not path.is_file():
                    self._log_trace(f"[bold red]⚠ Image not found: {path}[/]")
                    return None
                modality = self.query_one(f"#{modality_id}", Select).value
                if image_bytes is None:
                    image_bytes = path.read_bytes()
                images.append(ImageMetadata(
                    filename=path.name, path=str(path), modality=modality,
                ))
            return images, image_bytes

        async def _submit_query(self) -> None:
            """Submit the current query to the controller with streaming feedback."""
            if self._processing:
                self._log_trace("[bold gold1]⚠ Already processing a query[/]")
                return

            input_widget = self.query_one("#query-input", Input)
            question = input_widget.value.strip()

            if not question:
                self._log_trace("[bold gold1]⚠ Empty query — type a question first[/]")
                return

            if not self._planner:
                self._log_trace("[bold red]⚠ Controller not initialised[/]")
                return

            collected = self._collect_images()
            if collected is None:
                return
            images, image_bytes = collected

            # Clear report and show processing state
            report_log = self.query_one("#report-log", RichLog)
            report_log.clear()
            self._processing = True
            submit_btn = self.query_one("#submit-btn", Button)
            submit_btn.label = "⏳ Processing..."
            submit_btn.disabled = True

            self._log_trace(f"📤 [bold]Query:[/] {question}")
            if images:
                self._log_trace(f"   {len(images)} image(s) attached")

            try:
                from controller.schemas import Query

                query = Query(text=question, images=images)
                result = await self._planner.process_query(query, image_bytes)

                # Stream plan info to trace
                plan_steps = [s.tool_name for s in result.plan.steps]
                self._log_trace(f"📋 [bold]Plan:[/] {' → '.join(plan_steps)}")
                self._log_trace(f"⏱  [dim]{result.execution_time_s}s[/dim]")

                # Stream specialist outputs to trace
                for out in result.outputs:
                    # SpecialistOutput has no success/timing fields; a step that
                    # could not run says so in its own result payload.
                    failed = out.result.get("analysis_unavailable", False)
                    status, colour = ("✗", "red") if failed else ("✓", "spring_green1")
                    self._log_trace(f"  [{colour}]{status} {out.tool_name}[/]")

                # Render the final answer in the report panel
                report_log.write("[bold dodger_blue1]━━━ Analysis Report ━━━[/]\n")
                report_log.write(result.final_answer)
                report_log.write("")  # blank line

                # Show confidence if available from specialist outputs
                for out in result.outputs:
                    conf = out.result.get("raw_confidence") or out.result.get("confidence")
                    if conf is not None:
                        report_log.write(
                            f"[dim]Confidence ({out.tool_name}): {conf}[/dim]"
                        )

            except Exception as exc:
                self._log_trace(f"[bold red]✗ Query failed: {exc}[/]")
                report_log.write(f"[bold red]Error: {exc}[/]")
            finally:
                self._processing = False
                submit_btn.label = "▶ Submit"
                submit_btn.disabled = False
                await self._update_status_bar()

        def action_clear_log(self) -> None:
            """Clear the trace log."""
            self.query_one("#trace-log", RichLog).clear()

        def action_cancel_query(self) -> None:
            """Cancel indicator (actual cancellation needs task management)."""
            if self._processing:
                self._log_trace("[bold gold1]⚠ Query is still running...[/]")

        def action_focus_query(self) -> None:
            """Focus the query input."""
            self.query_one("#query-input", Input).focus()


def run_console() -> None:
    """Launch the Aetheris operator console."""
    if not _check_textual():
        print(
            "Textual is not installed. Install with:\n"
            "  pip install textual\n"
            "Or use the API interface instead:\n"
            "  python run.py serve"
        )
        sys.exit(1)

    console = AetherisConsole()
    console.run()


if __name__ == "__main__":
    run_console()
