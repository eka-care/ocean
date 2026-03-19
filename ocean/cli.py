"""Ocean CLI — unified AI from the command line."""

from __future__ import annotations

import sys

import click
from dotenv import load_dotenv

from ocean.exceptions import OceanError

load_dotenv()


class OceanGroup(click.Group):
    """Custom group that treats unknown subcommands as model names."""

    def parse_args(self, ctx, args):
        # If the first arg looks like a known subcommand, let Click handle it.
        # Otherwise, inject "ask" so `ocean gpt-4o "hello"` → `ocean ask gpt-4o "hello"`
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            args = ["ask"] + args
        return super().parse_args(ctx, args)


@click.group(cls=OceanGroup, context_settings={"help_option_names": ["-h", "--help"]})
def main():
    """Ocean — unified AI CLI.

    \b
    Quick start:
      ocean gpt-5.4 "hello world"
      ocean gemini-2.5-flash "explain quantum computing" --max-tokens 200
      ocean gpt-image-1 "a sunset over mountains" -o sunset.png
      ocean chat gpt-5.4
      ocean chat gemini-2.5-flash --system "You are a helpful tutor"
      ocean list
      ocean list --provider openai
    """


def _make_thinking(
    thinking: bool | None,
    thinking_effort: str | None,
    thinking_budget: int | None,
    show_thinking: bool,
):
    """Build a ThinkingConfig from CLI flags, or None if nothing was set."""
    from ocean.models import ThinkingConfig

    if thinking is None and thinking_effort is None and thinking_budget is None and not show_thinking:
        return None
    return ThinkingConfig(
        enabled=thinking,
        effort=thinking_effort,
        budget=thinking_budget,
        show=show_thinking,
    )


_THINKING_OPTIONS = [
    click.option("--thinking/--no-thinking", default=None, help="Enable or disable model thinking/reasoning."),
    click.option("--thinking-effort", type=click.Choice(["low", "medium", "high"]), default=None,
                 help="Thinking effort level (OpenAI o-series) or maps to budget (Gemini/Claude)."),
    click.option("--thinking-budget", type=int, default=None,
                 help="Thinking token budget (Gemini 2.5+/Claude). Overrides --thinking-effort."),
    click.option("--show-thinking", is_flag=True, default=False,
                 help="Display the model's reasoning/thinking process."),
]


def add_thinking_options(f):
    for option in reversed(_THINKING_OPTIONS):
        f = option(f)
    return f


@main.command()
@click.argument("model")
@click.argument("prompt", default="")
@click.option("--max-tokens", type=int, default=None, help="Maximum tokens to generate.")
@click.option("--temperature", type=float, default=None, help="Sampling temperature.")
@click.option("--image", type=click.Path(exists=True), default=None, help="Image file for vision models.")
@click.option("--audio", type=click.Path(exists=True), default=None, help="Audio file for transcription.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path (images, audio).")
@click.option("--voice", default=None, help="Voice name for TTS.")
@click.option("--no-stream", is_flag=True, default=False, help="Disable streaming output.")
@add_thinking_options
def ask(
    model: str,
    prompt: str,
    max_tokens: int | None,
    temperature: float | None,
    image: str | None,
    audio: str | None,
    output: str | None,
    voice: str | None,
    no_stream: bool,
    thinking: bool | None,
    thinking_effort: str | None,
    thinking_budget: int | None,
    show_thinking: bool,
) -> None:
    """Send a prompt to an AI model.

    \b
    Usage: ocean MODEL [PROMPT] [OPTIONS]
    """
    from ocean import invoke as ocean_invoke

    stream = not no_stream
    thinking_cfg = _make_thinking(thinking, thinking_effort, thinking_budget, show_thinking)

    try:
        result = ocean_invoke(
            model,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            image=image,
            audio=audio,
            output=output,
            voice=voice,
            thinking=thinking_cfg,
        )

        if stream:
            from ocean.providers.base import THINK_PREFIX
            for chunk in result:
                if chunk.startswith(THINK_PREFIX):
                    sys.stderr.write(chunk[len(THINK_PREFIX):])
                    sys.stderr.flush()
                else:
                    click.echo(chunk, nl=False)
            click.echo()
        else:
            if show_thinking and result.thinking_text:
                click.echo(f"[thinking]\n{result.thinking_text}\n[/thinking]\n", err=True)
            if result.text:
                click.echo(result.text)

    except OceanError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command("chat")
@click.argument("model")
@click.option("--max-tokens", type=int, default=None, help="Maximum tokens per response.")
@click.option("--temperature", type=float, default=None, help="Sampling temperature.")
@click.option("--system", "-s", default=None, help="System prompt.")
@add_thinking_options
def chat(
    model: str,
    max_tokens: int | None,
    temperature: float | None,
    system: str | None,
    thinking: bool | None,
    thinking_effort: str | None,
    thinking_budget: int | None,
    show_thinking: bool,
) -> None:
    """Start an interactive conversation with a model.

    \b
    Usage: ocean chat MODEL [OPTIONS]

    \b
    Commands inside chat:
      /clear    Clear conversation history
      /model    Show current model
      /system   Set / show system prompt
      /history  Show conversation history
      /exit     Exit  (or Ctrl+D)
    """
    from rich.console import Console
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.text import Text
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.styles import Style

    from ocean.providers.base import THINK_PREFIX
    from ocean.registry import resolve

    try:
        provider = resolve(model)
    except OceanError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    console = Console()
    messages: list[dict] = []
    current_system = system
    current_thinking = _make_thinking(thinking, thinking_effort, thinking_budget, show_thinking)

    # ── header ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold cyan]{model}[/bold cyan]\n"
        "[dim]/clear  /system  /history  /exit  ·  Ctrl+D to quit[/dim]",
        title="[bold]Ocean Chat[/bold]",
        border_style="cyan",
        padding=(0, 2),
    ))
    console.print()

    # ── prompt_toolkit session ───────────────────────────────────────────────
    pt_style = Style.from_dict({"prompt": "bold ansicyan"})
    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        style=pt_style,
        mouse_support=False,
    )

    def _print_info(msg: str) -> None:
        console.print(f"[dim]  {msg}[/dim]\n")

    while True:
        try:
            user_input = session.prompt(HTML("<prompt>  you  </prompt> ▸ ")).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye![/dim]")
            break

        if not user_input:
            continue

        # ── slash commands ───────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd_name = parts[0].lower()

            if cmd_name in ("/exit", "/quit", "/q"):
                console.print("[dim]Bye![/dim]")
                break

            elif cmd_name == "/clear":
                messages.clear()
                console.clear()
                console.print(Panel(
                    f"[bold cyan]{model}[/bold cyan]\n"
                    "[dim]/clear  /system  /history  /exit  ·  Ctrl+D to quit[/dim]",
                    title="[bold]Ocean Chat[/bold]",
                    border_style="cyan",
                    padding=(0, 2),
                ))
                console.print()
                _print_info("conversation cleared")

            elif cmd_name == "/model":
                _print_info(f"model: [cyan]{model}[/cyan]")

            elif cmd_name == "/system":
                if len(parts) > 1:
                    current_system = parts[1]
                    _print_info("system prompt updated")
                else:
                    if current_system:
                        console.print(Panel(
                            current_system,
                            title="system prompt",
                            border_style="dim",
                        ))
                        console.print()
                    else:
                        _print_info("no system prompt set")

            elif cmd_name == "/history":
                if not messages:
                    _print_info("no history yet")
                else:
                    console.print()
                    for msg in messages:
                        role_color = "cyan" if msg["role"] == "assistant" else "green"
                        preview = msg["content"][:200]
                        if len(msg["content"]) > 200:
                            preview += "…"
                        console.print(f"  [{role_color}]{msg['role']}[/{role_color}]  {preview}")
                    console.print()

            elif cmd_name == "/help":
                console.print(Panel(
                    "  [bold]/clear[/bold]     Clear conversation history\n"
                    "  [bold]/model[/bold]     Show current model\n"
                    "  [bold]/system[/bold]    Set or show system prompt\n"
                    "  [bold]/history[/bold]   Show conversation history\n"
                    "  [bold]/exit[/bold]      Exit chat",
                    title="commands",
                    border_style="dim",
                ))
                console.print()

            continue

        # ── send message ─────────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        console.print()
        console.print(f"[bold cyan]  {model}[/bold cyan]")

        full_reply: list[str] = []
        thinking_chunks: list[str] = []
        interrupted = False

        try:
            # When show_thinking: render a live thinking panel first, then the answer
            with Live(
                Markdown(""),
                console=console,
                refresh_per_second=15,
                vertical_overflow="visible",
            ) as live:
                in_thinking = False

                for chunk in provider.chat_stream(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=current_system,
                    thinking=current_thinking,
                ):
                    if chunk.startswith(THINK_PREFIX):
                        # Thinking chunk
                        thinking_chunks.append(chunk[len(THINK_PREFIX):])
                        if not in_thinking:
                            in_thinking = True
                        thinking_text = "".join(thinking_chunks)
                        from rich.padding import Padding
                        from rich.columns import Columns
                        live.update(Panel(
                            Markdown(thinking_text),
                            title="[dim]thinking[/dim]",
                            border_style="dim",
                            padding=(0, 1),
                        ))
                    else:
                        # Answer chunk — switch from thinking panel to answer markdown
                        if in_thinking:
                            in_thinking = False
                        full_reply.append(chunk)
                        live.update(Markdown("".join(full_reply)))

        except OceanError as e:
            console.print(f"\n[red]Error:[/red] {e}\n")
            messages.pop()
            continue
        except KeyboardInterrupt:
            interrupted = True

        console.print()

        reply_text = "".join(full_reply)
        if reply_text:
            messages.append({"role": "assistant", "content": reply_text})

        if interrupted:
            _print_info("interrupted")


@main.command("list")
@click.option("--provider", "-p", default=None, help="Filter by provider (openai, gemini, bedrock).")
@click.option("--modality", "-m", default=None, help="Filter by modality (text, vision, image_gen, transcription, tts).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def list_models(provider: str | None, modality: str | None, as_json: bool) -> None:
    """List all supported models.

    \b
    Examples:
      ocean list
      ocean list --provider openai
      ocean list --modality tts
      ocean list -p gemini -m image_gen
      ocean list --json
    """
    from ocean.registry import list_models as _list_models

    models = _list_models(provider=provider, modality=modality)

    if not models:
        click.echo("No models found matching the filters.")
        return

    if as_json:
        import json
        click.echo(json.dumps(models, indent=2))
        return

    # Table output
    # Calculate column widths
    model_w = max(len(m["model"]) for m in models)
    prov_w = max(len(m["provider"]) for m in models)
    mod_w = max(len(", ".join(m["modalities"])) for m in models)
    desc_w = max(len(m["description"]) for m in models)

    model_w = max(model_w, 5)  # "MODEL"
    prov_w = max(prov_w, 8)    # "PROVIDER"
    mod_w = max(mod_w, 10)     # "MODALITIES"
    desc_w = max(desc_w, 11)   # "DESCRIPTION"

    header = f"{'MODEL':<{model_w}}  {'PROVIDER':<{prov_w}}  {'MODALITIES':<{mod_w}}  {'DESCRIPTION'}"
    sep = f"{'─' * model_w}  {'─' * prov_w}  {'─' * mod_w}  {'─' * desc_w}"

    click.echo(header)
    click.echo(sep)
    for m in models:
        mods = ", ".join(m["modalities"])
        click.echo(f"{m['model']:<{model_w}}  {m['provider']:<{prov_w}}  {mods:<{mod_w}}  {m['description']}")

    click.echo(f"\n{len(models)} model(s) found.")
