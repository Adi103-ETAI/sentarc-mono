"""Run extension hooks."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from sentarc_coding_agent.core.extensions.types import Extension, ExtensionContext


async def run_hook(
    extensions: List[Extension],
    hook_name: str,
    context: ExtensionContext,
    **kwargs: Any,
) -> None:
    """Run a named hook across all extensions."""
    for ext in extensions:
        hook = getattr(ext, hook_name, None)
        if hook is None:
            continue
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(context, **kwargs)
            else:
                hook(context, **kwargs)
        except Exception as e:
            print(f"Warning: Extension '{ext.name}' hook '{hook_name}' failed: {e}")


async def run_extension_command(
    extensions: List[Extension],
    command_name: str,
    args: str,
    context: ExtensionContext,
) -> Optional[str]:
    """Run a registered extension command. Returns result string or None if not found."""
    for ext in extensions:
        for cmd in ext.commands:
            if cmd.name == command_name:
                try:
                    if asyncio.iscoroutinefunction(cmd.handler):
                        return await cmd.handler(args, context)
                    else:
                        return cmd.handler(args, context)
                except Exception as e:
                    return f"Extension command '/{command_name}' failed: {e}"
    return None


def get_extension_commands(extensions: List[Extension]) -> Dict[str, str]:
    """Get all registered extension commands as {name: description}."""
    commands: Dict[str, str] = {}
    for ext in extensions:
        for cmd in ext.commands:
            commands[cmd.name] = cmd.description
    return commands
