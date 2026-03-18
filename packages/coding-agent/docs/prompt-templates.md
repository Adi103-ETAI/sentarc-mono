# Prompt Templates

Prompt templates are reusable Markdown snippets that you can inject into the system prompt to kick-start reviews, triage workflows, or long instructions. Loading logic lives in `sentarc_coding_agent.core.prompt_templates`.

## Discovery

`load_prompt_templates()` accepts:

1. `~/.arc/agent/prompts/` (global, respects `ARC_CODING_AGENT_DIR`).
2. `<project>/.arc/prompts/` relative to the current working directory.
3. Extra paths supplied via `--prompt-template <path>` (directories and individual files are supported).

Relative arguments resolve against the cwd; `~` expands to your home directory. Pass `--no-prompt-templates`/`-np` to disable discovery entirely.

## File format

Templates are plain text or Markdown with optional YAML frontmatter parsed by `utils.frontmatter.parse_frontmatter()`.

| Frontmatter key | Description |
| --- | --- |
| `name` | Friendly identifier (defaults to the filename stem). |
| `description` | Human-readable summary shown in future pickers. |

`load_prompt_template_from_file()` strips the frontmatter block and stores the body inside `PromptTemplate.content`. Parsing issues are returned as warnings alongside the template list so your tooling can surface them.

## API usage

`load_prompt_templates(...)` returns `(templates, warnings)` where each template is `PromptTemplate(name, content, file_path, description)`. Consumers decide how to surface them—common patterns include:

- Prepending `content` to `system_prompt.build_system_prompt()`.
- Turning `name` into a slash command (TypeScript exposes `/templatename`; the Python interactive mode has not wired this UI yet).
- Feeding descriptions into selection menus.

Until the CLI exposes a first-class picker, you can call the loader inside extensions or RPC clients and inject the `content` wherever makes sense for your workflow.
