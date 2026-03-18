# Skills

`sentarc_coding_agent.core.skills` turns Markdown playbooks into structured metadata so the agent knows when to `read` project-specific instructions before acting.

## Discovery & precedence

When `--no-skills` is **not** supplied, `load_skills()` gathers entries in this order:

1. `~/.arc/agent/skills/**` (respecting `ARC_CODING_AGENT_DIR`).
2. `<project>/.arc/skills/**` rooted at the current working directory.
3. Every file or directory passed via `--skill <path>`.

Each unique skill name can win only once. The first successful load takes the slot; subsequent collisions append a `ResourceDiagnostic(type="collision")` that records the winner and loser paths, letting you clean up duplicates.

## File format & validation

Each skill lives in its own directory. By convention the directory name **is** the skill name, and a `SKILL.md` file holds the content, though any `.md` file works. Files may start with YAML frontmatter parsed by `utils.frontmatter.parse_frontmatter()`.

| Frontmatter key | Required | Behaviour |
| --- | --- | --- |
| `name` | No | Overrides the directory name. Must remain lowercase alphanumeric plus single hyphens (see `_validate_name`). |
| `description` | Yes | Short summary (≤1024 chars). Missing descriptions prevent the skill from loading. |
| `disable-model-invocation` | No | When `true`, the skill stays on disk but is hidden from the LLM so it will never be auto-loaded; humans can still open the file manually. |

`load_skill_from_file()` enforces name/description limits, trims whitespace, and emits `ResourceDiagnostic(type="warning")` entries for anything off-spec. Hidden directories, `node_modules`, and files ignored by `.gitignore` (via `pathspec`, when installed) are skipped.

## Dataclass shape

A loaded `Skill` includes:

- `name`, `description`, and `file_path` – used in prompts and tooling.
- `base_dir` – helpful when resolving relative links mentioned inside the Markdown.
- `source` – one of `user`, `project`, or `path` so you can see where it came from.
- `disable_model_invocation` – copied from the frontmatter flag.

## Prompt integration

`system_prompt.build_system_prompt()` calls `format_skills_for_prompt()` whenever the `read` tool is enabled. The helper renders:

```
<available_skills>
  <skill>
    <name>deploy-service</name>
    <description>Redeploys the staging service via Terraform</description>
    <location>/abs/path/.arc/skills/deploy-service/SKILL.md</location>
  </skill>
</available_skills>
```

During a run the model can issue `read` commands to inspect those files before using tools like `bash`, `edit`, or `write`.

## CLI controls

- `--skill <path>` – load additional directories or single Markdown files (relative paths resolve against the cwd; `~` expands to your home directory).
- `--no-skills` / `-ns` – disable discovery entirely.

Because skills are plain Markdown, you can version them alongside your codebase and reuse them across projects.
