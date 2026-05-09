# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Self-Driving GC Context

This project uses the **self-driving-gc** skill for autonomous build orchestration.
Workers are dispatched by Gas City's supervisor and execute beads from the queue.

### Key Files
- `GOALS.md` — high-level project goals, priorities, and roadmap context
- `.hermes/cron-state.json` — supervisor state, redirects, coordination
- `state/world-model.json` — learned project knowledge, architecture notes, patterns
- `.hermes/skills/*.md` — skill-specific execution guides (build.md, research.md, refactor.md, test.md, docs.md)

### Worker Workflow
1. Check supervisor redirects (priority override) from `cron-state.json`
2. Read `GOALS.md` for goal context
3. Read `world-model.json` for current project understanding
4. Claim a bead via `bd ready --json` → `bd update <id> --claim`
5. Read the matching skill prompt from `skills/` based on bead category
6. Execute the task, test, commit
7. Close bead, push, update `world-model.json` with learnings
8. Repeat until queue empty

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Build & Test

_Add your build and test commands here_

```bash
# Example:
# npm install
# npm test
```

## Architecture Overview

_Add a brief overview of your project architecture_

## Conventions & Patterns

### Commit Style
- Use conventional commits: `feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`
- Keep commits atomic — one logical change per commit
- Reference bead IDs when applicable: `fix: resolve auth timeout (closes #42)`

### Naming
- Use descriptive variable and function names
- Follow the existing project naming patterns
