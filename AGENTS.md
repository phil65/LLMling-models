# AGENTS.md

## Overview

`llmling-models` is a library of extra [pydantic-ai](https://ai.pydantic.dev/) `Model`
implementations, providers and toolsets, built primarily for
[AgentPool](https://github.com/phil65/agentpool) (a downstream consumer: keep the public API
stable). It also ships `llmling_models_config`, a lightweight package of pydantic config
models that build these models, plus a small CLI (OpenAI-compatible server and OAuth logins).

## Architecture

Two import packages live under `src/` (both built into one wheel, see
`[tool.uv.build-backend] module-name`):

- `src/llmling_models/`: the runtime implementation.
  - `__init__.py`: public API re-exports (`MultiModel`, `InputModel`, `DelegationMultiModel`,
    `UserSelectModel`, `ClaudeCodeModel`, `CodeModeToolset`, `infer_model`,
    `function_to_model`, the `ClaudeCode*Tool` classes).
  - `models/`: `Model` subclasses. `multi.py` (`MultiModel` base, a `schemez.Schema` +
    `Model`), `delegation.py`, `userselect.py`, `augmented.py` (pre/post prompts),
    `input_model/` (human input), `remote_input/` and `remote_model/` (each with a client
    `model.py` and a FastAPI `server.py`), `claude_code_model.py` (Claude Agent SDK),
    `function_to_model.py`, `test_model.py` (`FixedArgsTestModel`).
  - `models/helpers.py`: `infer_model()`, which resolves string ids. It handles custom
    prefixes (`openrouter:`, `grok:`, `copilot:`, `lm-studio:`, `zen:`, `anthropic-max:`,
    `input`, `remote_model:`, `remote_input:`, `import:`, `test:`, `claude-code:`, ...) and
    falls back to pydantic-ai.
  - `providers/`: pydantic-ai `Provider` subclasses (Copilot, LM Studio, Requesty, Moondream,
    Zen/opencode, Anthropic Max OAuth). `providers/__init__.py:infer_provider()` extends
    pydantic-ai's provider inference.
  - `auth/`: OAuth/device-flow token handling per provider (`anthropic_auth.py`,
    `github_auth.py`, `gemini_auth.py`, `antigravity_auth.py`, `openai_codex_auth.py`,
    `poe_auth.py`, `zen_auth.py`). `base.py`/`auth_service.py`/`models.py` define a common
    backend registry.
  - `builtin_tools.py`: `AbstractNativeTool` subclasses mapping to Claude Code tools (passed
    via `pydantic_ai.capabilities.NativeTool`).
  - `toolsets/codemode_toolset.py`: `CodeModeToolset`, wraps toolsets into one
    "execute_python" tool. Uses `schemez.ToolsetCodeGenerator`.
  - `tool_bridge.py`: in-process MCP server exposing Python callables to Claude Code.
  - `openai_server/`: OpenAI-compatible FastAPI server (`server.py:OpenAIServer`,
    `model_registry.py:ModelRegistry`).
  - `__main__.py`: argparse CLI `llmling-models` (`serve`, `copilot-auth`, `anthropic-auth`,
    `gemini-auth`, `antigravity-auth`). Imports the server eagerly, needs the `server` extra.
  - `git_hooks/ai_commit_message.py`: pre-commit hook exposed via `.pre-commit-hooks.yaml`.
  - `formatting.py`, `utils.py`, `log.py`: helpers.
- `src/llmling_models_config/configs.py`: discriminated-union config models
  (`AnyModelConfig`, `StringModelConfig`, `OpenAIModelConfig`, `AnthropicModelConfig`, ...).
  Each has `get_model()`, which imports from `llmling_models` lazily. Keep this package
  import-light.

## Where to look

| Task | Location |
|---|---|
| New model-string prefix | `src/llmling_models/models/helpers.py` (`infer_model`) |
| New provider | `src/llmling_models/providers/`, register in `providers/__init__.py` |
| New config type | `src/llmling_models_config/configs.py` + `AnyModelConfig` + `__init__.py` |
| OAuth / tokens | `src/llmling_models/auth/` |
| Claude Code integration | `models/claude_code_model.py`, `builtin_tools.py`, `tool_bridge.py` |
| OpenAI-compatible server / CLI | `openai_server/`, `__main__.py` |
| Public exports | `src/llmling_models/__init__.py` |
| Tests | `tests/` |
| Example script | `examples/codemode.py` |

## Commands

```bash
uv sync --all-extras                 # setup (CI: --no-group docs --no-group lint)
uv run pytest                        # tests (asyncio_mode = "auto")
uv run ruff check . && uv run ruff format --check .
uv run ty check                      # type-check src/ only; warnings fail
uv run deptry src                    # dependency declaration check
uv run mkdocs-mknodes build          # docs (CI: documentation.yml)
uv build
```

`duties.py` wraps these: `duty test`, `duty lint`, `duty lint_check`, `duty build`,
`duty serve`, `duty update`, `duty version <bump>`.

## Conventions

- Python >= 3.13, `from __future__ import annotations` everywhere; ruff with preview,
  line length 100, Google-style docstrings.
- Type checking is ty only (no mypy). Use `# ty: ignore[rule]` sparingly; no mypy-style
  ignores.
- Optional third-party imports (fastapi, uvicorn, claude_agent_sdk, fastmcp, xxhash) are done
  lazily inside functions or in modules that are only imported on demand.
- HTTP: everything uses `httpx2` (pydantic-ai v2 / openai / anthropic SDKs expect httpx2
  clients); don't import `httpx`.
- Tests are plain pytest functions in `tests/test_*.py`; they don't hit the network.
- Conventional commits (commitizen hook in pre-commit).

## Specs

OAuth flows and provider protocols mirrored from other tools (no public spec; the source is the
reference implementation).

| Spec | URL | Targets | Code | Update check |
|---|---|---|---|---|
| Anthropic Claude Pro/Max OAuth + Messages API as Claude Code | https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/auth/oauth/anthropic.ts | mirrors Claude Code CLI 2.1.87 (`CC_VERSION` in `providers/anthropic_max_provider.py`; user-agent, billing header, betas); pi-mono port, commit unrecorded | `src/llmling_models/auth/anthropic_auth.py`, `src/llmling_models/providers/anthropic_max_provider.py` | https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md, pi-mono `anthropic.ts` history |
| OpenAI Codex (ChatGPT Plus/Pro) OAuth | https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/auth/oauth/openai-codex.ts | mirrors the Codex CLI login (`codex-rs/login`), version unknown; pi-mono port, commit unrecorded | `src/llmling_models/auth/openai_codex_auth.py` | https://github.com/openai/codex/tree/main/codex-rs/login, pi-mono `openai-codex.ts` history |
| Gemini CLI OAuth (Cloud Code Assist) | https://github.com/badlogic/pi-mono/blob/ebc60aa825/packages/ai/src/utils/oauth/google-gemini-cli.ts | mirrors Gemini CLI, version unknown; pi-mono port (pi-mono removed it in `fe66edd943`, 2026-04-30; last version `ebc60aa825`) | `src/llmling_models/auth/gemini_auth.py` | https://github.com/google-gemini/gemini-cli/releases |
| Google Antigravity OAuth | https://github.com/badlogic/pi-mono/blob/ebc60aa825/packages/ai/src/utils/oauth/google-antigravity.ts | mirrors Antigravity, version unknown; pi-mono port (removed upstream like Gemini CLI) | `src/llmling_models/auth/antigravity_auth.py` | none upstream any more; check against https://antigravity.google |
| GitHub Copilot device flow + API | https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/auth/oauth/github-copilot.ts | mirrors Copilot Chat 0.35.0 / VS Code 1.107.0 (headers in `auth/github_auth.py`) | `src/llmling_models/auth/github_auth.py`, `src/llmling_models/providers/copilot_provider.py` | pi-mono `github-copilot.ts` history |
| Claude Agent SDK (Python) | https://github.com/anthropics/claude-agent-sdk-python | `claude-agent-sdk>=0.1.16` (`claude` extra; unpinned, tracks latest) | `src/llmling_models/models/claude_code_model.py`, `builtin_tools.py`, `tool_bridge.py` | https://github.com/anthropics/claude-agent-sdk-python/blob/main/CHANGELOG.md |

## Pitfalls

- Copier-managed project (`.copier-answers.yml`, template `copier-phil65`). Don't hand-edit
  template-owned parts of `pyproject.toml`, workflows, `duties.py`, `mkdocs.yml`; add
  dependencies below the `# Only add below (Copier)` markers.
- pydantic-ai moves fast. V2 renamed `builtin_tools` to `native_tools` and removed
  `OpenAIModel` (use `OpenAIChatModel`). Native tools are passed with
  `capabilities=[NativeTool(...)]`. Check imports after `uv lock --upgrade`.
- `llmling_models` imports `pydantic_ai.models.openai` eagerly, hence the
  `pydantic-ai-slim[openai]` dependency.
- Extras: `server`/`remote` (FastAPI), `claude` (claude-agent-sdk), `codemode` (fastmcp),
  `anthropic-max` (anthropic, xxhash), `default` (full pydantic-ai).
- Env vars: `OPENROUTER_API_KEY`, `X_AI_API_KEY`/`GROK_API_KEY`, `DEEPSEEK_API_KEY`,
  `PERPLEXITY_API_KEY`, `GITHUB_COPILOT_API_KEY`, `LM_STUDIO_BASE_URL`, `REQUESTY_API_KEY`,
  `OPENCODE_API_KEY`. OAuth tokens are stored on disk by the `auth/` token stores.
- `TestModelConfig` triggers a harmless pytest collection warning (name starts with `Test`).
- The docs site is the mkdocs-mknodes default site: README.md is the index page, `docs/` is
  empty.
