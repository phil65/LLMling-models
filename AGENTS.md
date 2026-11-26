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
    `UserSelectModel`, `CodeModeToolset`, `infer_model`, `function_to_model`).
  - `models/`: `Model` subclasses. `multi.py` (`MultiModel` base, a `schemez.Schema` +
    `Model`), `delegation.py`, `userselect.py`, `augmented.py` (pre/post prompts),
    `input_model/` (human input), `remote_input/` and `remote_model/` (each with a client
    `model.py` and a FastAPI `server.py`), `function_to_model.py`, `test_model.py` (`FixedArgsTestModel`).
  - `models/helpers.py`: `infer_model()`, which resolves string ids. It handles custom
    prefixes (`openrouter:`, `grok:`, `copilot:`, `lm-studio:`, `zen:`,
    `input`, `remote_model:`, `remote_input:`, `import:`, `test:`, ...) and
    falls back to pydantic-ai.
  - `providers/`: pydantic-ai `Provider` subclasses (Copilot, LM Studio, Requesty, Moondream,
    Zen/opencode). `providers/__init__.py:infer_provider()` extends
    pydantic-ai's provider inference.
  - `auth/`: OAuth/device-flow token handling per provider (`github_auth.py`,
    `openai_codex_auth.py`, `poe_auth.py`, `zen_auth.py`).
    `base.py`/`auth_service.py`/`models.py` define a common backend registry.
  - `toolsets/codemode_toolset.py`: `CodeModeToolset`, wraps toolsets into one
    "execute_python" tool. Uses `schemez.ToolsetCodeGenerator`.
  - `tool_bridge.py`: in-process MCP server exposing Python callables to MCP clients.
  - `openai_server/`: OpenAI-compatible FastAPI server (`server.py:OpenAIServer`,
    `model_registry.py:ModelRegistry`).
  - `__main__.py`: argparse CLI `llmling-models` (`serve`, `copilot-auth`).
    Imports the server eagerly, needs the `server` extra.
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

- Python >= 3.14, no `from __future__ import annotations` (PEP 649; banned via ruff TID251); ruff with preview,
  line length 100, Google-style docstrings.
- Type checking is ty only (no mypy). Use `# ty: ignore[rule]` sparingly; no mypy-style
  ignores.
- Optional third-party imports (fastapi, uvicorn, fastmcp) are done
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
| OpenAI Codex (ChatGPT Plus/Pro) OAuth | https://github.com/earendil-works/pi/blob/main/packages/ai/src/auth/oauth/openai-codex.ts | mirrors the Codex CLI login (`codex-rs/login`), version unknown; pi-mono port, commit unrecorded | `src/llmling_models/auth/openai_codex_auth.py` | https://github.com/openai/codex/tree/main/codex-rs/login, pi-mono `openai-codex.ts` history |
| GitHub Copilot device flow + API | https://github.com/earendil-works/pi/blob/main/packages/ai/src/auth/oauth/github-copilot.ts | mirrors Copilot Chat 0.35.0 / VS Code 1.107.0 (headers in `auth/github_auth.py`) | `src/llmling_models/auth/github_auth.py`, `src/llmling_models/providers/copilot_provider.py` | pi-mono `github-copilot.ts` history |

## Pitfalls

- Copier-managed project (`.copier-answers.yml`, template `copier-phil65`). Don't hand-edit
  template-owned parts of `pyproject.toml`, workflows, `duties.py`, `mkdocs.yml`; add
  dependencies below the `# Only add below (Copier)` markers.
- pydantic-ai moves fast. V2 renamed `builtin_tools` to `native_tools` and removed
  `OpenAIModel` (use `OpenAIChatModel`). Native tools are passed with
  `capabilities=[NativeTool(...)]`. Check imports after `uv lock --upgrade`.
- `llmling_models` imports `pydantic_ai.models.openai` eagerly, hence the
  `pydantic-ai-slim[openai]` dependency.
- Extras: `server`/`remote` (FastAPI), `codemode` (fastmcp),
  `default` (full pydantic-ai).
- Env vars: `OPENROUTER_API_KEY`, `X_AI_API_KEY`/`GROK_API_KEY`, `DEEPSEEK_API_KEY`,
  `PERPLEXITY_API_KEY`, `GITHUB_COPILOT_API_KEY`, `LM_STUDIO_BASE_URL`, `REQUESTY_API_KEY`,
  `OPENCODE_API_KEY`. OAuth tokens are stored on disk by the `auth/` token stores.
- Anthropic Claude Pro/Max OAuth (`anthropic-max:`, `anthropic-auth`) was removed: Anthropic
  does not permit third-party tools to log in with or route requests through Claude plan
  credentials (https://code.claude.com/docs/en/legal-and-compliance).
- `ClaudeCodeModel` (`claude-code:`, the `claude` extra) and the `ClaudeCode*Tool` native tools
  were removed (2026-11-26); agentpool has its own Claude Code agent.
- Gemini CLI and Antigravity OAuth logins were removed: Google ended "Login with Google" for
  Gemini CLI on consumer tiers (2026-06-18) and the Antigravity terms forbid using Antigravity
  OAuth from third-party tools (account suspension). Use `GEMINI_API_KEY` / Vertex instead.
- `TestModelConfig` triggers a harmless pytest collection warning (name starts with `Test`).
- The docs site is the mkdocs-mknodes default site: README.md is the index page, `docs/` is
  empty.
