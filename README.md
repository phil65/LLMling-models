# LLMling-models

[![PyPI License](https://img.shields.io/pypi/l/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Package status](https://img.shields.io/pypi/status/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Monthly downloads](https://img.shields.io/pypi/dm/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Distribution format](https://img.shields.io/pypi/format/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Wheel availability](https://img.shields.io/pypi/wheel/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Python version](https://img.shields.io/pypi/pyversions/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Implementation](https://img.shields.io/pypi/implementation/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Releases](https://img.shields.io/github/downloads/phil65/llmling-models/total.svg)](https://github.com/phil65/llmling-models/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/llmling-models)](https://github.com/phil65/llmling-models/graphs/contributors)
[![Github Forks](https://img.shields.io/github/forks/phil65/llmling-models)](https://github.com/phil65/llmling-models/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/llmling-models)](https://github.com/phil65/llmling-models/issues)
[![Github Issues](https://img.shields.io/github/issues-pr/phil65/llmling-models)](https://github.com/phil65/llmling-models/pulls)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/llmling-models)](https://github.com/phil65/llmling-models/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/llmling-models)](https://github.com/phil65/llmling-models/stars)
[![Github Repository size](https://img.shields.io/github/repo-size/phil65/llmling-models)](https://github.com/phil65/llmling-models)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/llmling-models)](https://github.com/phil65/llmling-models/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/llmling-models)](https://github.com/phil65/llmling-models/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/llmling-models)](https://github.com/phil65/llmling-models)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/llmling-models)](https://github.com/phil65/llmling-models)
[![Package status](https://codecov.io/gh/phil65/llmling-models/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/llmling-models/)
[![PyUp](https://pyup.io/repos/github/phil65/llmling-models/shield.svg)](https://pyup.io/repos/github/phil65/llmling-models/)

Extra [pydantic-ai](https://ai.pydantic.dev/) models, providers and toolsets: human-in-the-loop
models, model routing, prompt augmentation, remote models, a "code mode"
toolset and an extended `infer_model()`. Built for
[AgentPool](https://github.com/phil65/agentpool), but everything works with plain pydantic-ai.

**Documentation:** https://phil65.github.io/LLMling-models/

## Installation

```bash
uv add llmling-models
```

Optional extras:

| Extra | Enables |
|---|---|
| `server` | OpenAI-compatible server and the `llmling-models` CLI |
| `remote` | Remote model / remote input servers (FastAPI, websockets) |
| `codemode` | fastmcp, for the MCP tool bridge (`llmling_models.tool_bridge`) |
| `default` | The full `pydantic-ai` package with all of its providers |

## Quickstart

```python
from pydantic_ai import Agent
from llmling_models import infer_model

# Drop-in replacement for pydantic-ai's model inference, with extra prefixes
model = infer_model("test:Hello from llmling-models")
agent = Agent(model)
result = agent.run_sync("Say hi")
print(result.output)  # Hello from llmling-models
```

## Features

- **Extended `infer_model()`**: extra prefixes (`openrouter:`, `grok:`, `deepseek:`,
  `perplexity:`, `copilot:`, `lm-studio:`, `zen:`, `input`,
  `remote_model:`, `remote_input:`, `import:`, `test:`) and comma-separated fallback chains.
- **Models**: `InputModel`, `UserSelectModel`, `DelegationMultiModel`, `AugmentedModel`,
  `RemoteInputModel`, `RemoteProxyModel`, `function_to_model()`.
- **Providers**: GitHub Copilot, LM Studio, Requesty, Moondream, OpenCode Zen.
- **`CodeModeToolset`**: exposes tools as a Python API the model can script against.
- **Config models**: `llmling_models_config` holds pydantic configs (`StringModelConfig`,
  `OpenAIModelConfig`, `AnthropicModelConfig`, ...) with a `get_model()` factory.
- **CLI**: OpenAI-compatible API server and OAuth logins.

## Extended infer_model

```python
from llmling_models import infer_model

model = infer_model("openai:gpt-4o")  # standard pydantic-ai ids still work
model = infer_model("openrouter:anthropic/claude-sonnet-4.5")  # OPENROUTER_API_KEY
model = infer_model("grok:grok-4")  # X_AI_API_KEY or GROK_API_KEY
model = infer_model("deepseek:deepseek-chat")  # DEEPSEEK_API_KEY
model = infer_model("perplexity:sonar")  # PERPLEXITY_API_KEY
model = infer_model("copilot:gpt-4o-mini")  # GitHub Copilot token
model = infer_model("lm-studio:model-name")  # local LM Studio server

model = infer_model("input")  # human types the answer
model = infer_model("remote_model:ws://host:8000/v1/completion/stream")
model = infer_model("remote_input:ws://host:8000/v1/chat/stream")
model = infer_model("import:my_package.models:MyModel")  # import a Model from a path
model = infer_model("test:Custom response")  # TestModel with fixed output

# Fallback chain: tries models in order
model = infer_model("openai:gpt-4o,anthropic:claude-sonnet-4-5")
```

| Provider | Environment variable |
|---|---|
| OpenRouter | `OPENROUTER_API_KEY` |
| Grok (X.AI) | `X_AI_API_KEY` or `GROK_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |
| Perplexity | `PERPLEXITY_API_KEY` |
| Copilot | `GITHUB_COPILOT_API_KEY` (or `llmling-models copilot-auth`) |
| LM Studio | `LM_STUDIO_BASE_URL` (default `http://localhost:1234/v1`) |
| Requesty | `REQUESTY_API_KEY` |
| OpenCode Zen | `OPENCODE_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |

## Models

### Input Model

Delegates responses to a human, for testing, debugging or hybrid human-AI workflows:

```python
from pydantic_ai import Agent
from llmling_models import InputModel

model = InputModel(
    prompt_template="Question: {prompt}",
    show_system=True,
    input_prompt="Your answer: ",
)
agent = Agent(model=model, system_prompt="You are helping test an input model.")
result = await agent.run("What's your favorite color?")
print(result.output)
```

Custom input sources can be plugged in via the `handler` argument (see
`llmling_models.models.input_model.input_handlers`).

### User Select Model

Lets the user pick which model answers each prompt:

```python
from pydantic_ai import Agent
from llmling_models import UserSelectModel

model = UserSelectModel(models=["openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"])
agent = Agent(model)
result = await agent.run("What is the meaning of life?")
```

### Delegation Model

A selector model picks the most suitable model for each prompt:

```python
from pydantic_ai import Agent
from llmling_models import DelegationMultiModel

model = DelegationMultiModel(
    selector_model="openai:gpt-4o-mini",
    models=["openai:gpt-4o", "openai:gpt-4o-mini"],
    model_descriptions={
        "openai:gpt-4o": "Complex reasoning, math and coding tasks",
        "openai:gpt-4o-mini": "Simple queries and chat",
    },
    selection_prompt="Select the most appropriate model for the task.",
)
agent = Agent(model)
result = await agent.run("Solve this complex mathematical proof...")
```

### Augmented Model

Pre- and post-processes prompts with auxiliary models:

```python
from pydantic_ai import Agent
from llmling_models.models import AugmentedModel
from llmling_models.models.augmented import PrePostPromptConfig

model = AugmentedModel(
    main_model="openai:gpt-4o",
    pre_prompt=PrePostPromptConfig(
        text="Expand this question: {input}", model="openai:gpt-4o-mini"
    ),
    post_prompt=PrePostPromptConfig(
        text="Summarize this response concisely: {output}", model="openai:gpt-4o-mini"
    ),
)
agent = Agent(model)
result = await agent.run("What is AI?")
```

When streaming (`agent.run_stream()`, `agent.iter()`), only the pre-prompt is applied: the main
model's stream is passed through unchanged, so the post-prompt is skipped.

### Remote Input and Remote Proxy Models

Forward requests to a remote human operator (`RemoteInputModel`) or to a model served elsewhere
(`RemoteProxyModel`), over WebSocket (streaming) or REST. Requires the `remote` extra.

```python
from pydantic_ai import Agent
from llmling_models.models import RemoteInputModel

model = RemoteInputModel(url="ws://operator:8000/v1/chat/stream", api_key="your-api-key")
agent = Agent(model)
result = await agent.run("What's the meaning of life?")
```

Serving a pydantic-ai model for `RemoteProxyModel`:

```python
from llmling_models.models.remote_model.server import ModelServer

server = ModelServer(model="openai:gpt-4o-mini", api_key="your-secret-key")
server.run(host="0.0.0.0", port=8000)
```

The matching operator server for `RemoteInputModel` lives in
`llmling_models.models.remote_input.server`.

## CodeModeToolset

`CodeModeToolset` wraps other toolsets and exposes a single Python execution tool in which all
wrapped tools are available as async functions. LLMs are good at writing code, so chaining several
tool calls in one script is often more reliable and cheaper than classic tool calling
([background](https://blog.cloudflare.com/code-mode/)).

```python
import webbrowser

from pydantic_ai import Agent
from pydantic_ai.toolsets.function import FunctionToolset
from llmling_models import CodeModeToolset

browser_toolset = FunctionToolset(tools=[webbrowser.open])
code_toolset = CodeModeToolset(toolsets=[browser_toolset])

agent = Agent(model="openai:gpt-4o-mini", toolsets=[code_toolset])
async with agent:
    result = await agent.run("Open google.com in a new tab")
    # The model writes e.g.: await open("https://google.com", new=2)
```

## Providers

pydantic-ai `Provider` implementations for OpenAI-compatible services:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from llmling_models.providers.lm_studio_provider import LMStudioProvider

provider = LMStudioProvider(base_url="http://localhost:1234/v1")
model = OpenAIChatModel("model-name", provider=provider)  # a model loaded in LM Studio
agent = Agent(model)
```

`llmling_models.providers.infer_provider()` extends pydantic-ai's provider lookup with
`copilot`, `grok`, `perplexity`, `lm-studio`, `requesty`, `zen` and
`opencode-go`, and falls back to pydantic-ai for everything else.

## Model configs

`llmling_models_config` is a lightweight package of pydantic models for declaring models in
config files:

```python
from llmling_models_config import StringModelConfig

config = StringModelConfig(identifier="openai:gpt-4o-mini", temperature=0.2)
model = config.get_model()
settings = config.get_model_settings()
```

## CLI

With the `server` extra installed:

```bash
# OpenAI-compatible API server
llmling-models serve --model gpt4=openai:gpt-4o --port 8000 --api-key secret
llmling-models serve --config models.yml
llmling-models serve --auto-discover

# GitHub Copilot device-flow login
llmling-models copilot-auth
```

Anthropic does not permit other tools to log in with a Claude plan
([Claude Code legal and compliance](https://code.claude.com/docs/en/legal-and-compliance)), so
there is no Claude Pro/Max login here; use an Anthropic API key.

## License

MIT
