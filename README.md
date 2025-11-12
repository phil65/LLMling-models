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
[![Github Discussions](https://img.shields.io/github/discussions/phil65/llmling-models)](https://github.com/phil65/llmling-models/discussions)
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

# llmling-models

Collection of model wrappers and adapters for use with [LLMling-Agent](https://github.com/phil65/llmling-agent), but should work with the underlying pydantic-ai API without issues.

**WARNING**:

This is just a prototype for now and will likely change in the future.
Also, pydantic-ais APIs dont seem stable yet, so things might not work across all pydantic-ai versions.
I will try to keep this up to date as fast as possible.

## CodeModeToolset

The `CodeModeToolset` wraps other pydantic-ai toolsets and provides Python code execution with all their tools available as async functions. This approach is more effective than traditional tool calling because LLMs have extensive training on real-world code but limited exposure to synthetic tool-calling examples. When tools are presented as a programming API, LLMs can handle more complex scenarios, chain multiple operations efficiently without token overhead between calls, and leverage their code-writing strengths. (more info: [https://blog.cloudflare.com/code-mode/](https://blog.cloudflare.com/code-mode/))

```python
import webbrowser
from pydantic_ai import Agent
from pydantic_ai.toolsets.function import FunctionToolset
from llmling_models import CodeModeToolset

# Create toolsets with tools
browser_toolset = FunctionToolset(tools=[webbrowser.open])

# Wrap in CodeModeToolset  
code_toolset = CodeModeToolset([browser_toolset])

# Use with agent
agent = Agent(model="openai:gpt-4", toolsets=[code_toolset])

async with agent:
    result = await agent.run("Open google.com in a new tab using Python code")
    # The LLM can now write: await open("https://google.com", new=2)
```

The toolset generates documentation for all available tools and handles the execution environment automatically.

## Available Models


### Multi-Models


### Augmented Model

Enhances prompts through pre- and post-processing steps using auxiliary language models:

```python
from llmling_models import AugmentedModel

model = AugmentedModel(
    main_model="openai:gpt-4",
    pre_prompt={
        "text": "Expand this question: {input}",
        "model": "openai:gpt-3.5-turbo"
    },
    post_prompt={
        "text": "Summarize this response concisely: {output}",
        "model": "openai:gpt-3.5-turbo"
    }
)
agent = Agent(model)

# The question will be expanded before processing
# and the response will be summarized afterward
result = await agent.run("What is AI?")
```

### Input Model

A model that delegates responses to human input, useful for testing, debugging, or creating hybrid human-AI workflows:

```python
from pydantic_ai import Agent
from llmling_models import InputModel

# Basic usage with default console input
model = InputModel(
    prompt_template="🤖 Question: {prompt}",
    show_system=True,
    input_prompt="Your answer: ",
)

# Create agent with system context
agent = Agent(
    model=model,
    system_prompt="You are helping test an input model. Be concise.",
)

# Run interactive conversation
result = await agent.run("What's your favorite color?")
print(f"You responded: {result.output}")

# Supports streaming input
async with agent.run_stream("Tell me a story...") as response:
    async for chunk in response.stream():
        print(chunk, end="", flush=True)
```

Features:
- Interactive console input for testing and debugging
- Support for streaming input (character by character, but not "true" async with default handler)
- Configurable message formatting
- Custom input handlers for different input sources
- System message display control
- Full conversation context support

This model is particularly useful for:
- Testing complex prompt chains
- Creating hybrid human-AI workflows
- Debugging agent behavior
- Collecting human feedback
- Educational scenarios where human input is needed


### User Select Model

An interactive model that lets users manually choose which model to use for each prompt:

```python
from pydantic_ai import Agent
from llmling_models import UserSelectModel

# Basic setup with model list
model = UserSelectModel(
    models=["openai:gpt-4o-mini", "openai:gpt-3.5-turbo", "anthropic:claude-3"]
)

agent = Agent(model)

# The user will be shown the prompt and available models,
# and can choose which one to use for the response
result = await agent.run("What is the meaning of life?")
```

#### Model Delegation

Dynamically selects models based on given prompt. Uses a selector model to choose the most appropriate model for each task:

```python
from pydantic_ai import Agent
from llmling_models import DelegationMultiModel

# Basic setup with model list
delegation_model = DelegationMultiModel(
    selector_model="openai:gpt-4-turbo",
    models=["openai:gpt-4", "openai:gpt-3.5-turbo"],
    selection_prompt="Pick gpt-4 for complex tasks, gpt-3.5-turbo for simple queries."
)

# Advanced setup with model descriptions
delegation_model = DelegationMultiModel(
    selector_model="openai:gpt-4-turbo",
    models=["openai:gpt-4", "anthropic:claude-2", "openai:gpt-3.5-turbo"],
    model_descriptions={
        "openai:gpt-4": "Complex reasoning, math problems, and coding tasks",
        "anthropic:claude-2": "Long-form analysis and research synthesis",
        "openai:gpt-3.5-turbo": "Simple queries, chat, and basic information"
    },
    selection_prompt="Select the most appropriate model for the task."
)

agent = Agent(delegation_model)

# The selector model will analyze the prompt and choose the most suitable model
result = await agent.run("Solve this complex mathematical proof...")
```

The cost-optimized model ensures you stay within budget while getting the best possible model for your needs, while the token-optimized model automatically handles varying input lengths by selecting models with appropriate context windows.


### Remote Input Model

A model that connects to a remote human operator, allowing distributed human-in-the-loop operations:

```python
from pydantic_ai import Agent
from llmling_models import RemoteInputModel

# Basic setup with WebSocket (preferred for streaming)
model = RemoteInputModel(
    url="ws://operator:8000/v1/chat/stream",
    api_key="your-api-key"
)

# Or use REST API
model = RemoteInputModel(
    url="http://operator:8000/v1/chat",
    api_key="your-api-key"
)

agent = Agent(model)

# The request will be forwarded to the remote operator
result = await agent.run("What's the meaning of life?")
print(f"Remote operator responded: {result.output}")

# Streaming also works with WebSocket protocol
async with agent.run_stream("Tell me a story...") as response:
    async for chunk in response.stream():
        print(chunk, end="", flush=True)
```


Features:
- Distributed human-in-the-loop operations
- WebSocket support for real-time streaming
- REST API for simpler setups
- Full conversation context support
- Secure authentication via API keys

#### Setting up a Remote Model Server

Setting up a remote model server is straightforward. You just need a pydantic-ai model and can start serving it:

```python
from llmling_models.remote_model.server import ModelServer

# Create and start server
server = ModelServer(
    model="openai:gpt-4",
    api_key="your-secret-key",  # Optional authentication
)
server.run(port=8000)
```

That's it! The server now accepts both REST and WebSocket connections and handles all the message protocol details for you.

Features:
- Simple setup - just provide a model
- Optional API key authentication
- Automatic handling of both REST and WebSocket protocols
- Full pydantic-ai message protocol support
- Usage statistics forwarding
- Built-in error handling and logging

For development, you might want to run the server locally:

```python
server = ModelServer(
    model="openai:gpt-4",
    api_key="dev-key"
)
server.run(host="localhost", port=8000)
```

For production, you'll typically want to run it on a public server with proper authentication:

```python
server = ModelServer(
    model="openai:gpt-4",
    api_key="your-secure-key",  # Make sure to use a strong key
    title="Production GPT-4 Server",
    description="Serves GPT-4 model for production use"
)
server.run(
    host="0.0.0.0",  # Accept connections from anywhere
    port=8000,
    workers=4  # Multiple workers for better performance
)
```

Both REST and WebSocket protocols are supported, with WebSocket being preferred for streaming capabilities. They also maintain the full pydantic-ai message protocol, ensuring compatibility with all features of the framework.



All multi models are generically typed to follow pydantic best practices. Usefulness for that is debatable though. :P

## Providers

LLMling-models extends the capabilities of pydantic-ai with additional provider implementations that make it easy to connect to various LLM API services.

### Available Providers

The package includes the following provider implementations:


#### GitHub Copilot Provider

Connect to GitHub Copilot's API for code-focused tasks (requires token management):

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from llmling_models.providers.copilot_provider import CopilotProvider

# Requires tokonomics.CopilotTokenManager to handle token management
provider = CopilotProvider()  # Uses tokonomics for authentication
model = OpenAIModel("gpt-4o-mini", provider=provider)
agent = Agent(model=model)
result = await agent.run("Write a function to calculate Fibonacci numbers")
```

#### LM Studio Provider

Connect to local LM Studio inference server for open-source models:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from llmling_models.providers.lm_studio_provider import LMStudioProvider

provider = LMStudioProvider(base_url="http://localhost:11434/v1")
model = OpenAIModel("model_name", provider=provider)  # Use model loaded in LM Studio
agent = Agent(model=model)
result = await agent.run("Tell me about yourself")
```

### Provider Utility Functions

#### infer_provider

The `infer_provider` function extends pydantic-ai's provider inference to include all LLMling-models providers:

```python
from llmling_models.providers import infer_provider

# Get provider by name
provider = infer_provider("openrouter")  # Returns OpenRouterProvider instance
provider = infer_provider("grok")        # Returns GrokProvider instance
provider = infer_provider("perplexity")  # Returns PerplexityProvider instance
provider = infer_provider("copilot")     # Returns CopilotProvider instance
provider = infer_provider("lm-studio")   # Returns LMStudioProvider instance

# Still works with standard providers too
provider = infer_provider("openai")      # Returns pydantic_ai's OpenAIProvider
```

## Extended infer_model Function

LLMling-models provides an extended `infer_model` function that resolves various model notations to appropriate instances:

```python
from llmling_models import infer_model

# Provider prefixes (requires appropriate API keys as env vars)
model = infer_model("openai:gpt-4o")             # OpenAI models
model = infer_model("openrouter:anthropic/opus") # OpenRouter (requires OPENROUTER_API_KEY)
model = infer_model("grok:grok-2-1212")          # Grok/X.AI (requires X_AI_API_KEY)
model = infer_model("perplexity:sonar-medium")   # Perplexity (requires PERPLEXITY_API_KEY)
model = infer_model("deepseek:deepseek-chat")    # DeepSeek (requires DEEPSEEK_API_KEY)
model = infer_model("copilot:gpt-4o-mini")       # GitHub Copilot (requires token management)
model = infer_model("lm-studio:model-name")      # LM Studio local models

# LLMling's special models
model = infer_model("simple-openai:gpt-4")      # Simple HTTPX-based OpenAI client
model = infer_model("input")                    # Interactive human input model
model = infer_model("remote_model:ws://url")    # Remote model proxy
model = infer_model("remote_input:ws://url")    # Remote human input
model = infer_model("import:module.path:Class") # Import model from Python path

# Testing
model = infer_model("test:Custom response")     # Test model with fixed output
```

The function provides a fallback to a simple HTTPX-based OpenAI client in environments where the full OpenAI library is not available (like Pyodide/WebAssembly contexts).

### Environment Variable Configuration

For convenience, most providers support configuration via environment variables:

| Provider    | Environment Variable    | Purpose                    |
|-------------|-------------------------|----------------------------|
| OpenRouter  | `OPENROUTER_API_KEY`    | API key for authentication |
| Grok (X.AI) | `X_AI_API_KEY` or `GROK_API_KEY` | API key for authentication |
| DeepSeek    | `DEEPSEEK_API_KEY`      | API key for authentication |
| Perplexity  | `PERPLEXITY_API_KEY`    | API key for authentication |
| Copilot     | Uses tokonomics token management | - |
| LM Studio   | `LM_STUDIO_BASE_URL`    | Base URL for local server |
| OpenAI      | `OPENAI_API_KEY`        | API key for authentication |
```


## Installation

```bash
pip install llmling-models
```

## Requirements

- Python 3.12+
- pydantic-ai
- Either tokenizers or transformers for improved token calculation

## License

MIT
