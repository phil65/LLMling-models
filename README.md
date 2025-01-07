# LLMling-models

[![PyPI License](https://img.shields.io/pypi/l/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Package status](https://img.shields.io/pypi/status/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Daily downloads](https://img.shields.io/pypi/dd/llmling-models.svg)](https://pypi.org/project/llmling-models/)
[![Weekly downloads](https://img.shields.io/pypi/dw/llmling-models.svg)](https://pypi.org/project/llmling-models/)
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
[![Github commits this week](https://img.shields.io/github/commit-activity/w/phil65/llmling-models)](https://github.com/phil65/llmling-models)
[![Github commits this month](https://img.shields.io/github/commit-activity/m/phil65/llmling-models)](https://github.com/phil65/llmling-models)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/llmling-models)](https://github.com/phil65/llmling-models)
[![Package status](https://codecov.io/gh/phil65/llmling-models/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/llmling-models/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyUp](https://pyup.io/repos/github/phil65/llmling-models/shield.svg)](https://pyup.io/repos/github/phil65/llmling-models/)

# llmling-models

Collection of model wrappers and adapters for use with [LLMling-Agent](https://github.com/phil65/llmling-agent), but should work with the underlying pydantic-ai API without issues.

**WARNING**:

This is just a prototype for now and will likely change in the future.
Also, pydantic-ais APIs dont seem stable yet, so things might not work across all pydantic-ai versions.
I will try to keep this up to date as fast as possible.

## Available Models

### LLM Library Adapter

Adapter to use models from the [LLM library](https://llm.datasette.io/) with Pydantic-AI:

```python
from pydantic_ai import Agent
from llmling_models.llm_adapter import LLMAdapter

# Basic usage
adapter = LLMAdapter(model_name="gpt-4o-mini")
agent = Agent(model=adapter)
result = await agent.run("Write a short poem")

# Streaming support
async with agent.run_stream("Test prompt") as response:
    async for chunk in response.stream():
        print(chunk)

# Usage statistics
result = await agent.run("Test prompt")
usage = result.usage()
print(f"Request tokens: {usage.request_tokens}")
print(f"Response tokens: {usage.response_tokens}")
```
(Examples need to be wrapped in async function and run with `asyncio.run`)

### AISuite Adapter

Adapter to use models from [AISuite](https://github.com/andrewyng/aisuite) with Pydantic-AI:

```python
from pydantic_ai import Agent
from llmling_models.aisuite_adapter import AISuiteAdapter

# Basic usage
adapter = AISuiteAdapter(model="model_name")
agent = Agent(adapter)
result = await agent.run("Write a story")
```

### Multi-Models

#### Fallback Model

Tries models in sequence until one succeeds. Perfect for handling rate limits or service outages:

```python
from llmling_models import FallbackMultiModel

fallback_model = FallbackMultiModel(
    models=[
        "openai:gpt-4",           # Try this first
        "openai:gpt-3.5-turbo",   # Fallback option
        "anthropic:claude-2"       # Last resort
    ]
)
agent = Agent(fallback_model)
result = await agent.run("Complex question")
```


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
print(f"You responded: {result.data}")

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


#### Cost-Optimized Model

Selects models based on input cost limits, automatically choosing the most appropriate model within your budget constraints:

```python
from pydantic_ai import Agent
from llmling_models import CostOptimizedMultiModel

# Use cheapest model that can handle the task
cost_model = CostOptimizedMultiModel(
    models=[
        "openai:gpt-4",           # More expensive
        "openai:gpt-3.5-turbo",   # Less expensive
    ],
    max_input_cost=0.1,          # Maximum cost in USD per request
    strategy="cheapest_possible"  # Use cheapest model that fits
)

# Or use the best model within budget
cost_model = CostOptimizedMultiModel(
    models=[
        "openai:gpt-4-32k",      # Most expensive
        "openai:gpt-4",          # Medium cost
        "openai:gpt-3.5-turbo",  # Cheapest
    ],
    max_input_cost=0.5,              # Higher budget
    strategy="best_within_budget"     # Use best model within budget
)

agent = Agent(cost_model)
result = await agent.run("Your prompt here")
```

#### Token-Optimized Model

Automatically selects models based on input token count and context window requirements:

```python
from pydantic_ai import Agent
from llmling_models import TokenOptimizedMultiModel

# Create model that automatically handles different context lengths
token_model = TokenOptimizedMultiModel(
    models=[
        "openai:gpt-4-32k",        # 32k context
        "openai:gpt-4",            # 8k context
        "openai:gpt-3.5-turbo",    # 4k context
    ],
    strategy="efficient"           # Use smallest sufficient context window
)

# Or maximize context window availability
token_model = TokenOptimizedMultiModel(
    models=[
        "openai:gpt-4-32k",        # 32k context
        "openai:gpt-4",            # 8k context
        "openai:gpt-3.5-turbo",    # 4k context
    ],
    strategy="maximum_context"     # Use largest available context window
)

agent = Agent(token_model)

# Will automatically select appropriate model based on input length
result = await agent.run("Your long prompt here...")

# Long inputs automatically use models with larger context windows
result = await agent.run("Very long document..." * 1000)
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
    protocol="websocket",
    api_key="your-api-key"
)

# Or use REST API
model = RemoteInputModel(
    url="http://operator:8000/v1/chat",
    protocol="rest",
    api_key="your-api-key"
)

agent = Agent(model)

# The request will be forwarded to the remote operator
result = await agent.run("What's the meaning of life?")
print(f"Remote operator responded: {result.data}")

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
from pydantic_ai.models import infer_model
from llmling_models.remote_model.server import ModelServer

# Get any pydantic-ai model
model = infer_model("openai:gpt-4")

# Create and start server
server = ModelServer(
    model=model,
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
    model=infer_model("openai:gpt-4"),
    api_key="dev-key"
)
server.run(host="localhost", port=8000)
```

For production, you'll typically want to run it on a public server with proper authentication:

```python
server = ModelServer(
    model=infer_model("openai:gpt-4"),
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

## Installation

```bash
pip install llmling-models
```

## Requirements

- Python 3.12+
- pydantic-ai
- llm (optional, for LLM adapter)
- aisuite (optional, for aisuite adapter)
- Either tokenizers or transformers for improved token calculation

## License

MIT
