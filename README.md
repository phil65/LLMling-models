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

[Read the documentation!](https://phil65.github.io/llmling-models/)

# llmling-models

Collection of model wrappers and adapters for use with Pydantic-AI.
WARNING:

This is just a quick first shot for now and will likely change in the future.
Also, pydantic-ais APIs dont seem stable yet, so things might not work across all pydantic-ai version.
I will try to keep this up to date as fast as possible.

## Available Models

### LLM Library Adapter

Adapter to use models from the [LLM library](https://llm.datasette.io/) with Pydantic-AI:

```python
from pydantic_ai import Agent
from llmling_models.llm_adapter import LLMAdapter

# Basic usage
adapter = LLMAdapter(model_name="gpt-4o-mini")
agent = Agent(adapter)
result = await agent.run("Write a short poem")

# Streaming support
async with agent.run_stream("Test prompt") as response:
    async for chunk in response.stream_text():
        print(chunk, end="")

# Usage statistics
result = await agent.run("Test prompt")
usage = result.usage()
print(f"Request tokens: {usage.request_tokens}")
print(f"Response tokens: {usage.response_tokens}")
```
(Examples need to be wrapped in async function and run with `asyncio.run`)

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

#### Random Model

Randomly selects from a pool of models for each request. Useful for load balancing or A/B testing:

```python
from pydantic_ai import Agent
from llmling_models import RandomMultiModel

# Create random model with multiple options
random_model = RandomMultiModel(
    models=["openai:gpt-4", "openai:gpt-3.5-turbo"]
)
agent = Agent(random_model)

# Each call will randomly use one of the models
result = await agent.run("What is AI?")
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


## Installation

```bash
pip install llmling-models
```

## Requirements

- Python 3.12+
- pydantic-ai
- llm (optional, for LLM adapter)

## License

MIT
