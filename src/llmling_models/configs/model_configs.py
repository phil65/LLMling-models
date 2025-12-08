from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, ImportString, SecretStr
from pydantic_ai.settings import ModelSettings as PyAIModelSettings
from schemez import Schema
from tokonomics import ModelName
from tokonomics.model_names.anthropic import AnthropicModels
from tokonomics.model_names.gemini import GeminiModels
from tokonomics.model_names.openai import OpenaiModels


if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.models.anthropic import AnthropicModelSettings
    from pydantic_ai.models.fallback import FallbackModel
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.models.gemini import GeminiModelSettings
    from pydantic_ai.models.openai import OpenAIChatModelSettings

    from llmling_models import DelegationMultiModel, InputModel
    from llmling_models.models.augmented import AugmentedModel


class BaseModelConfig(Schema):
    """Base for model configurations."""

    type: str = Field(init=False)
    """Type discriminator for model configs."""

    def get_model(self) -> Model:
        """Create and return actual model instance."""
        msg = f"Model creation not implemented for {self.__class__.__name__}"
        raise NotImplementedError(msg)

    def get_model_settings(self) -> PyAIModelSettings:
        """Return model settings as a dictionary."""
        return PyAIModelSettings()


class PrePostPromptConfig(Schema):
    """Configuration for pre/post prompts."""

    text: str = Field(
        examples=["You are a helpful assistant", "Process this carefully"],
        title="Prompt text",
    )
    """The prompt text to be applied."""

    model: ModelName | BaseModelConfig | str = Field(
        examples=[["openai:gpt-5-nano", "anthropic:claude-sonnet-4-5"]],
        title="Model identifier",
    )
    """The model to use for processing the prompt."""


class AugmentedModelConfig(BaseModelConfig):
    """Configuration for model with pre/post prompt processing."""

    type: Literal["augmented"] = Field(default="augmented", init=False)
    """Type identifier for augmented model."""

    main_model: ModelName | BaseModelConfig | str = Field(
        examples=["openai:gpt-5-nano", "anthropic:claude-sonnet-4-5"],
        title="Primary model",
    )
    """The primary model identifier."""

    pre_prompt: PrePostPromptConfig | None = Field(
        default=None,
        title="Pre-processing prompt",
    )
    """Optional configuration for prompt preprocessing."""

    post_prompt: PrePostPromptConfig | None = Field(
        default=None,
        title="Post-processing prompt",
    )
    """Optional configuration for prompt postprocessing."""

    def get_model(self) -> AugmentedModel:
        from llmling_models.models.augmented import AugmentedModel

        main_model = (
            self.main_model.get_model()
            if isinstance(self.main_model, BaseModelConfig)
            else StringModelConfig(identifier=self.main_model).get_model()
        )
        return AugmentedModel(
            main_model=main_model,
            pre_prompt=self.pre_prompt,  # type: ignore
            post_prompt=self.post_prompt,  # type: ignore
        )


class DelegationModelConfig(BaseModelConfig):
    """Configuration for delegation-based model selection."""

    type: Literal["delegation"] = Field(default="delegation", init=False)
    """Type identifier for delegation model."""

    selector_model: ModelName | str | BaseModelConfig = Field(
        examples=["openai:gpt-5-nano"],
        title="Selector model",
    )
    """Model responsible for selecting which model to use."""

    models: list[ModelName | str | BaseModelConfig] = Field(
        min_length=1,
        title="Available models",
        examples=[["openai:gpt-5-nano", "anthropic:claude-sonnet-4-5"]],
    )
    """List of available models to choose from."""

    selection_prompt: str = Field(
        examples=["Choose the best model for this task", "Select appropriate model"],
        title="Selection prompt",
    )
    """Prompt used to guide the selector model's decision."""

    model_descriptions: dict[str, str] | None = Field(
        default=None,
        title="Model descriptions",
        examples=[
            {
                "openai:gpt-5-nano": "A small, fast model",
                "anthropic:claude-sonnet-4-5": "A large, powerful model",
            }
        ],
    )
    """Optional descriptions of each model for selection purposes."""

    def get_model(self) -> DelegationMultiModel:
        from llmling_models.models import DelegationMultiModel

        # Convert selector if it's a config
        selector = (
            self.selector_model.get_model()
            if isinstance(self.selector_model, BaseModelConfig)
            else StringModelConfig(identifier=self.selector_model).get_model()
        )

        # Convert model list
        converted_models = [
            m.get_model()
            if isinstance(m, BaseModelConfig)
            else StringModelConfig(identifier=m).get_model()
            for m in self.models
        ]

        return DelegationMultiModel(
            selector_model=selector,
            models=converted_models,
            selection_prompt=self.selection_prompt,
        )


class FallbackModelConfig(BaseModelConfig):
    """Configuration for fallback strategy."""

    type: Literal["fallback"] = Field(default="fallback", init=False)
    """Type identifier for fallback model."""

    models: list[ModelName | str | BaseModelConfig] = Field(
        min_length=1,
        title="Fallback models",
        examples=[["openai:gpt-5-nano", "anthropic:claude-sonnet-4-5"]],
    )
    """Ordered list of models to try in sequence."""

    def get_model(self) -> FallbackModel:
        from pydantic_ai.models.fallback import FallbackModel

        # Convert nested configs to models
        converted_models = [
            m.get_model()
            if isinstance(m, BaseModelConfig)
            else StringModelConfig(identifier=m).get_model()
            for m in self.models
        ]
        return FallbackModel(*converted_models)


class ImportModelConfig(BaseModelConfig):
    """Configuration for importing external models."""

    type: Literal["import"] = Field(default="import", init=False)
    """Type identifier for import model."""

    model: ImportString[Any] = Field(
        examples=["my_models.CustomModel"],
        title="Model import path",
    )
    """Import path to the model class or function."""

    kw_args: dict[str, str] = Field(default_factory=dict, title="Model arguments")
    """Keyword arguments to pass to the imported model."""

    def get_model(self) -> Any:
        return self.model(**self.kw_args) if isinstance(self.model, type) else self.model


class InputModelConfig(BaseModelConfig):
    """Configuration for human input model."""

    type: Literal["input"] = Field(default="input", init=False)
    """Type identifier for input model."""

    prompt_template: str = Field(
        default="👤 Please respond to: {prompt}",
        examples=["👤 Please respond to: {prompt}", "User input required: {prompt}"],
        title="Prompt display template",
    )
    """Template for displaying the prompt to the user."""

    show_system: bool = Field(default=True, title="Show system messages")
    """Whether to show system messages."""

    input_prompt: str = Field(
        default="Your response: ",
        examples=["Your response: ", "Enter reply: "],
        title="Input request text",
    )
    """Text displayed when requesting input."""

    handler: ImportString[Any] = Field(
        default="llmling_models:DefaultInputHandler",
        validate_default=True,
        title="Input handler",
    )
    """Handler for processing user input."""

    def get_model(self) -> InputModel:
        from llmling_models.models.input_model import InputModel

        return InputModel(
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            handler=self.handler,
        )


class RemoteInputConfig(BaseModelConfig):
    """Configuration for remote human input."""

    type: Literal["remote-input"] = Field(default="remote-input", init=False)
    """Type identifier for remote input model."""

    url: str = Field(
        default="ws://localhost:8000/v1/chat/stream",
        examples=["ws://localhost:8000/v1/chat/stream", "wss://api.example.com/chat"],
        title="WebSocket URL",
    )
    """WebSocket URL for connecting to the remote input service."""

    api_key: SecretStr | None = Field(default=None, title="API key", examples=["abc123"])
    """Optional API key for authentication."""

    def get_model(self) -> Any:
        from llmling_models.models.remote_input import RemoteInputModel

        key = self.api_key.get_secret_value() if self.api_key else None
        return RemoteInputModel(url=self.url, api_key=key)


class RemoteProxyConfig(BaseModelConfig):
    """Configuration for remote model proxy."""

    type: Literal["remote-proxy"] = Field(default="remote-proxy", init=False)
    """Type identifier for remote proxy model."""

    url: str = Field(
        default="ws://localhost:8000/v1/completion/stream",
        examples=[
            "ws://localhost:8000/v1/completion/stream",
            "wss://api.example.com/completion",
        ],
        title="WebSocket URL",
    )
    """WebSocket URL for connecting to the remote model service."""

    api_key: SecretStr | None = Field(default=None, title="API key", examples=["abc123"])
    """Optional API key for authentication."""

    def get_model(self) -> Any:
        from llmling_models.models.remote_model import RemoteProxyModel

        key = self.api_key.get_secret_value() if self.api_key else None
        return RemoteProxyModel(url=self.url, api_key=key)


class UserSelectModelConfig(BaseModelConfig):
    """Configuration for interactive model selection."""

    type: Literal["user-select"] = Field(default="user-select", init=False)
    """Type identifier for user-select model."""

    models: list[ModelName | str | BaseModelConfig] = Field(
        min_length=1,
        title="Selectable models",
        examples=[["openai:gpt-5-nano", "anthropic:claude-sonnet-4-5"]],
    )
    """List of models the user can choose from."""

    prompt_template: str = Field(
        default="🤖 Choose a model for: {prompt}",
        examples=["🤖 Choose a model for: {prompt}", "Select model for: {prompt}"],
        title="Selection prompt template",
    )
    """Template for displaying the choice prompt to the user."""

    show_system: bool = Field(default=True, title="Show system messages")
    """Whether to show system messages during selection."""

    input_prompt: str = Field(
        default="Enter model number (0-{max}): ",
        examples=["Enter model number (0-{max}): ", "Choose (0-{max}): "],
        title="Selection input prompt",
    )
    """Text displayed when requesting model selection."""

    handler: ImportString[Any] = Field(
        default="llmling_models:DefaultInputHandler",
        validate_default=True,
        title="Selection handler",
    )
    """Handler for processing user selection input."""

    def get_model(self) -> Any:
        from llmling_models.models import UserSelectModel

        converted_models = [
            m.get_model()
            if isinstance(m, BaseModelConfig)
            else StringModelConfig(identifier=m).get_model()
            for m in self.models
        ]
        return UserSelectModel(
            models=converted_models,
            prompt_template=self.prompt_template,
            show_system=self.show_system,
            input_prompt=self.input_prompt,
            handler=self.handler,
        )


class StringModelConfig(BaseModelConfig):
    """Configuration for string-based model references."""

    type: Literal["string"] = Field(default="string", init=False)
    """Type identifier for string model."""

    identifier: ModelName | str = Field(
        examples=["openai:gpt-5-nano", "anthropic:claude-sonnet-4-5"],
        title="Model identifier",
    )
    """String identifier for the model."""

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        examples=[1024, 2048, 4096],
        title="Maximum tokens",
    )
    """The maximum number of tokens to generate before stopping."""

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        examples=[0.0, 0.7, 1.0, 2.0],
        title="Temperature",
    )
    """Amount of randomness injected into the response."""

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.9, 1.0],
        title="Top-p (nucleus sampling)",
    )
    """An alternative to sampling with temperature, called nucleus sampling."""

    timeout: float | None = Field(
        default=None,
        ge=0.0,
        examples=[30.0, 60.0, 120.0],
        title="Request timeout",
    )
    """Override the client-level default timeout for a request, in seconds."""

    parallel_tool_calls: bool | None = Field(
        default=None,
        title="Allow parallel tool calls",
    )
    """Whether to allow parallel tool calls."""

    seed: int | None = Field(
        default=None,
        examples=[42, 123, 999],
        title="Random seed",
    )
    """The random seed to use for the model, theoretically allowing for deterministic results."""

    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Presence penalty",
    )
    """Penalize new tokens based on whether they have appeared in the text so far."""

    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Frequency penalty",
    )
    """Penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: dict[str, int] | None = Field(
        default=None,
        title="Logit bias",
        examples=[{"5678": -100}, {"1234": 100}],
    )
    """Modify the likelihood of specified tokens appearing in the completion."""

    stop_sequences: list[str] | None = Field(
        default=None,
        examples=[["STOP", "END"], ["\n\n"]],
        title="Stop sequences",
    )
    """Sequences that will cause the model to stop generating."""

    extra_headers: dict[str, str] | None = Field(
        default=None,
        examples=[{"Custom-Header": "value"}],
        title="Extra headers",
    )
    """Extra headers to send to the model."""

    extra_body: Any | None = Field(
        default=None,
        title="Extra body",
    )
    """Extra body to send to the model."""

    def get_model_settings(self) -> PyAIModelSettings:
        """Get model settings in pydantic-ai format."""
        from pydantic_ai.settings import ModelSettings

        settings = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "parallel_tool_calls": self.parallel_tool_calls,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "stop_sequences": self.stop_sequences,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
        }
        return ModelSettings(**{k: v for k, v in settings.items() if v is not None})  # type: ignore[typeddict-item, no-any-return]

    def get_model(self) -> Model:
        from llmling_models import infer_model

        return infer_model(self.identifier)


class FunctionModelConfig(BaseModelConfig):
    """Configuration for function-based model references."""

    type: Literal["function"] = Field(default="function", init=False)
    """Type identifier for function model."""

    function: ImportString[Callable[..., Any]] = Field(title="Function import path")
    """Function identifier for the model."""

    def get_model(self) -> FunctionModel:
        from llmling_models import function_to_model

        return function_to_model(self.function)


class TestModelConfig(BaseModelConfig):
    """Configuration for test models."""

    type: Literal["test"] = Field(default="test", init=False)
    """Type identifier for test model."""

    custom_output_text: str | None = Field(
        default=None,
        examples=["Test response", "Mock output for testing"],
        title="Custom output text",
    )
    """Optional custom text to return from the test model."""

    call_tools: list[str] | Literal["all"] = Field(
        default="all",
        examples=["all", ["tool1", "tool2"]],
        title="Available tools",
    )
    """Tools that can be called by the test model."""

    def get_model(self) -> Any:
        from pydantic_ai.models.test import TestModel

        return TestModel(
            custom_output_text=self.custom_output_text,
            call_tools=self.call_tools,
        )


class OpenAIModelConfig(BaseModelConfig):
    """Configuration for OpenAI models."""

    type: Literal["openai"] = Field(default="openai", init=False)
    """Type identifier for OpenAI model."""

    identifier: OpenaiModels = Field(
        examples=["openai:gpt-4", "openai:gpt-4-turbo"],
        title="Model identifier",
    )
    """String identifier for the model."""

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        examples=[1024, 2048, 4096],
        title="Maximum tokens",
    )
    """The maximum number of tokens to generate before stopping."""

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        examples=[0.0, 0.7, 1.0, 2.0],
        title="Temperature",
    )
    """Amount of randomness injected into the response."""

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.9, 1.0],
        title="Top-p (nucleus sampling)",
    )
    """An alternative to sampling with temperature, called nucleus sampling."""

    timeout: float | None = Field(
        default=None,
        examples=[30.0, 60.0, 120.0],
        title="Request timeout",
    )
    """Override the client-level default timeout for a request, in seconds."""

    parallel_tool_calls: bool | None = Field(
        default=None,
        title="Allow parallel tool calls",
    )
    """Whether to allow parallel tool calls."""

    seed: int | None = Field(
        default=None,
        examples=[42, 123, 999],
        title="Random seed",
    )
    """The random seed to use for the model, theoretically allowing for deterministic results."""

    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Presence penalty",
    )
    """Penalize new tokens based on whether they have appeared in the text so far."""

    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Frequency penalty",
    )
    """Penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: dict[str, int] | None = Field(
        default=None,
        title="Logit bias",
        examples=[{"5678": -100}, {"1234": 100}],
    )
    """Modify the likelihood of specified tokens appearing in the completion."""

    stop_sequences: list[str] | None = Field(
        default=None,
        examples=[["STOP", "END"], ["\n\n"]],
        title="Stop sequences",
    )
    """Sequences that will cause the model to stop generating."""

    extra_headers: dict[str, str] | None = Field(
        default=None,
        examples=[{"Custom-Header": "value"}],
        title="Extra headers",
    )
    """Extra headers to send to the model."""

    extra_body: Any | None = Field(
        default=None,
        title="Extra body",
    )
    """Extra body to send to the model."""

    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        default=None,
        title="Reasoning effort",
    )
    """Constrains effort on reasoning for reasoning models."""

    logprobs: bool | None = Field(
        default=None,
        title="Include log probabilities",
    )
    """Include log probabilities in the response."""

    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        title="Top log probabilities",
    )
    """Include log probabilities of the top n tokens in the response."""

    user: str | None = Field(
        default=None,
        title="User identifier",
    )
    """A unique identifier representing the end-user."""

    service_tier: Literal["auto", "default", "flex", "priority"] | None = Field(
        default=None,
        title="Service tier",
    )
    """The service tier to use for the model request."""

    def get_model_settings(self) -> OpenAIChatModelSettings:
        """Get model settings in pydantic-ai format."""
        from pydantic_ai.models.openai import OpenAIChatModelSettings

        settings = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "parallel_tool_calls": self.parallel_tool_calls,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "stop_sequences": self.stop_sequences,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "openai_reasoning_effort": self.reasoning_effort,
            "openai_logprobs": self.logprobs,
            "openai_top_logprobs": self.top_logprobs,
            "openai_user": self.user,
            "openai_service_tier": self.service_tier,
        }
        return OpenAIChatModelSettings(**{k: v for k, v in settings.items() if v is not None})  # type: ignore[typeddict-item, no-any-return]

    def get_model(self) -> Any:
        from llmling_models import infer_model

        return infer_model(self.identifier)


class AnthropicModelConfig(BaseModelConfig):
    """Configuration for Anthropic models."""

    type: Literal["anthropic"] = Field(default="anthropic", init=False)
    """Type identifier for Anthropic model."""

    identifier: AnthropicModels = Field(
        examples=["anthropic:claude-3-opus", "anthropic:claude-3-sonnet"],
        title="Model identifier",
    )
    """String identifier for the model."""

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        examples=[1024, 2048, 4096],
        title="Maximum tokens",
    )
    """The maximum number of tokens to generate before stopping."""

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        examples=[0.0, 0.7, 1.0, 2.0],
        title="Temperature",
    )
    """Amount of randomness injected into the response."""

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.9, 1.0],
        title="Top-p (nucleus sampling)",
    )
    """An alternative to sampling with temperature, called nucleus sampling."""

    timeout: float | None = Field(
        default=None,
        examples=[30.0, 60.0, 120.0],
        title="Request timeout",
    )
    """Override the client-level default timeout for a request, in seconds."""

    parallel_tool_calls: bool | None = Field(
        default=None,
        title="Allow parallel tool calls",
    )
    """Whether to allow parallel tool calls."""

    seed: int | None = Field(
        default=None,
        examples=[42, 123, 999],
        title="Random seed",
    )
    """The random seed to use for the model, theoretically allowing for deterministic results."""

    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Presence penalty",
    )
    """Penalize new tokens based on whether they have appeared in the text so far."""

    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Frequency penalty",
    )
    """Penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: dict[str, int] | None = Field(
        default=None,
        title="Logit bias",
        examples=[{"5678": -100}, {"1234": 100}],
    )
    """Modify the likelihood of specified tokens appearing in the completion."""

    stop_sequences: list[str] | None = Field(
        default=None,
        examples=[["STOP", "END"], ["\n\n"]],
        title="Stop sequences",
    )
    """Sequences that will cause the model to stop generating."""

    extra_headers: dict[str, str] | None = Field(
        default=None,
        examples=[{"Custom-Header": "value"}],
        title="Extra headers",
    )
    """Extra headers to send to the model."""

    extra_body: Any | None = Field(
        default=None,
        title="Extra body",
    )
    """Extra body to send to the model."""

    metadata: dict[str, Any] | None = Field(
        default=None,
        title="Request metadata",
    )
    """An object describing metadata about the request."""

    cache_tool_definitions: bool | Literal["5m", "1h"] | None = Field(
        default=None,
        title="Cache tool definitions",
    )
    """Whether to add cache_control to the last tool definition."""

    cache_instructions: bool | Literal["5m", "1h"] | None = Field(
        default=None,
        title="Cache instructions",
    )
    """Whether to add cache_control to the last system prompt block."""

    cache_messages: bool | Literal["5m", "1h"] | None = Field(
        default=None,
        title="Cache messages",
    )
    """Convenience setting to enable caching for the last user message."""

    def get_model_settings(self) -> AnthropicModelSettings:
        """Get model settings in pydantic-ai format."""
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        settings = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "parallel_tool_calls": self.parallel_tool_calls,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "stop_sequences": self.stop_sequences,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "anthropic_metadata": self.metadata,
            "anthropic_cache_tool_definitions": self.cache_tool_definitions,
            "anthropic_cache_instructions": self.cache_instructions,
            "anthropic_cache_messages": self.cache_messages,
        }
        return AnthropicModelSettings(**{k: v for k, v in settings.items() if v is not None})  # type: ignore[typeddict-item, no-any-return]

    def get_model(self) -> Any:
        from llmling_models import infer_model

        return infer_model(self.identifier)


class GeminiModelConfig(BaseModelConfig):
    """Configuration for Gemini models."""

    type: Literal["gemini"] = Field(default="gemini", init=False)
    """Type identifier for Gemini model."""

    identifier: GeminiModels = Field(
        examples=["gemini:gemini-2.0-flash", "gemini:gemini-1.5-pro"],
        title="Model identifier",
    )
    """String identifier for the model."""

    max_tokens: int | None = Field(
        default=None,
        ge=1,
        examples=[1024, 2048, 4096],
        title="Maximum tokens",
    )
    """The maximum number of tokens to generate before stopping."""

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        examples=[0.0, 0.7, 1.0, 2.0],
        title="Temperature",
    )
    """Amount of randomness injected into the response."""

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.1, 0.9, 1.0],
        title="Top-p (nucleus sampling)",
    )
    """An alternative to sampling with temperature, called nucleus sampling."""

    timeout: float | None = Field(
        default=None,
        examples=[30.0, 60.0, 120.0],
        title="Request timeout",
    )
    """Override the client-level default timeout for a request, in seconds."""

    parallel_tool_calls: bool | None = Field(
        default=None,
        title="Allow parallel tool calls",
    )
    """Whether to allow parallel tool calls."""

    seed: int | None = Field(
        default=None,
        examples=[42, 123, 999],
        title="Random seed",
    )
    """The random seed to use for the model, theoretically allowing for deterministic results."""

    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Presence penalty",
    )
    """Penalize new tokens based on whether they have appeared in the text so far."""

    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[-1.0, 0.0, 0.5, 1.0],
        title="Frequency penalty",
    )
    """Penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: dict[str, int] | None = Field(
        default=None,
        title="Logit bias",
        examples=[{"5678": -100}, {"1234": 100}],
    )
    """Modify the likelihood of specified tokens appearing in the completion."""

    stop_sequences: list[str] | None = Field(
        default=None,
        examples=[["STOP", "END"], ["\n\n"]],
        title="Stop sequences",
    )
    """Sequences that will cause the model to stop generating."""

    extra_headers: dict[str, str] | None = Field(
        default=None,
        examples=[{"Custom-Header": "value"}],
        title="Extra headers",
    )
    """Extra headers to send to the model."""

    extra_body: Any | None = Field(
        default=None,
        title="Extra body",
    )
    """Extra body to send to the model."""

    safety_settings: list[dict[str, Any]] | None = Field(
        default=None,
        title="Safety settings",
    )
    """Safety settings options for Gemini model request."""

    thinking_config: dict[str, Any] | None = Field(
        default=None,
        title="Thinking configuration",
    )
    """Thinking features configuration."""

    labels: dict[str, str] | None = Field(
        default=None,
        title="Vertex AI labels",
    )
    """User-defined metadata to break down billed charges."""

    def get_model_settings(self) -> GeminiModelSettings:
        """Get model settings in pydantic-ai format."""
        from pydantic_ai.models.gemini import GeminiModelSettings

        settings = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
            "parallel_tool_calls": self.parallel_tool_calls,
            "seed": self.seed,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "logit_bias": self.logit_bias,
            "stop_sequences": self.stop_sequences,
            "extra_headers": self.extra_headers,
            "extra_body": self.extra_body,
            "gemini_safety_settings": self.safety_settings,
            "gemini_thinking_config": self.thinking_config,
            "gemini_labels": self.labels,
        }
        return GeminiModelSettings(**{k: v for k, v in settings.items() if v is not None})  # type: ignore[typeddict-item, no-any-return]

    def get_model(self) -> Any:
        from llmling_models import infer_model

        return infer_model(self.identifier)


class ModelSettings(Schema):
    """Settings to configure an LLM."""

    max_output_tokens: int | None = Field(
        default=None,
        examples=[1024, 2048, 4096],
        title="Maximum output tokens",
    )
    """The maximum number of tokens to generate."""

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        examples=[0.7, 1.0, 1.5],
        title="Temperature",
    )
    """Amount of randomness in the response (0.0 - 2.0)."""

    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        examples=[0.9, 0.95, 1.0],
        title="Top-p (nucleus sampling)",
    )
    """An alternative to sampling with temperature, called nucleus sampling."""

    timeout: float | None = Field(
        default=None,
        ge=0.0,
        examples=[30.0, 60.0, 120.0],
        title="Request timeout",
    )
    """Override the client-level default timeout for a request, in seconds."""

    parallel_tool_calls: bool | None = Field(
        default=None,
        title="Allow parallel tool calls",
    )
    """Whether to allow parallel tool calls."""

    seed: int | None = Field(default=None, examples=[42, 123, 999], title="Random seed")
    """The random seed to use for the model."""

    presence_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[0.0, 0.5, 1.0],
        title="Presence penalty",
    )
    """Penalize new tokens based on whether they have appeared in the text so far."""

    frequency_penalty: float | None = Field(
        default=None,
        ge=-2.0,
        le=2.0,
        examples=[0.0, 0.5, 1.0],
        title="Frequency penalty",
    )
    """Penalize new tokens based on their existing frequency in the text so far."""

    logit_bias: dict[str, int] | None = Field(
        default=None,
        title="Logit bias",
        examples=[{"5678": -100}, {"1234": 100}],
    )
    """Modify the likelihood of specified tokens appearing in the completion."""

    model_config = ConfigDict(frozen=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to TypedDict format for pydantic-ai."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


AnyModelConfig = Annotated[
    AugmentedModelConfig
    | DelegationModelConfig
    | FallbackModelConfig
    | FunctionModelConfig
    | ImportModelConfig
    | InputModelConfig
    | RemoteInputConfig
    | RemoteProxyConfig
    | StringModelConfig
    | TestModelConfig
    | UserSelectModelConfig
    | OpenAIModelConfig
    | AnthropicModelConfig
    | GeminiModelConfig,
    Field(discriminator="type"),
]
