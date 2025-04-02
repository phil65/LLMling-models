"""Command-line interface for running the OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path
import sys
from typing import Any, cast

from pydantic_ai.models import Model
import yaml

from llmling_models.log import get_logger
from llmling_models.openai_server import ModelRegistry, OpenAIServer


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an OpenAI-compatible API server using LLMling models."
    )

    # Server configuration
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on (default: 8000)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication (default: none, no authentication)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="LLMling OpenAI-Compatible API",
        help="API title for documentation",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="OpenAI-compatible API server powered by LLMling models",
        help="API description for documentation",
    )

    # Model configuration
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--auto-discover",
        action="store_true",
        help="Auto-discover available models from tokonomics",
    )
    model_group.add_argument(
        "--config", type=str, metavar="FILE", help="Path to YAML configuration file"
    )
    model_group.add_argument(
        "--model",
        action="append",
        metavar="MODEL_NAME=MODEL_ID",
        help="Add model mapping (can be specified multiple times)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def parse_models_arg(models_arg: list[str]) -> dict[str, str]:
    """Parse model mapping arguments."""
    result = {}
    for mapping in models_arg:
        if "=" not in mapping:
            logger.warning(
                "Ignoring invalid model mapping '%s': missing '=' separator", mapping
            )
            continue

        name, model_id = mapping.split("=", 1)
        result[name.strip()] = model_id.strip()

    return result


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    try:
        with path.open() as f:
            config = yaml.safe_load(f)

        # Validate required sections
        if not isinstance(config, dict):
            logger.error("Invalid configuration: root must be a mapping")
            sys.exit(1)
    except Exception:
        logger.exception("Failed to load configuration")
        sys.exit(1)
    else:
        return config


def process_config(config: dict[str, Any]) -> dict[str, str | Model]:
    """Process configuration and return model mappings."""
    models: dict[str, str | Model] = {}

    if "model_list" in config:
        for model_entry in config["model_list"]:
            if "model_name" not in model_entry or "litellm_params" not in model_entry:
                logger.warning("Skipping invalid model entry: %s", model_entry)
                continue

            name = model_entry["model_name"]

            if "model" in model_entry["litellm_params"]:
                # Convert LiteLLM format to our format
                provider_model = model_entry["litellm_params"]["model"]
                if "/" in provider_model:
                    provider, model_id = provider_model.split("/", 1)
                    if provider in ("openai", "azure"):
                        model_id = f"openai:{model_id}"
                    elif provider == "anthropic":
                        model_id = f"anthropic:{model_id}"
                    else:
                        model_id = f"{provider}:{model_id}"
                else:
                    model_id = provider_model

                models[name] = model_id
            else:
                logger.warning(
                    "Skipping model '%s': missing 'model' in litellm_params", name
                )

    if "llmling_models" in config:
        models.update(config["llmling_models"])

    return models


def process_server_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Process server settings from config."""
    settings = {
        "host": "0.0.0.0",
        "port": 8000,
        "api_key": None,
        "title": "LLMling-models OpenAI-Compatible API",
        "description": "OpenAI-compatible API server powered by LLMling-models",
    }

    if "server_settings" in config:
        server_config = config["server_settings"]
        if isinstance(server_config, dict):
            if "host" in server_config:
                settings["host"] = server_config["host"]
            if "port" in server_config:
                settings["port"] = server_config["port"]
            if "title" in server_config:
                settings["title"] = server_config["title"]
            if "description" in server_config:
                settings["description"] = server_config["description"]

    # Process authentication settings
    if "auth_settings" in config:
        auth_cfg = config["auth_settings"]
        if isinstance(auth_cfg, dict) and auth_cfg.get("enabled", False):
            settings["api_key"] = auth_cfg.get("admin_key") or auth_cfg.get("api_key")

    # Environment variables can override settings
    if "api_key" in os.environ:
        settings["api_key"] = os.environ["API_KEY"]
    if "OPENAI_API_KEY" in os.environ:
        settings["api_key"] = os.environ["OPENAI_API_KEY"]

    return settings


async def main() -> None:
    """Main entry point for the server."""
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Determine models to use
    models: dict[str, str | Model] = {}
    config_settings: dict[str, Any] | None = None

    if args.auto_discover:
        logger.info("Auto-discovering models from tokonomics...")
        try:
            registry = await ModelRegistry.create()
            logger.info("Discovered %d models", len(registry.models))

            # Convert registry to models dict
            models = dict(registry.models.items())
        except Exception:
            logger.exception("Failed to auto-discover models")
            sys.exit(1)

    elif args.config:
        logger.info("Loading configuration from %s", args.config)
        config = load_config(args.config)
        models = process_config(config)
        config_settings = process_server_settings(config)
        logger.info("Loaded %d models from configuration", len(models))

    elif args.model:
        logger.info("Using models specified on command line")
        models = cast(dict[str, str | Model], parse_models_arg(args.model))
        logger.info("Specified %d models", len(models))

    else:
        logger.info("No models specified, using default set")
        # Default set of models
        models = {
            "gpt-4": "openai:gpt-4",
            "gpt-4o-mini": "openai:gpt-4o-mini",
            "gpt-3.5-turbo": "openai:gpt-3.5-turbo",
        }

    if not models:
        logger.error("No models available, exiting")
        sys.exit(1)

    # Server settings
    host = config_settings["host"] if config_settings else args.host
    port = config_settings["port"] if config_settings else args.port
    api_key = config_settings["api_key"] if config_settings else args.api_key
    title = config_settings["title"] if config_settings else args.title
    description = config_settings["description"] if config_settings else args.description

    # Initialize models
    registry = ModelRegistry(models)

    # Create server
    server = OpenAIServer(
        registry=registry,
        api_key=api_key,
        title=title,
        description=description,
    )

    # Run server
    import uvicorn

    logger.info("Starting server at %s:%d with %d models", host, port, len(models))
    uvicorn_config = uvicorn.Config(
        app=server.app,
        host=host,
        port=port,
        log_level=args.log_level.lower(),
    )
    server_instance = uvicorn.Server(uvicorn_config)
    await server_instance.serve()


def main_cli() -> None:
    """Command-line entry point for the server."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception:
        logger.exception("Server error: %s")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
