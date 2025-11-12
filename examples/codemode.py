"""Simple test script for CodeModeToolset."""

from __future__ import annotations

import asyncio
import logging
import sys
import webbrowser

from pydantic_ai import Agent
from pydantic_ai.toolsets.function import FunctionToolset

from llmling_models.codemode_toolset import CodeModeToolset


async def main():
    """Test the CodeModeToolset with a simple example."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Create a function toolset with webbrowser.open
    function_toolset = FunctionToolset(tools=[webbrowser.open])

    # Wrap it in CodeModeToolset
    code_toolset = CodeModeToolset([function_toolset])

    print("Testing CodeModeToolset...")
    print("Available toolsets:", len(code_toolset.toolsets))

    # Create agent with the toolset
    agent = Agent(model="openai:gpt-4o-mini", toolsets=[code_toolset])

    async with agent:
        # Test 1: Simple code execution
        print("\n=== Test 1: Simple code execution ===")
        result = await agent.run(
            "Execute this Python code: print('Hello from CodeModeToolset!')"
        )
        print(f"Result: {result}")

        # Test 2: Using the wrapped tool
        print("\n=== Test 2: Using wrapped browser tool ===")
        result = await agent.run(
            "Open google.com in a new browser tab using the available tools"
        )
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
