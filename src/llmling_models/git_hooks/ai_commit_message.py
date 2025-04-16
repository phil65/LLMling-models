"""Git hook to generate commit messages using AI."""

from __future__ import annotations

import os
import subprocess
import sys

from pydantic_ai import Agent

from llmling_models import infer_model


def get_staged_diff() -> str:
    """Get the diff of staged changes."""
    result = subprocess.run(
        ["git", "diff", "--staged"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def main() -> None:
    """Generate and set AI commit message."""
    # Find the commit message file
    if len(sys.argv) > 1:
        commit_msg_file = sys.argv[1]
    else:
        # When run through pre-commit, we need to find the COMMIT_EDITMSG file
        git_dir = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        commit_msg_file = os.path.join(git_dir, "COMMIT_EDITMSG")  # noqa: PTH118
        if not os.path.exists(commit_msg_file):  # noqa: PTH110
            print("Cannot find commit message file, exiting")
            return

    # Check if this is a merge commit, etc. where we shouldn't interfere
    # Only check when we have command-line args (when Git runs us directly)
    if len(sys.argv) > 2 and sys.argv[2] in ("message", "template", "merge", "squash"):  # noqa: PLR2004
        return

    # Get the staged changes
    diff = get_staged_diff()
    if not diff:
        print("No staged changes found, skipping AI commit message generation")
        return

    try:
        print("Generating commit message with AI...")

        # Initialize an agent with a suitable model from your library
        model = infer_model("openai:gpt-3.5-turbo")
        agent = Agent(model=model)

        # Create prompt for generating commit message
        prompt = f"""
        Generate a concise, informative git commit message based on the following diff.
        Use the conventional commits format: type(scope): description

        Types: feat, fix, docs, style, refactor, test, chore

        The message should be no more than 72 characters for the first line.
        Do not include explanations or details about your reasoning.
        Just return the commit message.

        Here's the diff:
        {diff}
        """

        # Generate commit message
        result = agent.run_sync(prompt)
        commit_message = result.output.strip()
        print(f"Generated message: {commit_message}")

        # Write the AI-generated message to the commit message file
        with open(commit_msg_file, "w") as f:  # noqa: PTH123
            f.write(commit_message)

    except Exception as e:  # noqa: BLE001
        print(f"Error generating commit message: {e}")
        # Don't fail the commit if AI generation fails


if __name__ == "__main__":
    main()
