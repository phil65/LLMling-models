"""CodeModeToolset helpers."""

from __future__ import annotations


def validate_code(python_code: str) -> None:
    """Validate code structure and raise ModelRetry for fixable issues."""
    from pydantic_ai import ModelRetry

    code = python_code.strip()
    if not code:
        msg = (
            "Empty code provided. Please write code inside 'async def main():' function."
        )
        raise ModelRetry(msg)

    if "async def main(" not in code:
        msg = (
            "Code must be wrapped in 'async def main():' function. "
            "Please rewrite your code like:\n"
            "async def main():\n"
            "    # your code here\n"
            "    return result"
        )
        raise ModelRetry(msg)

    # Check if code contains a return statement
    if "return " not in code:
        msg = (
            "The main() function should return a value. "
            "Add 'return result' or 'return \"completed\"' at the end of your function."
        )
        raise ModelRetry(msg)
