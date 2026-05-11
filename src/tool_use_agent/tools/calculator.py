"""Calculator tool — evaluates simple math expressions safely."""

import ast
import operator


# Whitelisted operations. ast.parse can read any expression; we only execute
# operations we explicitly allow. Anything else (function calls, attribute
# access, etc.) raises and gets reported back to the LLM as a tool error.
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval(node):
    """Recursively evaluate an AST node, allowing only whitelisted operations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


def calculator(expression: str) -> str:
    """Evaluate a math expression like '3 + 4 * 2' or '(10 / 2) ** 3'.
    
    Returns the result as a string (LLM-readable). Returns an error message
    string if the expression is invalid or contains disallowed operations.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Calculator error: {type(e).__name__}: {e}"