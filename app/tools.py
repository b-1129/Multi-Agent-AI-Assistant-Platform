"""
Tools are how an agent reaches outside the LLM to do something real.

Two patterns are shown here on purpose:
1. 'calculator' - a custom tool with a Pydantic args schema, so you can see
   exactly how LangChain validates tool input before it ever runs.
2. 'web_search' - an off-the-shelf community tool (DuckDuckGo, no API key
   needed), so you can see how to plug in a pre-built tool with no extra code.

In a later phase, these same tools get exposed behind an MCP server instead
of being wired directly into the agent - the tool *contract* (name, schema,
description) stays identical, only how it's transported changes.
"""
import ast
import operator

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg
}

# The Problem: Letting an LLM evaluate math by passing a raw string to Python's built-in eval() function is a massive security risk. It allows malicious prompts to inject code that could delete files or hack the system.

# The Solution: This code builds a safe math engine. It uses Python's ast (Abstract Syntax Tree) module to break a math string (like "2 + 3") down into a tree of structural nodes.

# How it works: The _safe_eval function recursively goes through that tree. It only executes the operation if it perfectly matches the strict whitelist defined in _ALLOWED_OPERATORS (Addition, Subtraction, Multiplication, Division, Powers, and Negative numbers). Anything else throws an error.

def _safe_eval(node):
    """Evaluate a restricted arithmetic AST - never use plain eval() on user input."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.operand)
        )
    raise ValueError("Unsupported Expression")


class CalculatorInput(BaseModel):
    expression: str= Field(..., description= "A basic arithmetic operation. e.g '11 * (3 + 4)'.")

def _calculate(expression:str)->str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as exc: # surface the error to the agent, not a crash
        return f"Could not evaluate {expression} : {exc}"
    
calculator_tool = StructuredTool.from_function(
    func=_calculate,
    name="calculator",
    description="Evaluate a basic arithmetic expression (+, -, *, /, **, parentheses).",
    args_schema=CalculatorInput,
)

# LangChain Packaging (StructuredTool.from_function): Converts the plain Python function into a formal LangChain tool object, explicitly attaching its name, strict Pydantic argument structure, and agent-facing instructions.

web_search_tool = DuckDuckGoSearchRun(
    name="web_search",
    description="Search the web for current information. Use for anything "
    "you don't already know or that may have changed recently.",
)

def get_tools():
    return [calculator_tool, web_search_tool]