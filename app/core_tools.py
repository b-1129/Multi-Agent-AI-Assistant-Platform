"""
The actual business logic behind each tool, with no framework wrapping.

Phase 1-3 wrapped these directly as LangChain StructuredTools in
`app/tools.py`. Phase 4 exposes the *same* logic over MCP instead
(`mcp_server/server.py`) -- this module is the one place that logic lives,
so both the MCP server and the local fallback tools in `app/tools.py` call
the same code instead of two copies drifting apart.
"""

import ast
import operator

from app.vectorstore import get_vectorstore

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node):
    """Evaluate a restricted arithmetic AST -- never use plain eval() on user input."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
        return _ALLOWED_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")

def search_documents(query: str, k: int = 4) -> str:
    """Search the user's uploaded documents for relevant passages."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    if not results:
        return "No relevant chunks found. The user may not have uploaded any documents yet."

    formatted = []
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        formatted.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(formatted)

def web_search(query: str) -> str:
    """Search the web for current information."""
    # Imported lazily: DuckDuckGoSearchRun pulls in langchain_community, which
    # we'd rather not import at all in the lightweight MCP server process if
    # we don't have to -- but in practice both processes need it, so this
    # just keeps the import next to its one use.
    from langchain_community.tools import DuckDuckGoSearchRun

    return DuckDuckGoSearchRun().run(query)