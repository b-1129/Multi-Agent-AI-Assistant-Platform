from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.tools import get_tools

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

Use the calculator tool for any arithmetic instead of computing it yourself.
Use the web_search tool for anything current, factual, or that you're not
fully certain about. Otherwise, answer directly and concisely."""

def build_agent():
    llm = ChatGoogleGenerativeAI(
        model = settings.model_name,
        temperature = settings.model_temperature,
        api_key= settings.google_api_key
    )
    return create_agent(
        model=llm,
        tools=get_tools(),
        system_prompt=SYSTEM_PROMPT
    )

agent = build_agent()