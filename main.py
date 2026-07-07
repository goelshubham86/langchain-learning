import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()
# tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


# @tool
# def search_web(query: str) -> str:
#     """Search the web for information about the query

#     Args:
#         query: The query to search the web for

#     Returns:
#         The search results
#     """
#     return tavily_client.search(query=query)


llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    print("Hello from langchain-learning!")
    result = agent.invoke(
        {
            "messages": HumanMessage(
                content="Search  3 job posting for an AI Engineer using langchain in Austin area on linkedin and list their details."
            )
        }
    )
    print(result)


if __name__ == "__main__":
    main()
