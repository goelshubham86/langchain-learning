from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

#---- Tools (LangChain @tool decorator) ----

@tool
def get_product_price(product_name: str) -> float:
    """lookup the price of a product in the catalog"""
    print(f"   >> Executing get_product_price(product_name='{product_name}')")
    prices = { "laptop": 1299.99, "mouse": 55.95, "keyboard": 89.50 }
    return prices.get(product_name, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a price and return the final price
    available discount tiers: bronze, silver, gold."""
    print(f"   >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discounts_percentage = { "bronze": 5, "silver": 10, "gold": 15 }
    discount_percentage = discounts_percentage.get(discount_tier, 0)
    return round(price * (1 - discount_percentage / 100), 2)


#---- Agent Loop ----

@traceable(name="Langchain Agent Loop")
def run_agent(query: str) -> str:
    tools = [get_product_price, apply_discount]
    tools_dict = { tool.name: tool for tool in tools }
    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    print(f"Question: {query}")
    print("="*60)
    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that can lookup product prices and apply discounts."
                "You have access to the following tools: {tools_dict}"
                "STRICT RULES - You must follow these rules strictly: "
                "1. NEVER guess or assume any product price."
                "you must call get_product_price() to get the price of a product."
                "2. only call apply_discount() if you have the price of a product." 
                "do not pass a made up price to apply_discount().\n"
                "3. always call apply_discount() with the price of the product."
                "4. if the user does not specify a discount tier "
                "ask the user to specify a discount tier.\n"
            )
        ),

        HumanMessage(content=query)
    ]
    for i in range(MAX_ITERATIONS):
        print(f"=== Iteration {i+1} ===")
        print("="*60)
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls
        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"   >> [Final Answer]: {ai_message.content}")
            return ai_message.content
        # If tool calls, execute the tool calls
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        print(f"   >> [Tool Selected] {tool_name}(**{tool_args})")

        tool_to_call = tools_dict.get(tool_name)
        if tool_to_call is None:
            raise ValueError(f"Tool {tool_name} not found")
        observation = tool_to_call.invoke(tool_args)
        print(f"   [Tool Result]: {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
            )
    print("ERROR: Exceeded maximum number of iterations")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
    