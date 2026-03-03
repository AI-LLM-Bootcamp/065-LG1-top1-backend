# ============================================================
# agent.py — The LangGraph Customer Service Agent
#
# This file contains the SAME agent from notebook 028,
# packaged as a module that the FastAPI backend can import.
# ============================================================

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables (the OpenAI API key)
load_dotenv()


# ============================================================
# 1. THE "DATABASE" (simulated with dictionaries)
# ============================================================
# In a real app, this data would come from a real database.
# For this demo, we use simple Python dictionaries.

ORDERS_DB = {
    "ORD-1001": {
        "customer": "Alice Johnson",
        "product": "Laptop Pro 15",
        "price": 1299.99,
        "status": "delivered",
        "delivery_date": "2026-02-05",
    },
    "ORD-1002": {
        "customer": "Bob Smith",
        "product": "Wireless Headphones X3",
        "price": 199.99,
        "status": "shipped",
        "estimated_delivery": "2026-02-14",
    },
    "ORD-1003": {
        "customer": "Carol Davis",
        "product": "Smartphone Ultra 12",
        "price": 899.99,
        "status": "processing",
        "estimated_delivery": "2026-02-18",
    },
}

RETURN_POLICY = {
    "standard_return_window": "30 days from delivery",
    "refund_processing_time": "5-7 business days",
    "conditions": [
        "Item must be in original packaging",
        "Item must not show signs of damage caused by the customer",
        "Proof of purchase (order ID) is required",
    ],
    "non_returnable": ["Earbuds (hygiene reasons)", "Software licenses"],
}


# ============================================================
# 2. THE TOOLS (functions the agent can call)
# ============================================================
# These are the same 3 tools from notebook 028.
# The @tool decorator makes them usable by the LLM.

@tool
def check_order_status(order_id: str) -> str:
    """Check the current status of a customer order.

    Use this tool when a customer asks about their order status,
    delivery date, or order details.

    Args:
        order_id: The order ID (e.g., 'ORD-1001').
    """
    order = ORDERS_DB.get(order_id)

    if not order:
        return f"Order '{order_id}' not found. Please check the order ID and try again."

    info = f"Order {order_id}:\n"
    info += f"  Customer: {order['customer']}\n"
    info += f"  Product: {order['product']}\n"
    info += f"  Price: ${order['price']}\n"
    info += f"  Status: {order['status']}\n"

    if order["status"] == "delivered":
        info += f"  Delivered on: {order['delivery_date']}"
    else:
        info += f"  Estimated delivery: {order['estimated_delivery']}"

    return info


@tool
def lookup_return_policy() -> str:
    """Look up BigCompany Inc.'s return and refund policy.

    Use this tool when a customer asks about returns, refunds,
    or the return policy.
    """
    policy = "BigCompany Inc. Return Policy:\n"
    policy += f"  Return window: {RETURN_POLICY['standard_return_window']}\n"
    policy += f"  Refund processing: {RETURN_POLICY['refund_processing_time']}\n"
    policy += "  Conditions:\n"
    for condition in RETURN_POLICY["conditions"]:
        policy += f"    - {condition}\n"
    policy += "  Non-returnable items:\n"
    for item in RETURN_POLICY["non_returnable"]:
        policy += f"    - {item}\n"
    return policy


@tool
def process_return_request(order_id: str, reason: str) -> str:
    """Process a return request for a customer order.

    Use this tool when a customer confirms they want to return a product.
    Only use this AFTER checking the order status and confirming
    the customer wants to proceed.

    Args:
        order_id: The order ID to process the return for.
        reason: The customer's reason for returning the product.
    """
    order = ORDERS_DB.get(order_id)

    if not order:
        return f"Cannot process return: Order '{order_id}' not found."

    if order["status"] != "delivered":
        return (
            f"Cannot process return: Order {order_id} has status '{order['status']}'. "
            f"Returns can only be processed for delivered orders."
        )

    return (
        f"Return request APPROVED for order {order_id} ({order['product']}).\n"
        f"  Reason: {reason}\n"
        f"  Refund amount: ${order['price']}\n"
        f"  Refund will be processed within {RETURN_POLICY['refund_processing_time']}.\n"
        f"  A return shipping label has been sent to the customer's email."
    )


# Collect all tools in a list
tools = [check_order_status, lookup_return_policy, process_return_request]


# ============================================================
# 3. THE LLM (the agent's brain)
# ============================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a friendly and helpful customer service agent for BigCompany Inc., \
an online electronics store.

Your responsibilities:
- Help customers check their order status.
- Answer questions about our return and refund policy.
- Process return requests when customers want to return a product.
- Be polite, empathetic, and professional at all times.

Important guidelines:
- Always greet the customer warmly.
- When a customer asks about an order, use the check_order_status tool.
- When a customer asks about returns or refunds, use the lookup_return_policy tool.
- Before processing a return, always confirm with the customer first.
- If you cannot help with something, let the customer know you will \
connect them with a human agent.
- Keep responses concise but helpful.
"""


# ============================================================
# 4. THE GRAPH (the agent's workflow)
# ============================================================

# Node 1: The Assistant (calls the LLM)
def assistant(state: MessagesState):
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# Node 2: The Tool Executor
tool_node = ToolNode(tools)

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", tool_node)
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile with memory (so the agent remembers conversations)
memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)

print("BigCompany Inc. LangGraph agent loaded successfully!")