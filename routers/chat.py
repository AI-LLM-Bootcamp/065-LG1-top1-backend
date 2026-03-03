# ============================================================
# routers/chat.py — The Chat API Endpoint
#
# This is the "door" between the frontend and the LangGraph agent.
# It receives messages from the frontend, passes them to the agent,
# and sends the agent's response back.
# ============================================================

from fastapi import APIRouter
from langchain_core.messages import HumanMessage

from schemas import ChatRequest, ChatResponse
from agent import graph

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message to the BigCompany Inc. customer service agent.

    - The `message` is what the customer typed.
    - The `thread_id` identifies the conversation.
      Same thread_id = same conversation (the agent remembers context).
    """

    # The config tells LangGraph which conversation thread to use.
    # This is how the agent "remembers" previous messages.
    config = {"configurable": {"thread_id": request.thread_id}}

    # Send the customer's message to the LangGraph agent.
    # The agent will:
    #   1. Read the message
    #   2. Decide if it needs to use a tool
    #   3. If yes: call the tool, get results, think again
    #   4. Generate a final response
    result = graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config,
    )

    # The result contains ALL messages in the conversation.
    # The last message is the agent's final response.
    agent_response = result["messages"][-1].content

    return ChatResponse(
        response=agent_response,
        thread_id=request.thread_id,
    )