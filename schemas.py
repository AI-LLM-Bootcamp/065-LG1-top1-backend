# ============================================================
# schemas.py — Data shapes for the Chat API
#
# Pydantic models define what data goes in and out of the API.
# Think of them as a "contract" between the frontend and backend.
# ============================================================

from pydantic import BaseModel


class ChatRequest(BaseModel):
    """What the frontend sends to the backend."""
    message: str       # The customer's message (e.g., "What's my order status?")
    thread_id: str     # Conversation ID (same ID = same conversation)


class ChatResponse(BaseModel):
    """What the backend sends back to the frontend."""
    response: str      # The agent's reply
    thread_id: str     # Same conversation ID (so the frontend knows which conversation)