# ============================================================
# main.py — The FastAPI Application
#
# This is the entry point of the backend.
# It creates the FastAPI app, adds CORS, and includes the router.
# ============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import chat

# Create the FastAPI app
app = FastAPI(
    title="BigCompany Inc. Customer Service API",
    description="A LangGraph-powered customer service agent backend.",
)

# CORS middleware — allows the frontend to call our API.
# Without this, the browser would block requests from the frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the chat router (this adds the /chat endpoint)
app.include_router(chat.router)


# A simple health check endpoint
@app.get("/")
def read_root():
    return {"message": "BigCompany Inc. Customer Service API is running!"}