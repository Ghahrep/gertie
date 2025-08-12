# in main.py
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session # <-- THIS IS THE MISSING IMPORT

# Import your new orchestrator
from agents.orchestrator import FinancialOrchestrator
from api.routes import users, auth, portfolios # Import the routers
from db.session import get_db # Import your dependency
from api.schemas import User # Import the User schema

# --- API Application Setup ---
app = FastAPI(
    title="Gertie.ai - Multi-Agent Financial Platform",
    description="An API for interacting with a team of specialized financial AI agents.",
    version="1.0.0"
)

# Add CORS middleware - ADD THIS SECTION
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",  # In case you need alternative port
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include the routers in your main app
app.include_router(users.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(portfolios.router, prefix="/api/v1")

# Create a single, long-lived instance of the orchestrator
orchestrator = FinancialOrchestrator()

# --- API Request & Response Models ---
class ChatRequest(BaseModel):
    query: str

# --- API Endpoint ---
@app.post("/api/v1/chat", tags=["Agent Interaction"])
def chat_with_agent_team(
    request: ChatRequest, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(auth.get_current_user)
):
    """
    This is the main endpoint for communicating with the financial agent platform.
    It now provides the agent system with the authenticated user and a database session.
    """
    print(f"Authenticated user '{current_user.email}' sent query: '{request.query}'")
    
    # Pass the user and db session to the orchestrator
    result_data = orchestrator.route_query(
        user_query=request.query, 
        db_session=db, 
        current_user=current_user
    )
    return {"data": result_data}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Gertie.ai API. Please use the /docs endpoint to interact."}