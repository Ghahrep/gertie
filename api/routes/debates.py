# api/routes/debates.py
"""
Multi-Agent Debate API Routes - Simplified Implementation
========================================================
Incremental implementation matching your existing codebase patterns
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import uuid

# Import your existing patterns
from api.routes.auth import get_current_user
from api.schemas import User
from db.session import get_db
from db import models,crud

# Initialize router following your pattern
router = APIRouter()

# ============================================================================
# REQUEST/RESPONSE SCHEMAS (matching your existing schema patterns)
# ============================================================================

from pydantic import BaseModel, Field

class DebateCreateRequest(BaseModel):
    """Request to create a new debate - simplified"""
    topic: str = Field(..., description="Investment question for debate", min_length=10)
    preferred_agents: Optional[List[str]] = Field(None, description="Preferred agent types")
    urgency: str = Field(default="medium", description="Urgency level")
    
    class Config:
        schema_extra = {
            "example": {
                "topic": "Should I reduce portfolio risk given current market volatility?",
                "preferred_agents": ["quantitative_analyst", "market_intelligence"],
                "urgency": "medium"
            }
        }

class DebateResponse(BaseModel):
    """Basic debate response"""
    debate_id: str
    topic: str
    status: str
    created_at: str
    participants: List[str]
    current_stage: Optional[str] = None

# ============================================================================
# CORE API ENDPOINTS (minimal implementation)
# ============================================================================

@router.post("/debates", response_model=DebateResponse, tags=["Debates"])
async def create_debate(
    request: DebateCreateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    🎭 Create New Multi-Agent Debate
    
    Start a debate between AI agents on an investment question.
    """
    
    try:
        # Generate debate ID
        debate_id = str(uuid.uuid4())
        
        # Determine participants (simplified agent selection)
        participants = request.preferred_agents or ["quantitative_analyst", "market_intelligence"]
        
        # Create database record using your existing patterns
        debate = crud.create_debate(
            db=db,
            user_id=current_user.id,
            query=request.topic,  # Maps to your 'query' field
            description=f"Multi-agent debate on: {request.topic}",
            urgency_level=request.urgency,
            max_rounds=3
        )
        
        
        # Start debate processing in background (simplified)
        background_tasks.add_task(process_debate_background, debate_id, db)
        
        return DebateResponse(
            debate_id=str(debate.id),  # Use the actual UUID
            topic=debate.query,
            status=debate.status.value,  # Convert enum to string
            created_at=debate.created_at.isoformat(),
            participants=request.preferred_agents or [],
            current_stage=debate.current_stage.value
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to create debate: {str(e)}"
        )

@router.get("/debates/{debate_id}", response_model=DebateResponse, tags=["Debates"])
async def get_debate_status(
    debate_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    📊 Get Debate Status
    
    Check the current status of a debate.
    """
    
    try:
        # Query debate from database
        debate = db.query(models.AgentDebate).filter(
            models.AgentDebate.debate_id == debate_id,
            models.AgentDebate.user_id == current_user.id
        ).first()
        
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        # Parse participants from JSON
        participants = json.loads(debate.participants) if debate.participants else []
        
        return DebateResponse(
            debate_id=debate.debate_id,
            topic=debate.topic,
            status=debate.status,
            created_at=debate.created_at.isoformat(),
            participants=participants,
            current_stage=debate.current_stage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get debate status: {str(e)}"
        )

@router.get("/debates", response_model=List[DebateResponse], tags=["Debates"])
async def get_user_debates(
    limit: int = 10,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    📋 Get User's Debates
    
    List debates for the current user.
    """
    
    try:
        # Query user's debates
        query = db.query(models.AgentDebate).filter(
            models.AgentDebate.user_id == current_user.id
        )
        
        if status_filter:
            query = query.filter(models.AgentDebate.status == status_filter)
        
        debates = query.order_by(
            models.AgentDebate.created_at.desc()
        ).limit(limit).all()
        
        # Convert to response format
        debate_responses = []
        for debate in debates:
            participants = json.loads(debate.participants) if debate.participants else []
            
            debate_responses.append(DebateResponse(
                debate_id=debate.debate_id,
                topic=debate.topic,
                status=debate.status,
                created_at=debate.created_at.isoformat(),
                participants=participants,
                current_stage=debate.current_stage
            ))
        
        return debate_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get debates: {str(e)}"
        )

@router.get("/debates/{debate_id}/results", tags=["Debates"])
async def get_debate_results(
    debate_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    🏆 Get Debate Results
    
    Get the final results of a completed debate.
    """
    
    try:
        # Get debate from database
        debate = db.query(models.AgentDebate).filter(
            models.AgentDebate.debate_id == debate_id,
            models.AgentDebate.user_id == current_user.id
        ).first()
        
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        if debate.status != "completed":
            raise HTTPException(status_code=400, detail="Debate not yet completed")
        
        # Get results from database
        results = {
            "debate_id": debate.debate_id,
            "topic": debate.topic,
            "status": debate.status,
            "final_recommendation": json.loads(debate.results) if debate.results else None,
            "confidence_score": debate.confidence_score,
            "participants": json.loads(debate.participants) if debate.participants else [],
            "completed_at": debate.completed_at.isoformat() if debate.completed_at else None,
            "summary": debate.summary or "Debate completed successfully"
        }
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get debate results: {str(e)}"
        )

@router.delete("/debates/{debate_id}", tags=["Debates"])
async def cancel_debate(
    debate_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    ❌ Cancel Debate
    
    Cancel an active debate.
    """
    
    try:
        # Find debate
        debate = db.query(models.AgentDebate).filter(
            models.AgentDebate.debate_id == debate_id,
            models.AgentDebate.user_id == current_user.id
        ).first()
        
        if not debate:
            raise HTTPException(status_code=404, detail="Debate not found")
        
        if debate.status in ["completed", "cancelled"]:
            raise HTTPException(status_code=400, detail="Debate already finished")
        
        # Update status
        debate.status = "cancelled"
        debate.completed_at = datetime.utcnow()
        debate.summary = "Debate cancelled by user"
        
        db.commit()
        
        return {"message": "Debate cancelled successfully", "debate_id": debate_id}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to cancel debate: {str(e)}"
        )

# ============================================================================
# BACKGROUND PROCESSING (simplified)
# ============================================================================

async def process_debate_background(debate_id: str, db: Session):
    """
    Background task to process debate (simplified implementation)
    
    In the full implementation, this would:
    1. Initialize agents
    2. Conduct debate rounds
    3. Build consensus
    4. Save results
    """
    
    try:
        print(f"🎭 Processing debate {debate_id}")
        
        # Simulate debate processing
        import asyncio
        await asyncio.sleep(2)  # Simulate some processing time
        
        # Update debate status in database
        debate = db.query(models.AgentDebate).filter(
            models.AgentDebate.debate_id == debate_id
        ).first()
        
        if debate:
            # Simulate agent positions and consensus
            mock_results = {
                "recommendation": "Consider moderate risk reduction based on current volatility",
                "consensus_confidence": 0.78,
                "agent_positions": [
                    {
                        "agent": "quantitative_analyst",
                        "position": "Reduce risk by 15% given elevated VaR metrics",
                        "confidence": 0.85
                    },
                    {
                        "agent": "market_intelligence", 
                        "position": "Maintain position but hedge with protective puts",
                        "confidence": 0.71
                    }
                ],
                "implementation_priority": "medium"
            }
            
            debate.status = "completed"
            debate.current_stage = "consensus_reached"
            debate.results = json.dumps(mock_results)
            debate.confidence_score = 0.78
            debate.completed_at = datetime.utcnow()
            debate.summary = "Multi-agent consensus reached on risk management approach"
            
            db.commit()
            
            print(f"✅ Debate {debate_id} completed successfully")
        
    except Exception as e:
        print(f"❌ Error processing debate {debate_id}: {str(e)}")
        
        # Update debate status to error
        if 'debate' in locals() and debate:
            debate.status = "error"
            debate.summary = f"Processing error: {str(e)}"
            db.commit()

# ============================================================================
# HELPER ENDPOINTS (optional)
# ============================================================================

@router.get("/debates/templates", tags=["Debates"])
async def get_debate_templates():
    """Get predefined debate templates"""
    
    templates = {
        "risk_assessment": {
            "name": "Portfolio Risk Assessment",
            "description": "Multi-agent risk analysis with different perspectives",
            "suggested_agents": ["quantitative_analyst", "market_intelligence"],
            "example_topic": "Should I reduce my portfolio risk given current market conditions?"
        },
        "rebalancing_decision": {
            "name": "Rebalancing Strategy",
            "description": "Debate on optimal portfolio rebalancing approach",
            "suggested_agents": ["quantitative_analyst", "tax_strategist"],
            "example_topic": "What's the optimal rebalancing strategy for my portfolio?"
        },
        "market_timing": {
            "name": "Market Timing Analysis", 
            "description": "Multi-agent analysis on market entry/exit timing",
            "suggested_agents": ["market_intelligence", "quantitative_analyst"],
            "example_topic": "Should I increase equity exposure given current market trends?"
        }
    }
    
    return {"templates": templates}

@router.get("/debates/agents", tags=["Debates"])
async def get_available_agents():
    """Get list of available agents for debates"""
    
    agents = {
        "quantitative_analyst": {
            "name": "Quantitative Analyst",
            "description": "Statistical analysis and risk modeling expert",
            "specialties": ["risk_analysis", "statistical_modeling", "portfolio_optimization"]
        },
        "market_intelligence": {
            "name": "Market Intelligence",
            "description": "Market trends and timing analysis expert", 
            "specialties": ["market_timing", "trend_analysis", "sentiment_analysis"]
        },
        "tax_strategist": {
            "name": "Tax Strategist",
            "description": "Tax optimization and efficiency expert",
            "specialties": ["tax_optimization", "tax_loss_harvesting", "asset_location"]
        },
        "options_analyst": {
            "name": "Options Analyst",
            "description": "Options strategies and volatility expert",
            "specialties": ["options_strategies", "volatility_analysis", "hedging"]
        }
    }
    
    return {"available_agents": agents}

# ============================================================================
# EXPORT FOR MAIN APPLICATION
# ============================================================================

__all__ = ["router"]