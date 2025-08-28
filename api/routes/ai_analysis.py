# api/routes/ai_analysis.py
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional
import aiohttp
import logging
from datetime import datetime

from api.schemas import AIAnalysisRequest, AutonomousAnalysisResponse
from mcp.schemas import JobRequest, JobResponse, JobStatus
from api.routes.auth import get_current_user
from services.mcp_client import MCPClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ai-analysis", tags=["AI Analysis"])

# Initialize MCP client
mcp_client = MCPClient(base_url="http://localhost:8001")

@router.post("/portfolios/{portfolio_id}/ai-analysis", response_model=Dict)
async def get_autonomous_ai_analysis(
    portfolio_id: str,
    request: AIAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """
    Submit an autonomous AI analysis request for a portfolio.
    The MCP will orchestrate multiple agents to provide comprehensive analysis.
    """
    try:
        # Validate portfolio access
        if not await _validate_portfolio_access(portfolio_id, current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        # Create job request for MCP
        job_request = JobRequest(
            query=request.query,
            context={
                "portfolio_id": portfolio_id,
                "user_id": current_user.user_id,
                "analysis_depth": request.analysis_depth,
                "include_recommendations": request.include_recommendations,
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=7,  # High priority for user-initiated analysis
            timeout_seconds=300,  # 5 minute timeout
            required_capabilities=_determine_required_capabilities(request)
        )
        
        # Submit job to MCP
        job_response = await mcp_client.submit_job(job_request)
        
        if job_response.status == JobStatus.FAILED:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to start analysis: {job_response.message}"
            )
        
        logger.info(f"Started AI analysis job {job_response.job_id} for portfolio {portfolio_id}")
        
        # Store job reference for user tracking
        background_tasks.add_task(
            _track_analysis_job, 
            job_response.job_id, 
            portfolio_id, 
            current_user.user_id
        )
        
        return {
            "job_id": job_response.job_id,
            "status": job_response.status.value,
            "message": "AI analysis started successfully",
            "estimated_completion": job_response.estimated_completion_time,
            "analysis_type": request.analysis_depth,
            "agents_assigned": job_response.agents_involved or [],
            "tracking_url": f"/api/ai-analysis/jobs/{job_response.job_id}"
        }
        
    except Exception as e:
        logger.error(f"Error starting AI analysis for portfolio {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start AI analysis")

@router.get("/jobs/{job_id}", response_model=AutonomousAnalysisResponse)
async def get_analysis_job_status(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get the status and results of an AI analysis job.
    """
    try:
        # Get job status from MCP
        job_response = await mcp_client.get_job_status(job_id)
        
        if not job_response:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        
        # Verify user has access to this job
        if not await _validate_job_access(job_id, current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to analysis job")
        
        # Format response for client
        response = AutonomousAnalysisResponse(
            job_id=job_id,
            status=job_response.status.value,
            progress=job_response.progress or 0.0,
            workflow_used=_extract_workflow_info(job_response),
            agents_consulted=job_response.agents_involved or [],
            confidence_score=_extract_confidence_score(job_response.result),
            result=job_response.result,
            message=job_response.message,
            completed_at=None if job_response.status != JobStatus.COMPLETED else datetime.utcnow().isoformat()
        )
        
        logger.info(f"Retrieved analysis job {job_id} status: {job_response.status}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving analysis job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis status")

@router.get("/portfolios/{portfolio_id}/recent-analyses")
async def get_recent_analyses(
    portfolio_id: str,
    limit: int = 10,
    current_user = Depends(get_current_user)
):
    """
    Get recent AI analyses for a portfolio.
    """
    try:
        # Validate portfolio access
        if not await _validate_portfolio_access(portfolio_id, current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        # Get recent analyses from database
        recent_analyses = await _get_recent_analyses(portfolio_id, limit)
        
        return {
            "portfolio_id": portfolio_id,
            "analyses": recent_analyses,
            "total_count": len(recent_analyses)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving recent analyses for portfolio {portfolio_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve recent analyses")

@router.post("/portfolios/{portfolio_id}/analysis-templates/{template_id}")
async def run_analysis_template(
    portfolio_id: str,
    template_id: str,
    parameters: Optional[Dict] = None,
    current_user = Depends(get_current_user)
):
    """
    Run a predefined analysis template with custom parameters.
    """
    try:
        # Validate portfolio access
        if not await _validate_portfolio_access(portfolio_id, current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to portfolio")
        
        # Get template configuration
        template = await _get_analysis_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Analysis template not found")
        
        # Create job request from template
        job_request = JobRequest(
            query=template["query_template"].format(**(parameters or {})),
            context={
                "portfolio_id": portfolio_id,
                "user_id": current_user.user_id,
                "template_id": template_id,
                "template_parameters": parameters or {},
                "timestamp": datetime.utcnow().isoformat()
            },
            priority=template.get("priority", 5),
            required_capabilities=template["required_capabilities"]
        )
        
        # Submit job to MCP
        job_response = await mcp_client.submit_job(job_request)
        
        return {
            "job_id": job_response.job_id,
            "template_name": template["name"],
            "status": job_response.status.value,
            "estimated_completion": job_response.estimated_completion_time
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running analysis template {template_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to run analysis template")

@router.delete("/jobs/{job_id}")
async def cancel_analysis_job(
    job_id: str,
    current_user = Depends(get_current_user)
):
    """
    Cancel a running AI analysis job.
    """
    try:
        # Verify user has access to this job
        if not await _validate_job_access(job_id, current_user.user_id):
            raise HTTPException(status_code=403, detail="Access denied to analysis job")
        
        # Cancel job via MCP
        success = await mcp_client.cancel_job(job_id)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to cancel analysis job")
        
        return {"message": "Analysis job cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling analysis job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to cancel analysis job")

# Helper functions

def _determine_required_capabilities(request: AIAnalysisRequest) -> List[str]:
    """Determine what agent capabilities are needed based on the request"""
    capabilities = ["portfolio_analysis", "query_interpretation"]
    query_lower = request.query.lower()
    
    # Map query content to required capabilities
    if any(word in query_lower for word in ["risk", "volatility", "drawdown", "var"]):
        capabilities.extend(["risk_analysis", "quantitative_analysis"])
    
    if any(word in query_lower for word in ["tax", "harvest", "optimization", "savings"]):
        capabilities.append("tax_optimization")
    
    if any(word in query_lower for word in ["rebalance", "allocation", "weight"]):
        capabilities.append("strategy_rebalancing")
    
    if any(word in query_lower for word in ["market", "timing", "economic", "sentiment"]):
        capabilities.append("market_intelligence")
    
    if any(word in query_lower for word in ["options", "derivatives", "volatility"]):
        capabilities.append("options_analysis")
    
    if request.analysis_depth == "comprehensive":
        capabilities.extend([
            "risk_analysis", 
            "market_intelligence", 
            "strategy_rebalancing"
        ])
    
    return list(set(capabilities))  # Remove duplicates

async def _validate_portfolio_access(portfolio_id: str, user_id: str) -> bool:
    """Validate that user has access to the specified portfolio"""
    # Mock validation - replace with actual database check
    try:
        # In production, query database to verify portfolio ownership/access
        return True  # Simplified for now
    except Exception as e:
        logger.error(f"Error validating portfolio access: {str(e)}")
        return False

async def _validate_job_access(job_id: str, user_id: str) -> bool:
    """Validate that user has access to the specified analysis job"""
    # Mock validation - replace with actual database check
    try:
        # In production, query database to verify job ownership
        return True  # Simplified for now
    except Exception as e:
        logger.error(f"Error validating job access: {str(e)}")
        return False

async def _track_analysis_job(job_id: str, portfolio_id: str, user_id: str):
    """Track analysis job in database for user reference"""
    try:
        # Store job tracking information in database
        job_record = {
            "job_id": job_id,
            "portfolio_id": portfolio_id,
            "user_id": user_id,
            "started_at": datetime.utcnow(),
            "status": "running"
        }
        # await database.insert("analysis_jobs", job_record)
        logger.info(f"Tracking analysis job {job_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Error tracking analysis job {job_id}: {str(e)}")

def _extract_workflow_info(job_response: JobResponse) -> List[str]:
    """Extract workflow information from job response"""
    if not job_response.result:
        return []
    
    workflow_summary = job_response.result.get("workflow_summary", {})
    return workflow_summary.get("workflow_steps", [])

def _extract_confidence_score(result: Optional[Dict]) -> float:
    """Extract confidence score from analysis result"""
    if not result:
        return 0.0
    
    return result.get("confidence_score", 0.8)

async def _get_recent_analyses(portfolio_id: str, limit: int) -> List[Dict]:
    """Get recent AI analyses for a portfolio"""
    # Mock implementation - replace with actual database query
    return [
        {
            "job_id": "job_123",
            "query": "Analyze portfolio risk and suggest optimizations",
            "status": "completed",
            "confidence_score": 0.92,
            "started_at": "2024-01-15T14:00:00Z",
            "completed_at": "2024-01-15T14:02:30Z"
        },
        {
            "job_id": "job_124", 
            "query": "Tax loss harvesting opportunities",
            "status": "completed",
            "confidence_score": 0.88,
            "started_at": "2024-01-14T16:30:00Z",
            "completed_at": "2024-01-14T16:31:45Z"
        }
    ]

async def _get_analysis_template(template_id: str) -> Optional[Dict]:
    """Get analysis template configuration"""
    templates = {
        "comprehensive_analysis": {
            "name": "Comprehensive Portfolio Analysis",
            "query_template": "Provide a comprehensive analysis of portfolio including risk, performance, and optimization recommendations",
            "required_capabilities": ["portfolio_analysis", "risk_analysis", "strategy_rebalancing"],
            "priority": 8
        },
        "risk_assessment": {
            "name": "Risk Assessment",
            "query_template": "Analyze portfolio risk metrics and provide risk management recommendations",
            "required_capabilities": ["risk_analysis", "quantitative_analysis"],
            "priority": 7
        },
        "tax_optimization": {
            "name": "Tax Optimization Review",
            "query_template": "Review portfolio for tax optimization opportunities including tax-loss harvesting",
            "required_capabilities": ["tax_optimization", "portfolio_analysis"],
            "priority": 6
        }
    }
    
    return templates.get(template_id)