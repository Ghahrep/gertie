# workflows/orchestrator.py
"""
Consolidated Orchestrator - Intelligent routing between direct execution and MCP workflows
"""

import uuid
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import logging
from enum import Enum

from sqlalchemy.orm import Session
from db import crud
from db.models import WorkflowSessionDB, WorkflowStepDB
from api.schemas import QueryRequest, WorkflowResponse

from workflows.mcp_integration import MCPServerClient
from mcp.schemas import JobRequest, JobStatus
from agents.direct_agents import (
    PortfolioAnalysisAgent,
    RiskAssessmentAgent,
    StockScreeningAgent,
    MarketAnalysisAgent
)

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    DIRECT = "direct"
    MCP_WORKFLOW = "mcp_workflow"
    HYBRID = "hybrid"

class QueryComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ConsolidatedOrchestrator:
    """
    Intelligent orchestrator that routes queries between direct agents and MCP workflows
    based on complexity, performance history, and optimization criteria.
    """
    
    def __init__(self, db: Session, mcp_server_url: str = "http://localhost:8001"):
        self.db = db
        self.mcp_server = MCPServerClient(mcp_server_url)
        self.direct_agents = self._initialize_direct_agents()
        
        # Decision thresholds
        self.complexity_thresholds = {
            "low": 0.3,
            "medium": 0.7
        }
        
        # Performance tracking
        self.performance_window_days = 30
        self.min_samples_for_routing = 5
        
    def _initialize_direct_agents(self) -> Dict[str, Any]:
        """Initialize direct execution agents"""
        return {
            "portfolio_analysis": PortfolioAnalysisAgent(),
            "risk_assessment": RiskAssessmentAgent(),
            "stock_screening": StockScreeningAgent(),
            "market_analysis": MarketAnalysisAgent()
        }
    
    async def execute_query(
        self, 
        user_id: int, 
        query_request: QueryRequest
    ) -> WorkflowResponse:
        """
        Main entry point for query execution with intelligent routing
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        try:
            logger.info(f"Processing query for user {user_id}: {query_request.query[:100]}...")
            
            # Step 1: Analyze query and determine execution strategy
            analysis = await self._analyze_query(query_request.query)
            
            # Step 2: Create workflow session
            workflow_session = crud.create_workflow_session(
                db=self.db,
                session_id=session_id,
                user_id=user_id,
                query=query_request.query,
                workflow_type=analysis["workflow_type"],
                complexity_score=analysis["complexity_score"],
                execution_mode=analysis["recommended_mode"].value
            )
            
            logger.info(f"Created workflow session {session_id} with mode: {analysis['recommended_mode'].value}")
            
            # Step 3: Execute using recommended approach
            if analysis["recommended_mode"] == ExecutionMode.DIRECT:
                result = await self._execute_direct(workflow_session, analysis)
            elif analysis["recommended_mode"] == ExecutionMode.MCP_WORKFLOW:
                result = await self._execute_mcp_workflow(workflow_session, analysis)
            else:  # HYBRID
                result = await self._execute_hybrid(workflow_session, analysis)
            
            # Step 4: Update session with final results
            crud.update_workflow_session_state(
                db=self.db,
                session_id=session_id,
                state="complete",
                result=result
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Query completed in {execution_time_ms}ms using {analysis['recommended_mode'].value}")
            
            return WorkflowResponse(
                session_id=session_id,
                status="success",
                execution_mode=analysis["recommended_mode"].value,
                execution_time_ms=execution_time_ms,
                result=result
            )
            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            
            # Log error to database
            crud.add_workflow_session_error(
                db=self.db,
                session_id=session_id,
                error_message=str(e)
            )
            
            crud.update_workflow_session_state(
                db=self.db,
                session_id=session_id,
                state="error"
            )
            
            return WorkflowResponse(
                session_id=session_id,
                status="error",
                error_message=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine complexity, workflow type, and recommended execution mode
        """
        # Query complexity analysis
        complexity_score = self._calculate_complexity_score(query)
        complexity_level = self._determine_complexity_level(complexity_score)
        workflow_type = self._determine_workflow_type(query)
        
        # Add debugging output
        logger.info(f"Query analysis: complexity_score={complexity_score}, level={complexity_level.value}, workflow_type={workflow_type}")
        
        # Performance-based routing decision
        recommended_mode = await self._determine_execution_mode(
            workflow_type, complexity_level, complexity_score
        )
        
        logger.info(f"Routing decision: {recommended_mode.value}")
        
        return {
            "complexity_score": complexity_score,
            "complexity_level": complexity_level,
            "workflow_type": workflow_type,
            "recommended_mode": recommended_mode,
            "reasoning": self._generate_routing_reasoning(
                workflow_type, complexity_level, recommended_mode
            )
        }
    
    def _calculate_complexity_score(self, query: str) -> float:
        """Enhanced complexity scoring"""
        score = 0.0
        query_lower = query.lower()
        
        # Length factor (increased)
        if len(query) > 150:
            score += 0.3  # Increased from 0.2
        elif len(query) > 80:
            score += 0.2  # Increased from 0.1
        
        # Multi-step indicators (increased weight)
        multi_step_words = ['and', 'then', 'also', 'additionally', 'furthermore', 'compare', 'analyze', 'comprehensive', 'suggest', 'optimization', 'rebalancing']
        multi_step_matches = [word for word in multi_step_words if word in query_lower]
        multi_step_score = min(0.4, len(multi_step_matches) * 0.08)  # Increased
        score += multi_step_score
        logger.debug(f"Multi-step words found: {multi_step_matches}, score: +{multi_step_score}, total={score}")
        
        # Technical complexity indicators (increased weight)
        complex_terms = [
            'volatility', 'correlation', 'regression', 'optimization', 'backtesting',
            'monte carlo', 'var', 'sharpe', 'beta', 'alpha', 'risk-adjusted',
            'portfolio theory', 'capm', 'efficient frontier', 'factor model',
            'rebalancing', 'tax optimization', 'comprehensive', 'statistical'
        ]
        complex_matches = [term for term in complex_terms if term in query_lower]
        complex_score = min(0.4, len(complex_matches) * 0.08)  # Increased
        score += complex_score
        logger.debug(f"Complex terms found: {complex_matches}, score: +{complex_score}, total={score}")
        
        final_score = min(1.0, score)
        logger.info(f"Final complexity score: {final_score} for query: '{query[:50]}...'")
        return final_score
    
    def _determine_complexity_level(self, score: float) -> QueryComplexity:
        """Convert complexity score to level"""
        if score < self.complexity_thresholds["low"]:
            return QueryComplexity.LOW
        elif score < self.complexity_thresholds["medium"]:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.HIGH
    
    def _determine_workflow_type(self, query: str) -> str:
        """Determine the type of workflow based on query content"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['screen', 'find stocks', 'filter', 'criteria']):
            return "stock_screening"
        elif any(word in query_lower for word in ['risk', 'volatility', 'var', 'drawdown']):
            return "risk_analysis"
        elif any(word in query_lower for word in ['portfolio', 'allocation', 'optimize', 'rebalance']):
            return "portfolio_analysis"
        elif any(word in query_lower for word in ['market', 'trend', 'sentiment', 'economic']):
            return "market_analysis"
        elif any(word in query_lower for word in ['backtest', 'strategy', 'performance']):
            return "strategy_backtesting"
        else:
            return "general_analysis"
    
    async def _determine_execution_mode(
        self, 
        workflow_type: str, 
        complexity_level: QueryComplexity,
        complexity_score: float
    ) -> ExecutionMode:
        """
        Determine optimal execution mode - Enhanced with debugging
        """
        logger.info(f"Determining execution mode: workflow_type={workflow_type}, complexity_level={complexity_level.value}, score={complexity_score}")
        
        # Get historical performance for both modes
        direct_performance = crud.get_agent_performance_stats(
            db=self.db,
            agent_id=f"{workflow_type}_direct",
            capability=workflow_type,
            execution_mode="direct",
            days=self.performance_window_days
        )
        
        mcp_performance = crud.get_agent_performance_stats(
            db=self.db,
            agent_id=f"{workflow_type}_mcp",
            capability=workflow_type,
            execution_mode="mcp_workflow",
            days=self.performance_window_days
        )
        
        logger.info(f"Performance data - Direct executions: {direct_performance['total_executions']}, MCP executions: {mcp_performance['total_executions']}")
        
        # Decision logic with debugging
        if complexity_level == QueryComplexity.LOW:
            logger.info("LOW complexity - checking if MCP significantly outperforms direct")
            if (mcp_performance["total_executions"] >= self.min_samples_for_routing and
                mcp_performance["success_rate"] > direct_performance["success_rate"] + 0.2):
                logger.info("MCP significantly outperforms - choosing MCP_WORKFLOW")
                return ExecutionMode.MCP_WORKFLOW
            logger.info("Defaulting to DIRECT for low complexity")
            return ExecutionMode.DIRECT
            
        elif complexity_level == QueryComplexity.MEDIUM:
            logger.info("MEDIUM complexity - comparing performance data")
            if (direct_performance["total_executions"] >= self.min_samples_for_routing and
                mcp_performance["total_executions"] >= self.min_samples_for_routing):
                
                # Compare performance scores
                direct_score = self._calculate_performance_score(direct_performance)
                mcp_score = self._calculate_performance_score(mcp_performance)
                
                logger.info(f"Performance scores - Direct: {direct_score:.3f}, MCP: {mcp_score:.3f}")
                
                if abs(direct_score - mcp_score) < 0.1:
                    logger.info("Very close performance - choosing HYBRID")
                    return ExecutionMode.HYBRID
                elif direct_score > mcp_score:
                    logger.info("Direct outperforms - choosing DIRECT")
                    return ExecutionMode.DIRECT
                else:
                    logger.info("MCP outperforms - choosing MCP_WORKFLOW")
                    return ExecutionMode.MCP_WORKFLOW
            else:
                # Insufficient data, default based on complexity
                if complexity_score > 0.5:
                    logger.info("Insufficient data, high complexity score - choosing MCP_WORKFLOW")
                    return ExecutionMode.MCP_WORKFLOW
                else:
                    logger.info("Insufficient data, low complexity score - choosing DIRECT")
                    return ExecutionMode.DIRECT
        
        else:  # HIGH complexity
            logger.info("HIGH complexity - checking performance data")
            logger.info(f"Direct executions: {direct_performance['total_executions']}, MCP executions: {mcp_performance['total_executions']}")
            logger.info(f"Min samples required: {self.min_samples_for_routing}")
            
            # For high complexity, require substantial evidence that direct is better
            if (direct_performance["total_executions"] >= self.min_samples_for_routing and
                mcp_performance["total_executions"] >= self.min_samples_for_routing and
                direct_performance["success_rate"] > mcp_performance["success_rate"] + 0.3):
                logger.info("Direct significantly outperforms with sufficient MCP data - choosing DIRECT")
                return ExecutionMode.DIRECT
            logger.info("High complexity - defaulting to MCP_WORKFLOW")
            return ExecutionMode.MCP_WORKFLOW
    
    def _calculate_performance_score(self, performance: Dict[str, Any]) -> float:
        """Calculate weighted performance score"""
        if performance["total_executions"] == 0:
            return 0.0
        
        success_weight = 0.5
        speed_weight = 0.3
        confidence_weight = 0.2
        
        success_score = performance["success_rate"]
        speed_score = max(0, (10000 - performance["average_execution_time_ms"]) / 10000)
        confidence_score = performance["average_confidence_score"]
        
        return (success_score * success_weight + 
                speed_score * speed_weight + 
                confidence_score * confidence_weight)
    
    def _generate_routing_reasoning(
        self, 
        workflow_type: str, 
        complexity_level: QueryComplexity,
        execution_mode: ExecutionMode
    ) -> str:
        """Generate human-readable reasoning for routing decision"""
        return f"Query classified as {workflow_type} with {complexity_level.value} complexity. " \
               f"Routed to {execution_mode.value} based on performance optimization."
    
    async def _execute_direct(
        self, 
        session: WorkflowSessionDB, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute using direct agents"""
        workflow_type = analysis["workflow_type"]
        
        # Get appropriate direct agent
        agent_key = self._map_workflow_to_agent(workflow_type)
        if agent_key not in self.direct_agents:
            raise ValueError(f"No direct agent available for workflow type: {workflow_type}")
        
        agent = self.direct_agents[agent_key]
        
        # Create workflow step
        step_id = str(uuid.uuid4())
        step = crud.create_workflow_step(
            db=self.db,
            step_id=step_id,
            session_id=session.session_id,
            step_number=1,
            step_name="direct_execution",
            agent_id=f"{workflow_type}_direct",
            capability=workflow_type
        )
        
        try:
            # Update step status
            crud.update_workflow_step_status(
                db=self.db,
                step_id=step_id,
                status="running"
            )
            
            # Execute agent
            start_time = time.time()
            result = await agent.execute(session.query)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record performance
            crud.record_agent_performance(
                db=self.db,
                agent_id=f"{workflow_type}_direct",
                capability=workflow_type,
                execution_time_ms=execution_time_ms,
                success=True,
                execution_mode="direct",
                confidence_score=result.get("confidence", 0.8),
                query_complexity=analysis["complexity_score"],
                user_id=session.user_id,
                session_id=session.session_id,
                workflow_type=workflow_type
            )
            
            # Update step as completed
            crud.update_workflow_step_status(
                db=self.db,
                step_id=step_id,
                status="completed",
                result=result,
                confidence_score=result.get("confidence", 0.8),
                execution_time_ms=execution_time_ms
            )
            
            return result
            
        except Exception as e:
            # Record failure
            crud.record_agent_performance(
                db=self.db,
                agent_id=f"{workflow_type}_direct",
                capability=workflow_type,
                execution_time_ms=int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0,
                success=False,
                execution_mode="direct",
                query_complexity=analysis["complexity_score"],
                user_id=session.user_id,
                session_id=session.session_id,
                workflow_type=workflow_type
            )
            
            crud.update_workflow_step_status(
                db=self.db,
                step_id=step_id,
                status="failed",
                error_message=str(e)
            )
            
            raise
    
    async def _execute_mcp_workflow(self, session: WorkflowSessionDB, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute using your existing MCP system"""
        
        # Submit to your MCP server instead of mock client
        mcp_request = JobRequest(
            query=session.query,
            context={
                "workflow_type": analysis["workflow_type"],
                "complexity_score": analysis["complexity_score"],
                "session_id": session.session_id
            },
            priority=7,
            required_capabilities=self._map_workflow_to_capabilities(analysis["workflow_type"])
        )
        
        # Use your MCP server's submit_job endpoint
        response = await self.mcp_server.submit_job(mcp_request)
        
        # Track the MCP job
        crud.update_workflow_session_mcp_job(
            db=self.db,
            session_id=session.session_id,
            mcp_job_id=response.job_id
        )
        
        # Wait for completion and return results
        return await self._wait_for_mcp_completion(response.job_id)
        
    async def _execute_hybrid(
        self, 
        session: WorkflowSessionDB, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute using hybrid approach (try direct first, fallback to MCP)"""
        try:
            logger.info(f"Attempting direct execution for session {session.session_id}")
            result = await self._execute_direct(session, analysis)
            
            # If direct execution succeeds and confidence is high enough, return it
            if result.get("confidence", 0) > 0.7:
                return result
            else:
                logger.info(f"Direct execution low confidence, trying MCP for session {session.session_id}")
                # Low confidence, try MCP as well and compare
                try:
                    mcp_result = await self._execute_mcp_workflow(session, analysis)
                    
                    # Return higher confidence result
                    if mcp_result.get("confidence", 0) > result.get("confidence", 0):
                        return mcp_result
                    else:
                        return result
                        
                except Exception:
                    logger.warning(f"MCP fallback failed for session {session.session_id}, using direct result")
                    return result
        
        except Exception as e:
            logger.info(f"Direct execution failed for session {session.session_id}, trying MCP fallback")
            # Direct failed, try MCP
            return await self._execute_mcp_workflow(session, analysis)
    
    def _map_workflow_to_agent(self, workflow_type: str) -> str:
        """Map workflow type to direct agent key"""
        mapping = {
            "stock_screening": "stock_screening",
            "risk_analysis": "risk_assessment",
            "portfolio_analysis": "portfolio_analysis",
            "market_analysis": "market_analysis"
        }
        return mapping.get(workflow_type, "portfolio_analysis")  # Default fallback
    
    def _map_workflow_to_capabilities(self, workflow_type: str) -> List[str]:
        """Map workflow type to MCP capability requirements"""
        capability_mapping = {
            "stock_screening": ["market_data_fetch", "portfolio_analysis", "result_synthesis"],
            "risk_analysis": ["portfolio_data_fetch", "risk_analysis", "result_synthesis"],
            "portfolio_analysis": ["portfolio_data_fetch", "portfolio_analysis", "result_synthesis"],
            "market_analysis": ["market_data_fetch", "market_intelligence", "result_synthesis"],
            "strategy_backtesting": ["portfolio_data_fetch", "strategy_rebalancing", "result_synthesis"],
            "general_analysis": ["query_interpretation", "general_analysis", "result_synthesis"]
        }
        return capability_mapping.get(workflow_type, ["general_analysis", "result_synthesis"])

    async def _wait_for_mcp_completion(self, job_id: str) -> Dict[str, Any]:
        """Wait for MCP job completion and return results"""
        max_wait_time = 300  # 5 minutes
        poll_interval = 2    # Check every 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            try:
                # Get job status from MCP server
                job_response = await self.mcp_server.get_job_status(job_id)
                
                if job_response.status == JobStatus.COMPLETED:
                    return job_response.result or {}
                elif job_response.status == JobStatus.FAILED:
                    error_msg = job_response.error_details.get("error", "MCP job failed") if job_response.error_details else "MCP job failed"
                    raise Exception(f"MCP job failed: {error_msg}")
                elif job_response.status in [JobStatus.CANCELLED]:
                    raise Exception(f"MCP job was cancelled")
                
                # Update workflow session with progress
                if hasattr(job_response, 'progress') and job_response.progress is not None:
                    logger.info(f"MCP job {job_id} progress: {job_response.progress}%")
                
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
                
            except Exception as e:
                if "failed" in str(e) or "cancelled" in str(e):
                    raise  # Re-raise job failures
                # For connection/communication errors, log and continue polling
                logger.warning(f"Error polling MCP job {job_id}: {str(e)}")
                await asyncio.sleep(poll_interval)
                elapsed_time += poll_interval
        
        # Timeout
        raise Exception(f"MCP job {job_id} timed out after {max_wait_time} seconds")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of both orchestrator and MCP server"""
        try:
            mcp_health = await self.mcp_server.get_health()
            return {
                "orchestrator_status": "healthy",
                "mcp_server_status": mcp_health.get("status", "unknown"),
                "registered_agents": mcp_health.get("registered_agents", 0),
                "active_jobs": mcp_health.get("active_jobs", 0)
            }
        except Exception as e:
            return {
                "orchestrator_status": "healthy", 
                "mcp_server_status": "unhealthy",
                "error": str(e)
            }
    
    async def get_workflow_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow session"""
        return crud.get_workflow_session_summary(self.db, session_id)
    
    async def get_performance_analytics(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Get orchestrator performance analytics"""
        return crud.get_workflow_analytics(self.db, user_id=user_id)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics"""
        return crud.get_system_health_metrics(self.db)
        
    async def close(self):
        """Clean up resources"""
        if hasattr(self, 'mcp_server'):
            await self.mcp_server.close()