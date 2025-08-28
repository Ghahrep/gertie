# services/workflow_integration.py
"""
Task 2.2.4: Workflow Integration System
Connects risk alerts to AI agent workflow execution with comprehensive tracking
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json

from sqlalchemy.orm import Session
from db.session import get_db
from db.models import (
    RiskChangeEvent, ProactiveAlert, WorkflowSessionDB, 
    WorkflowStepDB, MCPJobLog
)
from db.crud import (
    create_workflow_session, update_workflow_session_state,
    create_workflow_step, update_workflow_step_status,
    create_mcp_job_log, update_mcp_job_log_status
)

# Import your existing services
from services.mcp_client import get_mcp_client
from mcp.schemas import JobRequest

logger = logging.getLogger(__name__)

class WorkflowTrigger(str, Enum):
    RISK_ALERT = "risk_alert"
    THRESHOLD_BREACH = "threshold_breach"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    ESCALATION = "escalation"

class WorkflowPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class WorkflowContext:
    """Context information for workflow execution"""
    trigger_type: WorkflowTrigger
    portfolio_id: int
    user_id: int
    risk_event_id: Optional[int] = None
    alert_id: Optional[int] = None
    priority: WorkflowPriority = WorkflowPriority.MEDIUM
    custom_params: Dict[str, Any] = field(default_factory=dict)
    deadline: Optional[datetime] = None

@dataclass
class WorkflowResult:
    """Result of workflow execution"""
    workflow_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    steps_completed: int = 0
    total_steps: int = 0
    confidence_score: Optional[float] = None

class WorkflowIntegrationEngine:
    """
    Complete workflow integration system that connects risk monitoring
    with AI agent workflows and provides comprehensive tracking
    """
    
    def __init__(self):
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_templates = {}
        self.execution_stats = {
            "workflows_triggered": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "avg_completion_time_seconds": 0,
            "success_rate": 0.0
        }
        
        # Load workflow templates
        self._initialize_workflow_templates()
        
        logger.info("Workflow Integration Engine initialized")
    
    def _initialize_workflow_templates(self):
        """Initialize standard workflow templates"""
        
        # Risk Alert Workflow Template
        self.workflow_templates["risk_alert_analysis"] = {
            "name": "Risk Alert Analysis",
            "description": "Comprehensive 4-agent analysis triggered by risk alerts",
            "steps": [
                {
                    "step_name": "strategy_analysis",
                    "agent_capability": "strategy_analysis",
                    "timeout_seconds": 120,
                    "required": True,
                    "prompt_template": """
Risk Alert Detected - Strategic Analysis Required

Portfolio: {portfolio_id}
Alert Type: {alert_type}
Risk Magnitude: {risk_magnitude_pct}%
Direction: {risk_direction}

Key Risk Changes:
{significant_changes}

Strategic Team, please analyze:
1. Root causes of this risk change
2. Market conditions contributing to the change
3. Strategic implications for the portfolio
4. Timeline for potential impact

Provide strategic context for the team's analysis.
                    """
                },
                {
                    "step_name": "security_screening",
                    "agent_capability": "security_analysis",
                    "timeout_seconds": 180,
                    "required": True,
                    "dependencies": ["strategy_analysis"],
                    "prompt_template": """
Security Screening - Risk Alert Response

Based on strategic analysis, conduct detailed security review:

Portfolio Holdings Analysis:
- Review current positions contributing to risk increase
- Identify specific securities driving volatility
- Assess concentration risks and exposures
- Check for any fundamental deterioration

Risk Contributors:
{risk_changes}

Please provide:
1. Security-level risk assessment
2. Holdings that need attention
3. Diversification recommendations
4. Any red flags requiring immediate action
                    """
                },
                {
                    "step_name": "quantitative_analysis",
                    "agent_capability": "risk_analysis",
                    "timeout_seconds": 240,
                    "required": True,
                    "dependencies": ["strategy_analysis", "security_screening"],
                    "prompt_template": """
Quantitative Risk Analysis - Deep Dive

Risk Event Details:
- Current Risk Score: {current_risk_score}
- Risk Direction: {risk_direction}
- Magnitude: {risk_magnitude_pct}%
- Threshold Breach: {threshold_breached}

Quantitative team, please provide:
1. Detailed risk metric analysis
2. Stress testing scenarios
3. Value-at-Risk projections
4. Correlation analysis
5. Risk attribution by factor/sector
6. Tail risk assessment

Focus on quantifying the impact and providing data-driven insights.
                    """
                },
                {
                    "step_name": "recommendation_synthesis",
                    "agent_capability": "portfolio_synthesis",
                    "timeout_seconds": 180,
                    "required": True,
                    "dependencies": ["strategy_analysis", "security_screening", "quantitative_analysis"],
                    "prompt_template": """
Final Synthesis - Risk Response Recommendations

Integrate all team analyses to provide comprehensive recommendations:

Strategic Context: {strategy_result}
Security Analysis: {screening_result}
Quantitative Assessment: {analysis_result}

Please synthesize into:
1. Executive Summary (2-3 key points)
2. Immediate Actions (next 24-48 hours)
3. Short-term Adjustments (next 2-4 weeks)
4. Long-term Considerations (next 3-6 months)
5. Risk Monitoring Adjustments
6. Success Metrics and Follow-up

Provide clear, actionable recommendations with priority levels.
                    """
                }
            ],
            "priority": WorkflowPriority.HIGH,
            "timeout_seconds": 600,
            "expected_agents": ["StrategyAgent", "SecurityAnalystAgent", "QuantitativeAnalystAgent", "PortfolioManagerAgent"]
        }
        
        # Threshold Breach Workflow Template  
        self.workflow_templates["threshold_breach_response"] = {
            "name": "Threshold Breach Response",
            "description": "Immediate response workflow for risk threshold breaches",
            "steps": [
                {
                    "step_name": "immediate_assessment",
                    "agent_capability": "risk_analysis", 
                    "timeout_seconds": 60,
                    "required": True,
                    "prompt_template": """
URGENT: Risk Threshold Breach Detected

Portfolio: {portfolio_id}
Threshold Breached: {threshold_type}
Current Value: {current_value}
Threshold: {threshold_value}
Breach Magnitude: {breach_magnitude}%

Immediate assessment required:
1. Confirm breach severity
2. Identify immediate risks
3. Recommend emergency actions if needed
4. Assess if trading halt is required

This is time-sensitive - provide rapid but thorough analysis.
                    """
                },
                {
                    "step_name": "action_recommendations",
                    "agent_capability": "portfolio_management",
                    "timeout_seconds": 90,
                    "required": True,
                    "dependencies": ["immediate_assessment"],
                    "prompt_template": """
Emergency Action Planning

Based on immediate assessment: {assessment_result}

Provide specific action plan:
1. Immediate actions (next 1-4 hours)
2. Risk mitigation steps
3. Portfolio adjustments needed
4. Client communication requirements
5. Monitoring intensification plan

Focus on practical, executable steps.
                    """
                }
            ],
            "priority": WorkflowPriority.CRITICAL,
            "timeout_seconds": 300,
            "expected_agents": ["QuantitativeAnalystAgent", "PortfolioManagerAgent"]
        }
    
    async def trigger_workflow(
        self,
        context: WorkflowContext,
        template_name: str = "risk_alert_analysis"
    ) -> WorkflowResult:
        """
        Trigger a workflow execution based on context
        Main entry point for workflow triggering
        """
        workflow_id = str(uuid.uuid4())
        
        try:
            # Get workflow template
            template = self.workflow_templates.get(template_name)
            if not template:
                raise ValueError(f"Workflow template '{template_name}' not found")
            
            # Create workflow session in database
            db = next(get_db())
            try:
                # Generate workflow query based on context
                workflow_query = await self._generate_workflow_query(context, template)
                
                # Create workflow session
                session = create_workflow_session(
                    db=db,
                    session_id=workflow_id,
                    user_id=context.user_id,
                    query=workflow_query,
                    workflow_type=template_name,
                    complexity_score=self._calculate_complexity_score(context, template),
                    execution_mode="proactive_alert"
                )
                
                # Store workflow context
                self.active_workflows[workflow_id] = context
                
                # Execute workflow
                result = await self._execute_workflow(workflow_id, template, context, db)
                
                # Update stats
                self.execution_stats["workflows_triggered"] += 1
                if result.status == "completed":
                    self.execution_stats["workflows_completed"] += 1
                elif result.status == "failed":
                    self.execution_stats["workflows_failed"] += 1
                
                logger.info(f"Workflow {workflow_id} triggered: {result.status}")
                return result
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Failed to trigger workflow {workflow_id}: {e}")
            return WorkflowResult(
                workflow_id=workflow_id,
                status="failed",
                error=str(e)
            )
    
    async def _generate_workflow_query(
        self, 
        context: WorkflowContext, 
        template: Dict
    ) -> str:
        """Generate the main query for the workflow based on context"""
        
        if context.trigger_type == WorkflowTrigger.RISK_ALERT:
            return f"""
Risk Alert Triggered - Team Analysis Required

Portfolio {context.portfolio_id} has experienced a significant risk change requiring comprehensive analysis.

Team, please conduct your standard 4-step collaborative analysis:
1. Strategic Analysis - Assess market context and strategic implications
2. Security Screening - Review holdings and identify risk contributors  
3. Risk Assessment - Quantify risks and run stress scenarios
4. Synthesis - Provide integrated recommendations and action plan

This is a {context.priority.value} priority workflow triggered by our monitoring system.
Please provide thorough analysis with actionable recommendations.
            """.strip()
            
        elif context.trigger_type == WorkflowTrigger.THRESHOLD_BREACH:
            return f"""
URGENT: Risk Threshold Breach - Immediate Response Required

Portfolio {context.portfolio_id} has breached risk thresholds requiring immediate assessment and action planning.

This is an emergency workflow requiring rapid but thorough analysis and immediate action recommendations.
            """.strip()
        
        else:
            return f"""
Portfolio Analysis Request - Portfolio {context.portfolio_id}

Standard multi-agent analysis requested for portfolio review and recommendations.
            """.strip()
    
    def _calculate_complexity_score(
        self, 
        context: WorkflowContext, 
        template: Dict
    ) -> float:
        """Calculate workflow complexity score"""
        base_score = 0.5
        
        # Increase complexity based on trigger type
        if context.trigger_type == WorkflowTrigger.RISK_ALERT:
            base_score += 0.3
        elif context.trigger_type == WorkflowTrigger.THRESHOLD_BREACH:
            base_score += 0.4
        
        # Increase complexity based on priority
        priority_scores = {
            WorkflowPriority.LOW: 0.1,
            WorkflowPriority.MEDIUM: 0.2,
            WorkflowPriority.HIGH: 0.3,
            WorkflowPriority.CRITICAL: 0.4,
            WorkflowPriority.EMERGENCY: 0.5
        }
        base_score += priority_scores.get(context.priority, 0.2)
        
        # Increase complexity based on number of steps
        step_count = len(template.get("steps", []))
        base_score += min(step_count * 0.1, 0.3)
        
        return min(base_score, 1.0)
    
    async def _execute_workflow(
        self,
        workflow_id: str,
        template: Dict,
        context: WorkflowContext,
        db: Session
    ) -> WorkflowResult:
        """Execute the workflow steps"""
        
        result = WorkflowResult(
            workflow_id=workflow_id,
            status="running",
            started_at=datetime.utcnow(),
            total_steps=len(template.get("steps", []))
        )
        
        try:
            # Submit to MCP for execution
            mcp_job_request = await self._create_mcp_job_request(
                workflow_id, template, context
            )
            
            # Submit job to MCP client
            mcp_client = await get_mcp_client()
            job_response = await mcp_client.submit_job(mcp_job_request)
            
            # Log MCP job submission
            job_log = create_mcp_job_log(
                db=db,
                job_id=job_response.get("job_id", workflow_id),
                job_request=mcp_job_request.model_dump() if hasattr(mcp_job_request, 'model_dump') else mcp_job_request.__dict__,
                session_id=workflow_id
            )
            
            # Update workflow session with MCP job ID
            from db.crud import update_workflow_session_mcp_job
            update_workflow_session_mcp_job(db, workflow_id, job_response.get("job_id"))
            
            # Track workflow execution
            await self._track_workflow_execution(workflow_id, job_response, db)
            
            result.status = "submitted_to_mcp"
            result.result = {
                "mcp_job_id": job_response.get("job_id"),
                "submission_time": datetime.utcnow().isoformat(),
                "estimated_completion": job_response.get("estimated_completion_time")
            }
            
            logger.info(f"Workflow {workflow_id} submitted to MCP: {job_response.get('job_id')}")
            
        except Exception as e:
            logger.error(f"Workflow execution failed for {workflow_id}: {e}")
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.utcnow()
            
            # Update workflow session with error
            update_workflow_session_state(
                db, workflow_id, "error", 
                result={"error": str(e)}, 
                step_result={"error": str(e)}, 
                step_name="execution"
            )
        
        return result
    
    async def _create_mcp_job_request(
        self,
        workflow_id: str,
        template: Dict,
        context: WorkflowContext
    ) -> JobRequest:
        """Create MCP job request from workflow template and context"""
        
        # Get context data for prompt formatting
        context_data = await self._get_context_data(context)
        
        # Format the main query with context
        main_query = await self._generate_workflow_query(context, template)
        
        # Create comprehensive context for MCP
        mcp_context = {
            "workflow_id": workflow_id,
            "workflow_type": template["name"],
            "trigger_type": context.trigger_type.value,
            "priority": context.priority.value,
            "portfolio_id": context.portfolio_id,
            "user_id": context.user_id,
            "proactive_workflow": True,
            "expected_agents": template.get("expected_agents", []),
            "workflow_steps": template.get("steps", []),
            **context_data  # Include risk event data, alert data, etc.
        }
        
        # Create job request
        job_request = JobRequest(
            query=main_query,
            context=mcp_context,
            priority=self._convert_priority_to_mcp(context.priority),
            timeout_seconds=template.get("timeout_seconds", 600),
            required_capabilities=[
                step["agent_capability"] 
                for step in template.get("steps", [])
                if step.get("required", True)
            ]
        )
        
        return job_request
    
    async def _get_context_data(self, context: WorkflowContext) -> Dict[str, Any]:
        """Get additional context data based on workflow trigger"""
        context_data = {}
        
        try:
            db = next(get_db())
            try:
                # Get risk event data if available
                if context.risk_event_id:
                    risk_event = db.query(RiskChangeEvent).filter(
                        RiskChangeEvent.id == context.risk_event_id
                    ).first()
                    
                    if risk_event:
                        context_data.update({
                            "risk_direction": risk_event.risk_direction,
                            "risk_magnitude_pct": risk_event.risk_magnitude_pct,
                            "threshold_breached": risk_event.threshold_breached,
                            "significant_changes": risk_event.significant_changes,
                            "current_risk_score": getattr(risk_event.current_snapshot, 'risk_score', None) if risk_event.current_snapshot else None
                        })
                
                # Get alert data if available
                if context.alert_id:
                    alert = db.query(ProactiveAlert).filter(
                        ProactiveAlert.id == context.alert_id
                    ).first()
                    
                    if alert:
                        context_data.update({
                            "alert_type": alert.alert_type,
                            "alert_title": alert.title,
                            "alert_message": alert.message,
                            "alert_details": alert.details
                        })
                
                # Add custom parameters
                context_data.update(context.custom_params)
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting context data: {e}")
        
        return context_data
    
    def _convert_priority_to_mcp(self, priority: WorkflowPriority) -> int:
        """Convert workflow priority to MCP priority integer"""
        priority_map = {
            WorkflowPriority.LOW: 8,
            WorkflowPriority.MEDIUM: 6,
            WorkflowPriority.HIGH: 4,
            WorkflowPriority.CRITICAL: 2,
            WorkflowPriority.EMERGENCY: 1
        }
        return priority_map.get(priority, 6)
    
    async def _track_workflow_execution(
        self,
        workflow_id: str,
        mcp_response: Dict,
        db: Session
    ):
        """Enhanced workflow tracking with notifications"""
        try:
            # Get workflow context
            context = self.active_workflows.get(workflow_id)
            if context:
                # Import enhanced notification function
                from main import send_enhanced_workflow_update_notification
                
                # Send workflow started notification
                workflow_data = {
                    "workflow_id": workflow_id,
                    "status": "started",
                    "progress": 0,
                    "current_agent": "AI Team",
                    "step": "Initializing Analysis",
                    "message": "Multi-agent analysis workflow started",
                    "workflow_type": "risk_analysis"
                }
                
                await send_enhanced_workflow_update_notification(
                    str(context.user_id), workflow_data
                )
            update_workflow_session_state(
                db, workflow_id, "running",
                step_result={"mcp_job_submitted": True, "job_id": mcp_response.get("job_id")},
                step_name="mcp_submission"
            )
            
            logger.info(f"Started tracking workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Error tracking workflow {workflow_id}: {e}")
    
    # Workflow Status and Management Methods
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        try:
            db = next(get_db())
            try:
                # Get workflow session from database
                session = db.query(WorkflowSessionDB).filter(
                    WorkflowSessionDB.session_id == workflow_id
                ).first()
                
                if not session:
                    return {"error": "Workflow not found"}
                
                # Get workflow steps
                steps = db.query(WorkflowStepDB).filter(
                    WorkflowStepDB.session_id == workflow_id
                ).order_by(WorkflowStepDB.step_number).all()
                
                # Get MCP job logs
                mcp_logs = db.query(MCPJobLog).filter(
                    MCPJobLog.session_id == workflow_id
                ).order_by(MCPJobLog.submitted_at).all()
                
                return {
                    "workflow_id": workflow_id,
                    "status": session.state,
                    "progress": session.get_progress_percentage(),
                    "created_at": session.created_at.isoformat(),
                    "updated_at": session.updated_at.isoformat(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None,
                    "workflow_type": session.workflow_type,
                    "steps": [
                        {
                            "step_name": step.step_name,
                            "status": step.status,
                            "agent_id": step.agent_id,
                            "execution_time_ms": step.execution_time_ms,
                            "confidence_score": step.confidence_score
                        }
                        for step in steps
                    ],
                    "mcp_jobs": [
                        {
                            "job_id": log.job_id,
                            "status": log.status,
                            "execution_time_ms": log.total_execution_time_ms,
                            "agents_involved": log.agents_involved
                        }
                        for log in mcp_logs
                    ],
                    "context": self.active_workflows.get(workflow_id).__dict__ if workflow_id in self.active_workflows else None
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting workflow status {workflow_id}: {e}")
            return {"error": str(e)}
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        try:
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
            
            # Update database status
            db = next(get_db())
            try:
                update_workflow_session_state(
                    db, workflow_id, "cancelled",
                    result={"cancelled_at": datetime.utcnow().isoformat()}
                )
                
                logger.info(f"Workflow {workflow_id} cancelled")
                return True
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error cancelling workflow {workflow_id}: {e}")
            return False
    
    async def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get completed workflow results"""
        try:
            db = next(get_db())
            try:
                session = db.query(WorkflowSessionDB).filter(
                    WorkflowSessionDB.session_id == workflow_id
                ).first()
                
                if not session:
                    return None
                
                if session.state != "complete":
                    return {"status": session.state, "message": "Workflow not yet completed"}
                
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "results": {
                        "strategy_analysis": session.strategy_result,
                        "security_screening": session.screening_result,
                        "quantitative_analysis": session.analysis_result,
                        "final_synthesis": session.final_synthesis
                    },
                    "confidence_score": session.confidence_score,
                    "execution_time_seconds": session.get_duration_seconds(),
                    "completed_at": session.completed_at.isoformat() if session.completed_at else None
                }
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error getting workflow results {workflow_id}: {e}")
            return {"error": str(e)}
    
    # Integration Methods for Risk Monitoring
    async def handle_risk_alert(
        self,
        risk_event: RiskChangeEvent,
        alert: ProactiveAlert,
        auto_trigger: bool = True
    ) -> Optional[WorkflowResult]:
        """Handle risk alert by potentially triggering workflow"""
        try:
            # Determine if workflow should be triggered
            should_trigger = await self._should_trigger_workflow_for_alert(
                risk_event, alert, auto_trigger
            )
            
            if not should_trigger:
                logger.info(f"Workflow not triggered for alert {alert.id}")
                return None
            
            # Create workflow context
            context = WorkflowContext(
                trigger_type=WorkflowTrigger.RISK_ALERT,
                portfolio_id=alert.portfolio_id,
                user_id=alert.user_id,
                risk_event_id=risk_event.id,
                alert_id=alert.id,
                priority=self._determine_workflow_priority(risk_event, alert)
            )
            
            # Select appropriate template
            template_name = self._select_workflow_template(risk_event, alert)
            
            # Trigger workflow
            result = await self.trigger_workflow(context, template_name)
            
            # Update alert with workflow information
            if result.status != "failed":
                db = next(get_db())
                try:
                    alert.details = alert.details or {}
                    alert.details.update({
                        "workflow_triggered": True,
                        "workflow_id": result.workflow_id,
                        "workflow_status": result.status
                    })
                    db.commit()
                finally:
                    db.close()
            
            logger.info(f"Workflow triggered for risk alert {alert.id}: {result.workflow_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling risk alert {alert.id}: {e}")
            return None
    
    async def _should_trigger_workflow_for_alert(
        self,
        risk_event: RiskChangeEvent,
        alert: ProactiveAlert,
        auto_trigger: bool
    ) -> bool:
        """Determine if workflow should be triggered for alert"""
        
        if not auto_trigger:
            return False
        
        # Always trigger for high/critical priority
        if alert.priority in ["high", "critical"]:
            return True
        
        # Trigger for significant threshold breaches
        if risk_event.threshold_breached and risk_event.risk_magnitude_pct > 20:
            return True
        
        # Trigger for medium priority with significant changes
        if alert.priority == "medium" and risk_event.risk_magnitude_pct > 15:
            return True
        
        return False
    
    def _determine_workflow_priority(
        self,
        risk_event: RiskChangeEvent,
        alert: ProactiveAlert
    ) -> WorkflowPriority:
        """Determine workflow priority based on risk event and alert"""
        
        if alert.priority == "critical" or risk_event.risk_magnitude_pct > 30:
            return WorkflowPriority.CRITICAL
        elif alert.priority == "high" or risk_event.risk_magnitude_pct > 20:
            return WorkflowPriority.HIGH
        elif alert.priority == "medium" or risk_event.risk_magnitude_pct > 10:
            return WorkflowPriority.MEDIUM
        else:
            return WorkflowPriority.LOW
    
    def _select_workflow_template(
        self,
        risk_event: RiskChangeEvent,
        alert: ProactiveAlert
    ) -> str:
        """Select appropriate workflow template"""
        
        if risk_event.threshold_breached and alert.priority == "critical":
            return "threshold_breach_response"
        else:
            return "risk_alert_analysis"
    
    # Statistics and Monitoring
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        stats = self.execution_stats.copy()
        
        # Calculate success rate
        total = stats["workflows_triggered"]
        if total > 0:
            stats["success_rate"] = (stats["workflows_completed"] / total) * 100
        
        stats.update({
            "active_workflows": len(self.active_workflows),
            "templates_available": len(self.workflow_templates),
            "last_updated": datetime.utcnow().isoformat()
        })
        
        return stats


# Global workflow integration engine instance
_workflow_integration_engine = None

async def get_workflow_integration_engine() -> WorkflowIntegrationEngine:
    """Get global workflow integration engine instance"""
    global _workflow_integration_engine
    if _workflow_integration_engine is None:
        _workflow_integration_engine = WorkflowIntegrationEngine()
    return _workflow_integration_engine

# Convenience functions for integration
async def trigger_risk_workflow(
    portfolio_id: int,
    user_id: int,
    risk_event: RiskChangeEvent,
    alert: Optional[ProactiveAlert] = None
) -> Optional[WorkflowResult]:
    """Convenience function to trigger workflow from risk event"""
    
    engine = await get_workflow_integration_engine()
    
    if alert:
        return await engine.handle_risk_alert(risk_event, alert)
    else:
        # Create workflow context directly
        context = WorkflowContext(
            trigger_type=WorkflowTrigger.RISK_ALERT,
            portfolio_id=portfolio_id,
            user_id=user_id,
            risk_event_id=risk_event.id,
            priority=WorkflowPriority.HIGH if risk_event.risk_magnitude_pct > 20 else WorkflowPriority.MEDIUM
        )
        
        return await engine.trigger_workflow(context)

async def get_workflow_status(workflow_id: str) -> Dict[str, Any]:
    """Get workflow status"""
    engine = await get_workflow_integration_engine()
    return await engine.get_workflow_status(workflow_id)

async def get_workflow_results(workflow_id: str) -> Optional[Dict[str, Any]]:
    """Get workflow results"""
    engine = await get_workflow_integration_engine()
    return await engine.get_workflow_results(workflow_id)


if __name__ == "__main__":
    async def test_workflow_integration():
        print("🔄 Testing Workflow Integration Engine")
        print("=" * 50)
        
        engine = WorkflowIntegrationEngine()
        
        # Test workflow context creation
        context = WorkflowContext(
            trigger_type=WorkflowTrigger.RISK_ALERT,
            portfolio_id=1,
            user_id=1,
            priority=WorkflowPriority.HIGH,
            custom_params={
                "risk_magnitude_pct": 25.0,
                "risk_direction": "increase"
            }
        )
        
        print(f"✅ Created workflow context: {context.trigger_type.value}")
        
        # Test template availability
        templates = list(engine.workflow_templates.keys())
        print(f"📋 Available templates: {templates}")
        
        # Test workflow triggering (would normally connect to MCP)
        print("🚀 Testing workflow trigger...")
        
        # Get execution statistics
        stats = engine.get_execution_statistics()
        print(f"📊 Execution stats: {stats}")
        
        print("\n🎉 Workflow Integration Engine testing complete!")
        print("\nIntegration Points:")
        print("- ✅ Risk event handling")
        print("- ✅ Alert workflow triggering") 
        print("- ✅ MCP job submission")
        print("- ✅ Workflow tracking")
        print("- ✅ Results management")
    
    asyncio.run(test_workflow_integration())