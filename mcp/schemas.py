# mcp/schemas.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class JobStatus(str, Enum):
    """Status enumeration for jobs and workflow steps"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentCapability(str, Enum):
    """Enumeration of available agent capabilities"""
    RISK_ANALYSIS = "risk_analysis"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    TAX_OPTIMIZATION = "tax_optimization"
    STRATEGY_REBALANCING = "strategy_rebalancing"
    MARKET_INTELLIGENCE = "market_intelligence"
    REAL_TIME_DATA = "real_time_data"
    NEWS_SENTIMENT = "news_sentiment"
    OPTIONS_ANALYSIS = "options_analysis"
    QUERY_INTERPRETATION = "query_interpretation"
    RESULT_SYNTHESIS = "result_synthesis"
    PORTFOLIO_DATA_FETCH = "portfolio_data_fetch"
    MARKET_DATA_FETCH = "market_data_fetch"
    GENERAL_ANALYSIS = "general_analysis"

class AgentRegistration(BaseModel):
    """Schema for agent registration with the MCP"""
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_name: str = Field(..., description="Human-readable name for the agent")
    agent_type: str = Field(..., description="Type/class of the agent")
    capabilities: List[str] = Field(..., description="List of capabilities this agent provides")
    endpoint_url: Optional[str] = Field(None, description="URL endpoint for agent communication")
    max_concurrent_jobs: int = Field(default=5, description="Maximum concurrent jobs this agent can handle")
    response_time_sla: int = Field(default=30, description="Expected response time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional agent metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "quantitative_analyst_001",
                "agent_name": "Quantitative Analysis Agent",
                "agent_type": "QuantitativeAnalystAgent",
                "capabilities": ["risk_analysis", "portfolio_analysis", "query_interpretation"],
                "endpoint_url": "http://localhost:8002/agent/quantitative",
                "max_concurrent_jobs": 3,
                "response_time_sla": 45,
                "metadata": {
                    "version": "1.0.0",
                    "specialization": "risk_management"
                }
            }
        }

class JobRequest(BaseModel):
    """Schema for job requests submitted to the MCP"""
    query: str = Field(..., description="Natural language query or analysis request")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context for the query")
    priority: int = Field(default=5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    timeout_seconds: Optional[int] = Field(default=300, description="Maximum execution time in seconds")
    required_capabilities: Optional[List[str]] = Field(None, description="Specific capabilities required")
    preferred_agents: Optional[List[str]] = Field(None, description="Preferred agents for this job")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Analyze the risk profile of my portfolio and suggest optimizations",
                "context": {
                    "portfolio_id": "user_123_main",
                    "analysis_depth": "comprehensive",
                    "include_tax_implications": True
                },
                "priority": 7,
                "timeout_seconds": 180,
                "required_capabilities": ["risk_analysis", "portfolio_analysis"],
                "preferred_agents": ["quantitative_analyst_001"]
            }
        }

class JobResponse(BaseModel):
    """Schema for job responses from the MCP"""
    job_id: str = Field(..., description="Unique identifier for the job")
    status: JobStatus = Field(..., description="Current status of the job")
    message: str = Field(..., description="Human-readable status message")
    result: Optional[Dict[str, Any]] = Field(None, description="Job results (when completed)")
    progress: Optional[float] = Field(None, ge=0, le=100, description="Job progress percentage")
    agents_involved: Optional[List[str]] = Field(None, description="Agents working on this job")
    estimated_completion_time: Optional[str] = Field(None, description="Estimated completion time (ISO format)")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details (if failed)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "status": "running",
                "message": "Analysis in progress - risk assessment complete",
                "result": None,
                "progress": 65.0,
                "agents_involved": ["quantitative_analyst_001", "market_intelligence_002"],
                "estimated_completion_time": "2024-01-15T14:30:00Z",
                "error_details": None
            }
        }

class HealthCheck(BaseModel):
    """Schema for MCP health check responses"""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(..., description="Timestamp of health check")
    registered_agents: int = Field(..., description="Number of registered agents")
    active_jobs: int = Field(..., description="Number of currently active jobs")
    system_metrics: Optional[Dict[str, Any]] = Field(None, description="Additional system metrics")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-15T14:25:00Z",
                "registered_agents": 5,
                "active_jobs": 3,
                "system_metrics": {
                    "cpu_usage": 45.2,
                    "memory_usage": 67.8,
                    "avg_response_time": 1.25
                }
            }
        }

class AgentJobRequest(BaseModel):
    """Schema for dispatching jobs to individual agents"""
    job_id: str = Field(..., description="Parent job identifier")
    step_id: str = Field(..., description="Workflow step identifier")
    capability: str = Field(..., description="Required capability for this step")
    input_data: Dict[str, Any] = Field(..., description="Input data for the agent")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    timeout_seconds: int = Field(default=60, description="Timeout for this specific step")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "job_abc123",
                "step_id": "risk_analysis_step",
                "capability": "risk_analysis",
                "input_data": {
                    "portfolio_data": {...},
                    "analysis_parameters": {
                        "confidence_level": 0.95,
                        "time_horizon": "1_year"
                    }
                },
                "context": {
                    "user_risk_tolerance": "moderate",
                    "investment_goals": ["retirement", "growth"]
                },
                "timeout_seconds": 45
            }
        }

class AgentJobResponse(BaseModel):
    """Schema for agent responses to job requests"""
    step_id: str = Field(..., description="Workflow step identifier")
    status: JobStatus = Field(..., description="Status of the step execution")
    result: Optional[Dict[str, Any]] = Field(None, description="Step execution results")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence in the results")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if step failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "step_id": "risk_analysis_step",
                "status": "completed",
                "result": {
                    "var_95": 0.024,
                    "max_drawdown": 0.187,
                    "risk_metrics": {
                        "sharpe_ratio": 1.34,
                        "sortino_ratio": 1.67,
                        "beta": 0.89
                    }
                },
                "confidence_score": 0.92,
                "execution_time_ms": 1250,
                "error_message": None,
                "metadata": {
                    "model_version": "v2.1",
                    "data_freshness": "real_time"
                }
            }
        }

class WorkflowTemplate(BaseModel):
    """Schema for defining reusable workflow templates"""
    template_id: str = Field(..., description="Unique identifier for the workflow template")
    name: str = Field(..., description="Human-readable name for the template")
    description: str = Field(..., description="Description of what this workflow accomplishes")
    trigger_patterns: List[str] = Field(..., description="Query patterns that should trigger this workflow")
    required_capabilities: List[str] = Field(..., description="Capabilities required for this workflow")
    estimated_duration_seconds: int = Field(..., description="Estimated workflow duration")
    steps: List[Dict[str, Any]] = Field(..., description="Workflow step definitions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "template_id": "comprehensive_portfolio_analysis",
                "name": "Comprehensive Portfolio Analysis",
                "description": "Full portfolio analysis including risk, performance, and optimization recommendations",
                "trigger_patterns": ["analyze portfolio", "portfolio review", "comprehensive analysis"],
                "required_capabilities": ["portfolio_analysis", "risk_analysis", "strategy_rebalancing"],
                "estimated_duration_seconds": 120,
                "steps": [
                    {
                        "step_id": "data_collection",
                        "capability": "portfolio_data_fetch",
                        "dependencies": []
                    },
                    {
                        "step_id": "risk_analysis",
                        "capability": "risk_analysis", 
                        "dependencies": ["data_collection"]
                    }
                ]
            }
        }

class DebateRequest(BaseModel):
    """Schema for multi-agent debate requests"""
    topic: str = Field(..., description="Topic or question for the debate")
    participating_agents: List[str] = Field(..., description="Agents that should participate in the debate")
    debate_rounds: int = Field(default=3, ge=1, le=10, description="Number of debate rounds")
    require_consensus: bool = Field(default=True, description="Whether consensus is required")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for the debate")
    
class DebateResponse(BaseModel):
    """Schema for multi-agent debate responses"""
    debate_id: str = Field(..., description="Unique identifier for the debate")
    topic: str = Field(..., description="Debate topic")
    rounds_completed: int = Field(..., description="Number of rounds completed")
    agent_viewpoints: List[Dict[str, Any]] = Field(..., description="Individual agent perspectives")
    consensus_reached: bool = Field(..., description="Whether consensus was achieved")
    final_recommendation: Optional[Dict[str, Any]] = Field(None, description="Final consensus recommendation")
    confidence_score: Optional[float] = Field(None, description="Overall confidence in the consensus")
    
class SystemMetrics(BaseModel):
    """Schema for system performance metrics"""
    timestamp: datetime = Field(..., description="Metrics collection timestamp")
    cpu_usage_percent: float = Field(..., description="CPU usage percentage")
    memory_usage_percent: float = Field(..., description="Memory usage percentage")
    active_jobs: int = Field(..., description="Number of active jobs")
    completed_jobs_last_hour: int = Field(..., description="Jobs completed in the last hour")
    average_job_duration_seconds: float = Field(..., description="Average job completion time")
    agent_health_scores: Dict[str, float] = Field(..., description="Health scores for each agent")
    error_rate_percent: float = Field(..., description="Error rate percentage")
    
class ConfigurationUpdate(BaseModel):
    """Schema for MCP configuration updates"""
    setting_name: str = Field(..., description="Name of the setting to update")
    setting_value: Any = Field(..., description="New value for the setting")
    apply_immediately: bool = Field(default=True, description="Whether to apply the change immediately")
    
class AgentLoadBalanceConfig(BaseModel):
    """Schema for agent load balancing configuration"""
    load_balancing_algorithm: str = Field(default="round_robin", description="Load balancing algorithm")
    health_check_interval_seconds: int = Field(default=30, description="Health check frequency")
    max_retries: int = Field(default=3, description="Maximum retry attempts for failed requests")
    retry_backoff_seconds: int = Field(default=5, description="Backoff time between retries")
    circuit_breaker_threshold: int = Field(default=5, description="Failure threshold for circuit breaker")