# mcp/__init__.py
"""
Master Control Plane (MCP) Package

The MCP is the central orchestration system for Gertie.ai's multi-agent architecture.
It manages agent registration, job distribution, workflow execution, and result synthesis.
"""

from .schemas import (
    JobStatus,
    AgentRegistration,
    JobRequest,
    JobResponse,
    HealthCheck,
    AgentJobRequest,
    AgentJobResponse
)

from .workflow_engine import WorkflowEngine
from .server import app

__version__ = "1.0.0"
__all__ = [
    "JobStatus",
    "AgentRegistration", 
    "JobRequest",
    "JobResponse",
    "HealthCheck",
    "AgentJobRequest",
    "AgentJobResponse",
    "WorkflowEngine",
    "app"
]