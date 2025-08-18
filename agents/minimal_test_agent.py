# agents/minimal_test_agent.py
"""
Minimal test agent for MCP foundation testing.
This avoids complex dependencies and focuses on MCP integration.
"""

import asyncio
from typing import Dict, List, Any
from datetime import datetime
import logging

from agents.mcp_base_agent import MCPBaseAgent

logger = logging.getLogger(__name__)

class MinimalTestAgent(MCPBaseAgent):
    """Minimal agent for testing MCP foundation"""
    
    def __init__(self, agent_id: str = "minimal_test_agent"):
        super().__init__(
            agent_id=agent_id,
            agent_name="Minimal Test Agent",
            capabilities=[
                "test_analysis",
                "mock_portfolio_analysis", 
                "simple_calculation"
            ],
            max_concurrent_jobs=2
        )
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute simple test capabilities"""
        logger.info(f"Executing test capability: {capability}")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        if capability == "test_analysis":
            return await self.test_analysis(data, context)
        elif capability == "mock_portfolio_analysis":
            return await self.mock_portfolio_analysis(data, context)
        elif capability == "simple_calculation":
            return await self.simple_calculation(data, context)
        else:
            raise ValueError(f"Capability {capability} not supported")
    
    async def test_analysis(self, data: Dict, context: Dict) -> Dict:
        """Simple test analysis"""
        query = data.get("query", "")
        
        return {
            "analysis_type": "test_analysis",
            "query_processed": query,
            "result": "Test analysis completed successfully",
            "confidence_score": 0.95,
            "timestamp": datetime.utcnow().isoformat(),
            "test_data": {
                "input_length": len(query),
                "has_portfolio_context": "portfolio_id" in context,
                "processing_time_ms": 100
            }
        }
    
    async def mock_portfolio_analysis(self, data: Dict, context: Dict) -> Dict:
        """Mock portfolio analysis for testing"""
        portfolio_data = data.get("portfolio_data", {})
        
        if not portfolio_data:
            return {
                "analysis_type": "mock_portfolio_analysis",
                "result": "No portfolio data provided",
                "confidence_score": 0.0
            }
        
        # Mock some basic portfolio metrics
        holdings_count = len(portfolio_data.get("holdings", []))
        total_value = portfolio_data.get("total_value", 0)
        
        return {
            "analysis_type": "mock_portfolio_analysis",
            "portfolio_summary": {
                "holdings_count": holdings_count,
                "total_value": total_value,
                "analysis_date": datetime.utcnow().isoformat()
            },
            "mock_metrics": {
                "risk_score": 5.5,  # 1-10 scale
                "diversification_score": 7.2,
                "estimated_return": 0.08
            },
            "confidence_score": 0.85,
            "methodology": "Mock analysis for testing"
        }
    
    async def simple_calculation(self, data: Dict, context: Dict) -> Dict:
        """Simple mathematical calculation for testing"""
        numbers = data.get("numbers", [1, 2, 3, 4, 5])
        
        if not numbers:
            return {
                "error": "No numbers provided for calculation"
            }
        
        return {
            "analysis_type": "simple_calculation",
            "input_numbers": numbers,
            "results": {
                "sum": sum(numbers),
                "average": sum(numbers) / len(numbers),
                "count": len(numbers),
                "min": min(numbers),
                "max": max(numbers)
            },
            "confidence_score": 1.0,
            "calculation_time": datetime.utcnow().isoformat()
        }
    
    def get_response_time_sla(self) -> int:
        """Fast response for testing"""
        return 10  # 10 seconds
    
    def get_supported_data_sources(self) -> List[str]:
        """Test data sources"""
        return ["test_data", "mock_portfolio_api"]