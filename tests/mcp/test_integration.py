"""
Integration test scenarios
=========================
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from mcp.server import app
from fastapi.testclient import TestClient

class TestIntegrationScenarios:
    """Test realistic integration scenarios"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_portfolio_analysis_workflow(self, client, sample_portfolio_context):
        """Test complete portfolio analysis workflow"""
        # 1. Register multiple specialized agents
        agents = [
            {
                "agent_id": "quantitative_analyst",
                "agent_name": "Quantitative Analysis Agent",
                "agent_type": "QuantitativeAgent", 
                "capabilities": ["risk_analysis", "portfolio_analysis", "statistical_modeling"]
            },
            {
                "agent_id": "tax_strategist",
                "agent_name": "Tax Strategy Agent",
                "agent_type": "TaxAgent",
                "capabilities": ["tax_optimization", "after_tax_analysis", "regulatory_compliance"]
            },
            {
                "agent_id": "market_intelligence", 
                "agent_name": "Market Intelligence Agent",
                "agent_type": "MarketAgent",
                "capabilities": ["market_analysis", "trend_identification", "sentiment_analysis"]
            }
        ]
        
        # Register all agents
        for agent in agents:
            response = client.post("/register", json=agent)
            assert response.status_code == 200
        
        # 2. Submit comprehensive analysis job
        job_request = {
            "query": "Perform comprehensive portfolio analysis including risk assessment, tax optimization, and market outlook",
            "context": sample_portfolio_context,
            "priority": 8,
            "required_capabilities": ["risk_analysis", "tax_optimization", "market_analysis"]
        }
        
        response = client.post("/submit_job", json=job_request)
        assert response.status_code == 200
        
        job_data = response.json()
        job_id = job_data["job_id"]
        
        # 3. Monitor job progress
        status_response = client.get(f"/job/{job_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ["pending", "running", "completed"]
    
    @pytest.mark.asyncio
    async def test_multi_agent_debate_scenario(self, sample_portfolio_context):
        """Test multi-agent debate scenario"""
        from mcp.debate_engine import DebateEngine
        
        mock_mcp = AsyncMock()
        mock_mcp.execute_agent_analysis = AsyncMock(side_effect=[
            # Conservative agent response
            {
                "stance": "reduce portfolio risk",
                "key_arguments": ["Market volatility increasing", "Protect existing gains"],
                "evidence": [{"type": "statistical", "data": "VaR analysis shows 15% downside", "confidence": 0.9}],
                "confidence": 0.88,
                "risk_assessment": {"primary_risks": ["market_volatility", "sector_concentration"]}
            },
            # Aggressive agent response
            {
                "stance": "increase portfolio risk for growth",
                "key_arguments": ["Strong economic fundamentals", "Bull market continuation likely"],
                "evidence": [{"type": "analytical", "data": "GDP growth 3.5%, unemployment low", "confidence": 0.85}],
                "confidence": 0.83,
                "risk_assessment": {"primary_risks": ["inflation_risk", "interest_rate_risk"]}
            }
        ])
        
        debate_engine = DebateEngine(mock_mcp)
        
        # Initiate debate
        with patch.object(debate_engine, '_execute_debate', return_value=None):
            debate_id = await debate_engine.initiate_debate(
                query="Should we increase or decrease portfolio risk given current market conditions?",
                agents=["quantitative_analyst", "market_intelligence"],
                portfolio_context=sample_portfolio_context
            )
        
        # Verify debate initiated
        assert debate_id in debate_engine.active_debates
        debate_state = debate_engine.active_debates[debate_id]
        assert debate_state["status"] == "active"
        assert len(debate_state["participants"]) == 2
    
    def test_error_recovery_scenario(self, client):
        """Test system error recovery scenarios"""
        # Test job submission with no agents
        job_request = {
            "query": "Analyze portfolio with specialized capabilities",
            "required_capabilities": ["non_existent_capability"]
        }
        
        response = client.post("/submit_job", json=job_request)
        assert response.status_code == 400
        assert "No agents available" in response.json()["detail"]
        
        # Test invalid agent registration recovery
        invalid_agent = {
            "agent_id": "",  # Invalid empty ID
            "agent_name": "Test Agent",
            "agent_type": "TestType",
            "capabilities": []
        }
        
        response = client.post("/register", json=invalid_agent)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio 
    async def test_consensus_with_conflicting_agents(self):
        """Test consensus building with strongly conflicting agent opinions"""
        from mcp.consensus_builder import ConsensusBuilder
        
        # Create strongly conflicting positions
        conflicting_positions = [
            {
                "agent_id": "ultra_conservative",
                "stance": "sell all stocks immediately",
                "key_arguments": ["Market crash imminent", "Cash is king", "Preserve capital at all costs"],
                "supporting_evidence": [
                    {"type": "statistical", "data": "Technical indicators show 95% crash probability", "confidence": 0.95}
                ],
                "confidence_score": 0.95,
                "risk_assessment": {"primary_risks": ["total_market_collapse", "liquidity_crisis"]}
            },
            {
                "agent_id": "ultra_aggressive", 
                "stance": "leverage up and buy more stocks",
                "key_arguments": ["Historic bull run continues", "AI revolution just starting", "Don't miss generational opportunity"],
                "supporting_evidence": [
                    {"type": "analytical", "data": "Tech earnings up 45% YoY", "confidence": 0.92}
                ],
                "confidence_score": 0.93,
                "risk_assessment": {"primary_risks": ["opportunity_cost", "insufficient_exposure"]}
            },
            {
                "agent_id": "moderate_voice",
                "stance": "gradual portfolio rebalancing",
                "key_arguments": ["Balanced approach best", "Market timing impossible", "Diversification key"],
                "supporting_evidence": [
                    {"type": "empirical", "data": "Historical analysis shows balanced portfolios outperform", "confidence": 0.8}
                ],
                "confidence_score": 0.75,
                "risk_assessment": {"primary_risks": ["rebalancing_costs", "timing_risk"]}
            }
        ]
        
        consensus_builder = ConsensusBuilder()
        consensus = consensus_builder.calculate_weighted_consensus(
            conflicting_positions, {"query": "portfolio allocation strategy"}
        )
        
        # Should handle extreme disagreement gracefully
        assert "recommendation" in consensus
        assert 0 <= consensus["confidence_level"] <= 1
        
        # Should preserve minority opinions
        assert len(consensus["minority_opinions"]) >= 1
        
        # Confidence should be lower due to high disagreement
        assert consensus["confidence_level"] < 0.8  # High disagreement = lower confidence

