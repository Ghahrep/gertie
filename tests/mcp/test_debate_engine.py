# test_debate_engine_detailed.py
"""
Detailed debate engine tests
===========================
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from mcp.debate_engine import DebateEngine, DebateStage, AgentPerspective, DebatePosition
from datetime import datetime

class TestDebateEngineDetailed:
    """Detailed debate engine testing"""
    
    @pytest.fixture
    def mock_mcp_server(self):
        mock_mcp = Mock()
        mock_mcp.execute_agent_analysis = AsyncMock(return_value={
            "stance": "test_stance",
            "key_arguments": ["arg1", "arg2"],
            "evidence": [{"type": "analytical", "confidence": 0.8}],
            "confidence": 0.85,
            "risk_assessment": {"primary_risks": ["test_risk"]}
        })
        return mock_mcp
    
    @pytest.fixture 
    def debate_engine(self, mock_mcp_server):
        return DebateEngine(mock_mcp_server)
    
    def test_query_analysis_comprehensive(self, debate_engine):
        """Test comprehensive query analysis"""
        complex_query = """
        Given the current market volatility and rising interest rates,
        should we rebalance our aggressive growth portfolio to a more
        conservative allocation, considering tax implications and
        maintaining our long-term retirement goals?
        """
        
        portfolio_context = {
            "portfolio_id": "retirement_portfolio",
            "current_allocation": {"stocks": 0.8, "bonds": 0.15, "cash": 0.05},
            "time_horizon": "20_years",
            "risk_tolerance": "moderate_aggressive"
        }
        
        analysis = asyncio.run(
            debate_engine._analyze_query_requirements(complex_query, portfolio_context)
        )
        
        assert analysis["complexity_level"] == "high"
        assert "risk_analysis" in analysis["required_expertise"]
        assert "tax_optimization" in analysis["required_expertise"] 
        assert analysis["time_sensitivity"] in ["medium", "high"]
        assert len(analysis["expected_disagreement_areas"]) > 0
    
    @pytest.mark.asyncio
    async def test_agent_perspective_assignment(self, debate_engine):
        """Test intelligent agent perspective assignment"""
        agents = ["quantitative_analyst", "market_intelligence", "tax_strategist"]
        analysis = {
            "primary_focus": "risk_analysis",
            "required_expertise": ["risk_analysis", "tax_optimization"],
            "complexity_level": "high"
        }
        
        assignments = await debate_engine._assign_agent_perspectives(agents, analysis)
        
        # Verify all agents get assignments
        assert len(assignments) == len(agents)
        
        # Verify assignment structure
        for agent_id, assignment in assignments.items():
            assert "perspective" in assignment
            assert "role" in assignment
            assert "focus_areas" in assignment
            assert "expected_stance" in assignment
            
            # Verify perspective is valid enum value
            assert isinstance(assignment["perspective"], AgentPerspective)
    
    @pytest.mark.asyncio
    async def test_opening_positions_collection(self, debate_engine):
        """Test opening position collection process"""
        # Mock debate state
        debate_id = "test_debate"
        debate_engine.active_debates[debate_id] = {
            "query": "Should we increase portfolio risk?",
            "portfolio_context": {"portfolio_id": "test"},
            "participants": {
                "quantitative_analyst": {
                    "perspective": AgentPerspective.CONSERVATIVE,
                    "role": "risk_assessor",
                    "focus_areas": ["risk_analysis"]
                },
                "market_intelligence": {
                    "perspective": AgentPerspective.AGGRESSIVE,
                    "role": "opportunity_identifier", 
                    "focus_areas": ["market_analysis"]
                }
            },
            "current_stage": DebateStage.INITIALIZATION,
            "rounds": []
        }
        
        # Execute opening positions
        await debate_engine._conduct_opening_positions(debate_id)
        
        # Verify results
        debate_state = debate_engine.active_debates[debate_id]
        assert len(debate_state["rounds"]) == 1
        
        opening_round = debate_state["rounds"][0]
        assert opening_round.stage == DebateStage.OPENING_POSITIONS
        assert len(opening_round.positions) == 2  # Two participating agents
        
        # Verify position structure
        for position in opening_round.positions:
            assert hasattr(position, 'agent_id')
            assert hasattr(position, 'stance')
            assert hasattr(position, 'confidence_score')
    
    def test_challenge_generation_logic(self, debate_engine):
        """Test challenge generation between opposing positions"""
        # Create opposing positions
        conservative_pos = DebatePosition(
            agent_id="conservative_agent",
            stance="reduce portfolio risk",
            key_arguments=["Market volatility high", "Downside protection needed"],
            supporting_evidence=[{"type": "statistical", "confidence": 0.9}],
            confidence_score=0.85,
            risk_assessment={"primary_risks": ["market_risk"]},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        aggressive_pos = DebatePosition(
            agent_id="aggressive_agent", 
            stance="increase portfolio risk for higher returns",
            key_arguments=["Bull market continues", "Opportunity cost of safety"],
            supporting_evidence=[{"type": "analytical", "confidence": 0.8}],
            confidence_score=0.82,
            risk_assessment={"primary_risks": ["opportunity_cost"]},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        # Generate challenge
        challenge = asyncio.run(
            debate_engine._generate_challenge(
                conservative_pos, aggressive_pos, "portfolio allocation debate"
            )
        )
        
        assert challenge is not None
        assert "aggressive_agent" in challenge  # Should reference target
        assert len(challenge) > 50  # Should be substantial challenge
    
    def test_position_similarity_detection(self, debate_engine):
        """Test position similarity detection for grouping"""
        similar_pos1 = DebatePosition(
            agent_id="agent1",
            stance="conservative risk management approach",
            key_arguments=["Protect capital", "Minimize volatility"],
            supporting_evidence=[],
            confidence_score=0.8,
            risk_assessment={"primary_risks": ["market_risk", "volatility"]},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        similar_pos2 = DebatePosition(
            agent_id="agent2",
            stance="conservative capital preservation strategy", 
            key_arguments=["Protect capital", "Risk management priority"],
            supporting_evidence=[],
            confidence_score=0.75,
            risk_assessment={"primary_risks": ["market_risk", "drawdown"]},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        different_pos = DebatePosition(
            agent_id="agent3",
            stance="aggressive growth maximization",
            key_arguments=["High returns", "Accept volatility"],
            supporting_evidence=[],
            confidence_score=0.9,
            risk_assessment={"primary_risks": ["opportunity_cost"]},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        # Test similarity detection
        assert debate_engine._positions_similar(similar_pos1, similar_pos2)
        assert not debate_engine._positions_similar(similar_pos1, different_pos)
    
    @pytest.mark.asyncio
    async def test_consensus_building_from_debate(self, debate_engine):
        """Test consensus building from debate positions"""
        # Mock final debate positions
        final_positions = [
            DebatePosition(
                agent_id="agent1",
                stance="moderate risk increase", 
                key_arguments=["Balanced approach", "Gradual transition"],
                supporting_evidence=[{"type": "analytical", "confidence": 0.8}],
                confidence_score=0.85,
                risk_assessment={"primary_risks": ["transition_risk"]},
                counterarguments_addressed=["volatility_concerns"],
                timestamp=datetime.now()
            ),
            DebatePosition(
                agent_id="agent2",
                stance="moderate risk increase",
                key_arguments=["Market opportunity", "Risk-adjusted returns"],
                supporting_evidence=[{"type": "statistical", "confidence": 0.9}],
                confidence_score=0.82,
                risk_assessment={"primary_risks": ["timing_risk"]},
                counterarguments_addressed=["market_timing"],
                timestamp=datetime.now()
            )
        ]
        
        consensus = await debate_engine._build_weighted_consensus(
            final_positions, "portfolio risk optimization"
        )
        
        assert "recommendation" in consensus
        assert "confidence_level" in consensus
        assert "supporting_arguments" in consensus
        assert consensus["confidence_level"] > 0
        assert len(consensus["supporting_arguments"]) > 0