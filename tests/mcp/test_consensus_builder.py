"""
Detailed consensus building algorithm tests
==========================================
"""

import pytest
from mcp.consensus_builder import ConsensusBuilder, ConsensusMetrics

class TestConsensusAlgorithms:
    """Test consensus building algorithms in detail"""
    
    @pytest.fixture
    def consensus_builder(self):
        return ConsensusBuilder()
    
    def test_expertise_weighting_algorithm(self, consensus_builder):
        """Test expertise-based weighting calculations"""
        # Test query topic extraction
        risk_query = "analyze portfolio risk and implement downside protection"
        topics = consensus_builder._extract_query_topics(risk_query)
        assert "risk_analysis" in topics
        
        # Test expertise relevance calculation
        relevance = consensus_builder._calculate_expertise_relevance(
            "quantitative_analyst", topics
        )
        assert 0.5 <= relevance <= 1.0  # Should have high relevance for risk analysis
        
        # Test with non-expert agent
        irrelevant_relevance = consensus_builder._calculate_expertise_relevance(
            "unknown_agent", topics
        )
        assert irrelevant_relevance == 0.5  # Default moderate relevance
    
    def test_evidence_quality_assessment(self, consensus_builder):
        """Test evidence quality assessment algorithm"""
        # High quality evidence mix
        high_quality = [
            {
                "type": "statistical",
                "source": "Federal Reserve",
                "confidence": 0.95,
                "data": "Comprehensive economic indicators showing 3.2% GDP growth with 95% confidence interval",
                "methodology": "Time series analysis with robust statistical methods"
            },
            {
                "type": "empirical", 
                "source": "Peer-reviewed study",
                "confidence": 0.88,
                "data": "Historical analysis of 50-year market cycles"
            }
        ]
        
        quality_score = consensus_builder._assess_evidence_quality(high_quality)
        assert quality_score >= 0.8
        
        # Low quality evidence
        low_quality = [
            {
                "type": "opinion",
                "confidence": 0.4,
                "data": "I think the market will go up"
            }
        ]
        
        low_score = consensus_builder._assess_evidence_quality(low_quality)
        assert low_score < 0.5
    
    def test_minority_opinion_preservation(self, consensus_builder):
        """Test minority opinion preservation algorithm"""
        # Create a scenario where minority has important insights
        positions = [
            # Majority position (3 agents)
            {
                "agent_id": "agent_1",
                "stance": "maintain current allocation",
                "key_arguments": ["Market is stable", "Current performance good"],
                "supporting_evidence": [{"type": "analytical", "confidence": 0.7}],
                "confidence_score": 0.75,
                "risk_assessment": {"primary_risks": ["minor_volatility"]}
            },
            {
                "agent_id": "agent_2", 
                "stance": "maintain current allocation",
                "key_arguments": ["No major changes needed", "Risk level acceptable"],
                "supporting_evidence": [{"type": "analytical", "confidence": 0.7}],
                "confidence_score": 0.72,
                "risk_assessment": {"primary_risks": ["minor_volatility"]}
            },
            {
                "agent_id": "agent_3",
                "stance": "maintain current allocation", 
                "key_arguments": ["Portfolio balanced", "Good diversification"],
                "supporting_evidence": [{"type": "analytical", "confidence": 0.7}],
                "confidence_score": 0.73,
                "risk_assessment": {"primary_risks": ["minor_volatility"]}
            },
            # High-confidence minority with unique risk insight
            {
                "agent_id": "risk_specialist",
                "stance": "reduce risk exposure immediately",
                "key_arguments": ["Hidden correlation risk detected", "Systemic risk building"],
                "supporting_evidence": [
                    {"type": "statistical", "confidence": 0.95, "data": "Correlation analysis shows 85% correlation"},
                    {"type": "empirical", "confidence": 0.9, "data": "Historical precedent from 2008 crisis"}
                ],
                "confidence_score": 0.93,
                "risk_assessment": {"primary_risks": ["systemic_risk", "correlation_risk", "liquidity_risk"]}
            }
        ]
        
        consensus = consensus_builder.calculate_weighted_consensus(
            positions, {"query": "portfolio risk assessment"}
        )
        
        # Should preserve the high-confidence minority opinion
        assert len(consensus["minority_opinions"]) >= 1
        
        minority = consensus["minority_opinions"][0]
        assert "risk_specialist" in minority["agents"]
        assert minority["weight"] > 0  # Should have meaningful weight
    
    def test_confidence_calculation_algorithm(self, consensus_builder):
        """Test overall confidence calculation"""
        # High agreement scenario
        high_agreement_positions = [
            {"agent_id": f"agent_{i}", "confidence_score": 0.9, "stance": "bullish"} 
            for i in range(5)
        ]
        
        # Low agreement scenario  
        low_agreement_positions = [
            {"agent_id": "agent_1", "confidence_score": 0.9, "stance": "bullish"},
            {"agent_id": "agent_2", "confidence_score": 0.8, "stance": "bearish"},
            {"agent_id": "agent_3", "confidence_score": 0.7, "stance": "neutral"},
        ]
        
        # Mock the required methods for testing
        def mock_group_positions(positions):
            if len(set(p.get("stance", "") for p in positions)) == 1:
                return [positions]  # All same stance = one group
            else:
                return [[p] for p in positions]  # Different stances = separate groups
        
        consensus_builder._group_similar_positions = mock_group_positions
        
        # High agreement should yield higher confidence
        high_groups = mock_group_positions(high_agreement_positions)
        high_conf = consensus_builder._calculate_consensus_confidence(
            high_agreement_positions, high_groups
        )
        
        low_groups = mock_group_positions(low_agreement_positions)
        low_conf = consensus_builder._calculate_consensus_confidence(
            low_agreement_positions, low_groups
        )
        
        assert high_conf > low_conf  # High agreement should have higher confidence