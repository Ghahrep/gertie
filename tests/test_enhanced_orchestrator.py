# Create: tests/test_enhanced_orchestrator.py

import pytest
from agents.orchestrator import FinancialOrchestrator

class TestEnhancedOrchestrator:
    def setup_method(self):
        self.orchestrator = FinancialOrchestrator()
    
    def test_cross_asset_analyst_in_roster(self):
        """Test CrossAssetAnalyst is properly integrated"""
        assert "CrossAssetAnalyst" in self.orchestrator.roster
        cross_asset_agent = self.orchestrator.roster["CrossAssetAnalyst"]
        assert cross_asset_agent.name == "Cross-Asset Risk Analyst"
    
    def test_enhanced_complexity_analysis(self):
        """Test enhanced complexity analysis"""
        # High complexity query
        high_query = "Comprehensive cross-asset correlation analysis needed"
        should_trigger, workflow_type, score = self.orchestrator._should_trigger_workflow(high_query)
        
        assert should_trigger == True
        assert workflow_type in ["enhanced", "crisis"]
        assert score > 0.6
        
        # Low complexity query
        low_query = "What is diversification?"
        should_trigger, workflow_type, score = self.orchestrator._should_trigger_workflow(low_query)
        
        # Should still work with existing logic
        assert isinstance(should_trigger, bool)
    
    def test_committee_templates_initialized(self):
        """Test committee templates are properly set up"""
        templates = self.orchestrator.committee_templates
        
        assert "standard" in templates
        assert "enhanced" in templates
        assert "crisis" in templates
        
        enhanced = templates["enhanced"]
        assert len(enhanced.agents) >= 6
        assert "CrossAssetAnalyst" in enhanced.agents
    
    def test_agent_capabilities_mapping(self):
        """Test agent capabilities are mapped"""
        capabilities = self.orchestrator.agent_capabilities
        
        assert "CrossAssetAnalyst" in capabilities
        assert "correlation_analysis" in capabilities["CrossAssetAnalyst"]
        
        # Verify existing agents still mapped
        assert "QuantitativeAnalystAgent" in capabilities
        assert "SecurityScreenerAgent" in capabilities

# Run test
if __name__ == "__main__":
    pytest.main([__file__, "-v"])