import pytest
from fastapi.testclient import TestClient
from mcp.server import app, agent_registry
from mcp.schemas import AgentRegistration, JobRequest

class TestServerEndpoints:
    """Test all server endpoints thoroughly"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint_structure(self, client):
        """Test health endpoint returns correct structure"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = ["status", "timestamp", "registered_agents", "active_jobs"]
        assert all(field in data for field in required_fields)
        
        assert data["status"] == "healthy"
        assert isinstance(data["registered_agents"], int)
        assert isinstance(data["active_jobs"], int)
    
    def test_register_endpoint_validation(self, client):
        """Test agent registration endpoint validation"""
        # Test valid registration
        valid_agent = {
            "agent_id": "test_agent_001",
            "agent_name": "Test Agent",
            "agent_type": "TestAgent",
            "capabilities": ["test_capability"],
            "max_concurrent_jobs": 3
        }
        
        response = client.post("/register", json=valid_agent)
        assert response.status_code == 200
        assert "success" in response.json()["status"]
        
        # Test missing required fields
        invalid_agent = {"agent_id": "incomplete"}
        response = client.post("/register", json=invalid_agent)
        assert response.status_code == 422  # Validation error
    
    def test_job_submission_edge_cases(self, client):
        """Test job submission edge cases"""
        # Register a test agent first
        agent_registry.clear()
        test_agent = {
            "agent_id": "test_agent",
            "agent_name": "Test Agent", 
            "agent_type": "TestAgent",
            "capabilities": ["risk_analysis"]
        }
        client.post("/register", json=test_agent)
        
        # Test empty query
        empty_query = {"query": "", "context": {}}
        response = client.post("/submit_job", json=empty_query)
        assert response.status_code == 200  # Should accept empty query
        
        # Test very long query
        long_query = {"query": "x" * 10000, "context": {}}
        response = client.post("/submit_job", json=long_query)
        assert response.status_code == 200
        
        # Test invalid priority
        invalid_priority = {"query": "test", "priority": 15}
        response = client.post("/submit_job", json=invalid_priority)
        assert response.status_code == 422  # Validation error
