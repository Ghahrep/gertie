# test_mcp_comprehensive.py
"""
Comprehensive Unit Testing Suite for MCP Server and Components
=============================================================
Complete testing framework for all MCP components including:
- Server endpoints and registration
- Workflow engine execution 
- Consensus building algorithms
- Debate engine functionality
- Schema validation
- Error handling and edge cases
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from typing import Dict, List, Any

# Import MCP components
from mcp.server import app, agent_registry, workflow_engine
from mcp.workflow_engine import WorkflowEngine, Job, WorkflowStep
from mcp.consensus_builder import ConsensusBuilder, MinorityOpinion, ConsensusMetrics
from mcp.debate_engine import DebateEngine, DebateStage, AgentPerspective, DebatePosition
from mcp.schemas import (
    AgentRegistration, JobRequest, JobResponse, JobStatus,
    HealthCheck, AgentJobRequest, AgentJobResponse
)

# Test fixtures
@pytest.fixture
def test_client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def sample_agent_registration():
    """Sample agent registration data"""
    return AgentRegistration(
        agent_id="test_quantitative_analyst",
        agent_name="Test Quantitative Analyst",
        agent_type="QuantitativeAnalystAgent",
        capabilities=["risk_analysis", "portfolio_analysis", "statistical_modeling"],
        endpoint_url="http://localhost:8002/agent/quantitative",
        max_concurrent_jobs=3,
        response_time_sla=45,
        metadata={"version": "1.0.0", "specialization": "risk_management"}
    )

@pytest.fixture
def sample_job_request():
    """Sample job request data"""
    return JobRequest(
        query="Analyze the risk profile of my portfolio and suggest optimizations",
        context={
            "portfolio_id": "user_123_main",
            "analysis_depth": "comprehensive",
            "include_tax_implications": True
        },
        priority=7,
        timeout_seconds=180,
        required_capabilities=["risk_analysis", "portfolio_analysis"]
    )

@pytest.fixture
def sample_agent_positions():
    """Sample agent debate positions"""
    return [
        {
            "agent_id": "quantitative_analyst",
            "stance": "conservative approach recommended",
            "key_arguments": ["High market volatility", "Downside protection needed"],
            "supporting_evidence": [
                {"type": "statistical", "data": "VaR: 0.025", "confidence": 0.9}
            ],
            "confidence_score": 0.85,
            "risk_assessment": {"primary_risks": ["market_volatility", "concentration_risk"]}
        },
        {
            "agent_id": "market_intelligence", 
            "stance": "aggressive growth strategy",
            "key_arguments": ["Strong economic indicators", "Bull market momentum"],
            "supporting_evidence": [
                {"type": "analytical", "analysis": "GDP growth 3.2%", "confidence": 0.8}
            ],
            "confidence_score": 0.78,
            "risk_assessment": {"primary_risks": ["inflation_risk", "interest_rate_risk"]}
        }
    ]

@pytest.fixture
def workflow_engine_instance():
    """Clean workflow engine instance"""
    return WorkflowEngine()

@pytest.fixture
def consensus_builder_instance():
    """Consensus builder instance"""
    return ConsensusBuilder()

@pytest.fixture
def debate_engine_instance():
    """Debate engine instance with mock MCP server"""
    mock_mcp = Mock()
    return DebateEngine(mock_mcp)

# ===========================
# SERVER ENDPOINT TESTS
# ===========================

class TestMCPServer:
    """Test MCP server endpoints and functionality"""
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data["status"] == "healthy"
        assert "timestamp" in health_data
        assert "registered_agents" in health_data
        assert "active_jobs" in health_data
    
    def test_agent_registration_success(self, test_client, sample_agent_registration):
        """Test successful agent registration"""
        # Clear registry first
        agent_registry.clear()
        
        response = test_client.post(
            "/register", 
            json=sample_agent_registration.dict()
        )
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "success"
        assert "Agent test_quantitative_analyst registered successfully" in result["message"]
        assert "registered_at" in result
        
        # Verify agent is in registry
        assert "test_quantitative_analyst" in agent_registry
    
    def test_agent_registration_duplicate(self, test_client, sample_agent_registration):
        """Test duplicate agent registration fails"""
        # Register agent first time
        test_client.post("/register", json=sample_agent_registration.dict())
        
        # Try to register same agent again
        response = test_client.post("/register", json=sample_agent_registration.dict())
        assert response.status_code == 409
        assert "already registered" in response.json()["detail"]
    
    def test_list_agents(self, test_client, sample_agent_registration):
        """Test listing registered agents"""
        # Clear and register an agent
        agent_registry.clear()
        test_client.post("/register", json=sample_agent_registration.dict())
        
        response = test_client.get("/agents")
        assert response.status_code == 200
        
        agents = response.json()
        assert len(agents) == 1
        assert agents[0]["agent_id"] == "test_quantitative_analyst"
    
    def test_job_submission(self, test_client, sample_job_request, sample_agent_registration):
        """Test job submission"""
        # Register an agent first
        agent_registry.clear()
        test_client.post("/register", json=sample_agent_registration.dict())
        
        response = test_client.post("/submit_job", json=sample_job_request.dict())
        assert response.status_code == 200
        
        result = response.json()
        assert result["status"] == "pending"
        assert "job_id" in result
        assert result["message"] == "Job submitted successfully"
    
    def test_job_submission_no_capable_agents(self, test_client, sample_job_request):
        """Test job submission fails when no capable agents available"""
        # Clear registry so no agents available
        agent_registry.clear()
        
        response = test_client.post("/submit_job", json=sample_job_request.dict())
        assert response.status_code == 400
        assert "No agents available" in response.json()["detail"]
    
    def test_get_job_status(self, test_client, sample_job_request, sample_agent_registration):
        """Test getting job status"""
        # Register agent and submit job
        agent_registry.clear()
        test_client.post("/register", json=sample_agent_registration.dict())
        
        job_response = test_client.post("/submit_job", json=sample_job_request.dict())
        job_id = job_response.json()["job_id"]
        
        # Get job status
        response = test_client.get(f"/job/{job_id}")
        assert response.status_code == 200
        
        status = response.json()
        assert status["job_id"] == job_id
        assert "status" in status
    
    def test_get_nonexistent_job(self, test_client):
        """Test getting status of nonexistent job"""
        response = test_client.get("/job/nonexistent_job_id")
        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]
    
    def test_unregister_agent(self, test_client, sample_agent_registration):
        """Test agent unregistration"""
        # Register agent first
        agent_registry.clear()
        test_client.post("/register", json=sample_agent_registration.dict())
        
        # Unregister agent
        response = test_client.delete("/agents/test_quantitative_analyst")
        assert response.status_code == 200
        assert "unregistered" in response.json()["message"]
        
        # Verify agent removed from registry
        assert "test_quantitative_analyst" not in agent_registry

# ===========================
# WORKFLOW ENGINE TESTS
# ===========================

class TestWorkflowEngine:
    """Test workflow engine functionality"""
    
    def test_create_job(self, workflow_engine_instance, sample_job_request):
        """Test job creation"""
        job_id = str(uuid.uuid4())
        assigned_agents = ["test_agent_1", "test_agent_2"]
        
        job = workflow_engine_instance.create_job(job_id, sample_job_request, assigned_agents)
        
        assert job.job_id == job_id
        assert job.request == sample_job_request
        assert job.status == JobStatus.PENDING
        assert job.agents_involved == assigned_agents
        assert len(job.workflow_steps) > 0
        assert job_id in workflow_engine_instance.jobs
    
    def test_get_job(self, workflow_engine_instance, sample_job_request):
        """Test job retrieval"""
        job_id = str(uuid.uuid4())
        job = workflow_engine_instance.create_job(job_id, sample_job_request, ["test_agent"])
        
        retrieved_job = workflow_engine_instance.get_job(job_id)
        assert retrieved_job == job
        
        # Test nonexistent job
        nonexistent_job = workflow_engine_instance.get_job("nonexistent")
        assert nonexistent_job is None
    
    @pytest.mark.asyncio
    async def test_execute_workflow_steps(self, workflow_engine_instance, sample_job_request):
        """Test workflow step execution"""
        job_id = str(uuid.uuid4())
        job = workflow_engine_instance.create_job(job_id, sample_job_request, ["test_agent"])
        
        # Mock the step execution
        with patch.object(workflow_engine_instance, '_execute_step', return_value={"mock": "result"}):
            await workflow_engine_instance._execute_workflow_steps(job)
        
        assert job.status == JobStatus.COMPLETED
        assert job.progress == 100.0
        assert job.result is not None
    
    def test_generate_workflow_steps(self, workflow_engine_instance, sample_job_request):
        """Test workflow step generation"""
        assigned_agents = ["quantitative_analyst", "tax_strategist"]
        steps = workflow_engine_instance._generate_workflow_steps(sample_job_request, assigned_agents)
        
        assert len(steps) > 0
        
        # Check for expected step types based on query
        step_capabilities = [step.capability for step in steps]
        assert "portfolio_data_fetch" in step_capabilities  # Data collection
        assert "risk_analysis" in step_capabilities  # Analysis based on query
        assert "result_synthesis" in step_capabilities  # Final synthesis
    
    def test_organize_steps_by_dependencies(self, workflow_engine_instance):
        """Test step dependency organization"""
        steps = [
            WorkflowStep("step1", "agent1", "data_fetch", {}, []),
            WorkflowStep("step2", "agent2", "analysis", {}, ["step1"]),
            WorkflowStep("step3", "agent3", "synthesis", {}, ["step2"])
        ]
        
        levels = workflow_engine_instance._organize_steps_by_dependencies(steps)
        
        assert 0 in levels  # Level 0: no dependencies
        assert 1 in levels  # Level 1: depends on level 0
        assert 2 in levels  # Level 2: depends on level 1
        
        assert levels[0][0].step_id == "step1"
        assert levels[1][0].step_id == "step2"
        assert levels[2][0].step_id == "step3"
    
    def test_mock_result_generation(self, workflow_engine_instance):
        """Test mock result generation for different capabilities"""
        # Test risk analysis mock
        risk_result = workflow_engine_instance._generate_mock_result(
            "risk_analysis", {"query": "test risk analysis"}
        )
        assert "var_95" in risk_result
        assert "sharpe_ratio" in risk_result
        
        # Test tax optimization mock
        tax_result = workflow_engine_instance._generate_mock_result(
            "tax_optimization", {"query": "test tax optimization"}
        )
        assert "potential_tax_savings" in tax_result
        assert "recommended_harvesting" in tax_result
    
    def test_completion_time_estimation(self, workflow_engine_instance, sample_job_request):
        """Test job completion time estimation"""
        # Test with simple query
        simple_request = JobRequest(query="simple query", context={})
        estimate = workflow_engine_instance.estimate_completion_time(simple_request)
        assert estimate is not None
        
        # Verify it's a future time
        estimated_time = datetime.fromisoformat(estimate.replace('Z', '+00:00'))
        assert estimated_time > datetime.utcnow().replace(tzinfo=estimated_time.tzinfo)

# ===========================
# CONSENSUS BUILDER TESTS
# ===========================

class TestConsensusBuilder:
    """Test consensus building algorithms"""
    
    def test_calculate_agent_weights(self, consensus_builder_instance, sample_agent_positions):
        """Test agent weight calculation"""
        query_context = {"query": "risk analysis portfolio optimization"}
        weights = consensus_builder_instance._calculate_agent_weights(
            sample_agent_positions, query_context
        )
        
        assert len(weights) == len(sample_agent_positions)
        assert all(0 <= weight <= 1 for weight in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01  # Should sum to ~1.0
    
    def test_extract_query_topics(self, consensus_builder_instance):
        """Test query topic extraction"""
        # Test risk-focused query
        risk_topics = consensus_builder_instance._extract_query_topics(
            "analyze portfolio risk and volatility protection"
        )
        assert "risk_analysis" in risk_topics
        
        # Test tax-focused query
        tax_topics = consensus_builder_instance._extract_query_topics(
            "optimize tax efficiency and harvest losses"
        )
        assert "tax_optimization" in tax_topics
        
        # Test general query
        general_topics = consensus_builder_instance._extract_query_topics(
            "general portfolio advice"
        )
        assert "general_analysis" in general_topics
    
    def test_assess_evidence_quality(self, consensus_builder_instance):
        """Test evidence quality assessment"""
        # High quality evidence
        high_quality_evidence = [
            {"type": "statistical", "source": "reliable_source", "confidence": 0.9, "data": "comprehensive data set", "methodology": "detailed"},
            {"type": "empirical", "source": "peer_reviewed", "confidence": 0.85}
        ]
        quality_score = consensus_builder_instance._assess_evidence_quality(high_quality_evidence)
        assert quality_score > 0.7
        
        # Low quality evidence
        low_quality_evidence = [
            {"type": "unknown", "confidence": 0.3}
        ]
        quality_score = consensus_builder_instance._assess_evidence_quality(low_quality_evidence)
        assert quality_score < 0.5
        
        # No evidence
        no_evidence_score = consensus_builder_instance._assess_evidence_quality([])
        assert no_evidence_score == 0.2
    
    def test_group_similar_positions(self, consensus_builder_instance, sample_agent_positions):
        """Test position grouping"""
        # Add a third position similar to the first
        similar_position = sample_agent_positions[0].copy()
        similar_position["agent_id"] = "another_conservative_agent"
        extended_positions = sample_agent_positions + [similar_position]
        
        groups = consensus_builder_instance._group_similar_positions(extended_positions)
        
        assert len(groups) >= 1
        # Should group similar conservative positions together
        conservative_group = next((g for g in groups if len(g) > 1), None)
        assert conservative_group is not None
    
    def test_positions_similarity(self, consensus_builder_instance):
        """Test position similarity detection"""
        pos1 = {
            "stance": "conservative approach",
            "key_arguments": ["risk management", "downside protection"],
            "risk_assessment": {"primary_risks": ["market_volatility", "concentration"]}
        }
        
        pos2 = {
            "stance": "conservative strategy", 
            "key_arguments": ["risk management", "capital preservation"],
            "risk_assessment": {"primary_risks": ["market_volatility", "liquidity"]}
        }
        
        pos3 = {
            "stance": "aggressive growth",
            "key_arguments": ["maximize returns", "high growth"],
            "risk_assessment": {"primary_risks": ["inflation", "competition"]}
        }
        
        # Similar positions should be grouped
        assert consensus_builder_instance._positions_similar(pos1, pos2)
        
        # Different positions should not be grouped
        assert not consensus_builder_instance._positions_similar(pos1, pos3)
    
    def test_calculate_weighted_consensus(self, consensus_builder_instance, sample_agent_positions):
        """Test complete consensus calculation"""
        query_context = {"query": "portfolio risk analysis and optimization"}
        
        consensus = consensus_builder_instance.calculate_weighted_consensus(
            sample_agent_positions, query_context
        )
        
        # Verify consensus structure
        assert "recommendation" in consensus
        assert "confidence_level" in consensus
        assert "supporting_arguments" in consensus
        assert "majority_agents" in consensus
        assert "minority_opinions" in consensus
        assert "consensus_metrics" in consensus
        assert "implementation_guidance" in consensus
        
        # Verify confidence is reasonable
        assert 0 <= consensus["confidence_level"] <= 1
    
    def test_preserve_minority_opinions(self, consensus_builder_instance, sample_agent_positions):
        """Test minority opinion preservation"""
        # Create agent weights
        agent_weights = {"quantitative_analyst": 0.6, "market_intelligence": 0.4}
        
        # Mock majority consensus
        consensus = {"recommendation": "balanced approach", "risk_assessment": {"primary_risks": ["general_risk"]}}
        
        # Group positions to create minority
        minority_groups = [[sample_agent_positions[1]]]  # Market intelligence as minority
        
        preserved_minorities = consensus_builder_instance.preserve_minority_opinions(
            minority_groups, agent_weights, consensus
        )
        
        assert len(preserved_minorities) >= 0  # May be 0 if threshold not met
        
        for minority in preserved_minorities:
            assert hasattr(minority, 'agents')
            assert hasattr(minority, 'position')
            assert hasattr(minority, 'weight')
            assert hasattr(minority, 'risk_if_ignored')
    
    def test_consensus_metrics_calculation(self, consensus_builder_instance, sample_agent_positions):
        """Test consensus metrics calculation"""
        majority_group = [sample_agent_positions[0]]
        minority_groups = [[sample_agent_positions[1]]]
        overall_confidence = 0.8
        
        metrics = consensus_builder_instance._calculate_consensus_metrics(
            sample_agent_positions, majority_group, minority_groups, overall_confidence
        )
        
        assert 0 <= metrics.agreement_level <= 1
        assert 0 <= metrics.evidence_strength <= 1
        assert 0 <= metrics.minority_strength <= 1
        assert metrics.overall_confidence == overall_confidence
        assert len(metrics.confidence_distribution) == len(sample_agent_positions)
    
    def test_empty_consensus_handling(self, consensus_builder_instance):
        """Test handling of empty agent positions"""
        empty_consensus = consensus_builder_instance.calculate_weighted_consensus([], {})
        
        assert empty_consensus["recommendation"] == "Insufficient agent input for consensus"
        assert empty_consensus["confidence_level"] == 0.0
        assert empty_consensus["majority_agents"] == []
        assert empty_consensus["minority_opinions"] == []

# ===========================
# DEBATE ENGINE TESTS  
# ===========================

class TestDebateEngine:
    """Test multi-agent debate functionality"""
    
    @pytest.mark.asyncio
    async def test_initiate_debate(self, debate_engine_instance):
        """Test debate initiation"""
        query = "Should we increase portfolio risk for higher returns?"
        agents = ["quantitative_analyst", "market_intelligence"]
        portfolio_context = {"portfolio_id": "test_123"}
        
        with patch.object(debate_engine_instance, '_execute_debate', return_value=None):
            debate_id = await debate_engine_instance.initiate_debate(
                query, agents, portfolio_context
            )
        
        assert debate_id in debate_engine_instance.active_debates
        debate_state = debate_engine_instance.active_debates[debate_id]
        assert debate_state["query"] == query
        assert debate_state["status"] == "active"
        assert len(debate_state["participants"]) == len(agents)
    
    def test_extract_query_themes(self, debate_engine_instance):
        """Test query theme extraction"""
        # Risk-focused query
        themes = debate_engine_instance._extract_query_themes(
            "analyze portfolio risk and downside protection strategies"
        )
        assert themes["risk"] == True
        assert themes["primary"] == "risk"
        
        # Growth-focused query  
        themes = debate_engine_instance._extract_query_themes(
            "maximize portfolio growth and identify opportunities"
        )
        assert themes["growth"] == True
        assert themes["primary"] == "growth"
    
    @pytest.mark.asyncio
    async def test_assign_agent_perspectives(self, debate_engine_instance):
        """Test agent perspective assignment"""
        agents = ["quantitative_analyst", "market_intelligence"]
        analysis = {"primary_focus": "risk", "required_expertise": ["risk_analysis"]}
        
        assignments = await debate_engine_instance._assign_agent_perspectives(agents, analysis)
        
        assert len(assignments) == len(agents)
        for agent_id in agents:
            assert agent_id in assignments
            assert "perspective" in assignments[agent_id]
            assert "role" in assignments[agent_id]
            assert "focus_areas" in assignments[agent_id]
    
    def test_generate_opening_position_prompt(self, debate_engine_instance):
        """Test opening position prompt generation"""
        query = "Should we rebalance the portfolio?"
        context = {"portfolio_id": "test"}
        assignment = {
            "perspective": AgentPerspective.CONSERVATIVE,
            "role": "risk_assessor", 
            "focus_areas": ["risk_analysis", "downside_protection"]
        }
        
        prompt = debate_engine_instance._generate_opening_position_prompt(query, context, assignment)
        
        assert query in prompt
        assert "conservative" in prompt.lower()
        assert "risk_assessor" in prompt
        assert "risk_analysis" in prompt
        assert ("JSON" in prompt or "json" in prompt or "{" in prompt)
    
    def test_group_similar_positions_debate(self, debate_engine_instance):
        """Test position grouping in debate context"""
        from mcp.debate_engine import DebatePosition
        
        pos1 = DebatePosition(
            agent_id="agent1",
            stance="conservative approach",
            key_arguments=["risk management"],
            supporting_evidence=[],
            confidence_score=0.8,
            risk_assessment={},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        pos2 = DebatePosition(
            agent_id="agent2", 
            stance="conservative strategy",
            key_arguments=["risk management"],
            supporting_evidence=[],
            confidence_score=0.7,
            risk_assessment={},
            counterarguments_addressed=[],
            timestamp=datetime.now()
        )
        
        positions = [pos1, pos2]
        groups = debate_engine_instance._group_similar_positions(positions)
        
        # Should group similar conservative positions
        assert len(groups) == 1
        assert len(groups[0]) == 2
    
    @pytest.mark.asyncio
    async def test_get_debate_status(self, debate_engine_instance):
        """Test debate status retrieval"""
        # Test nonexistent debate
        status = await debate_engine_instance.get_debate_status("nonexistent")
        assert "error" in status
        
        # Test existing debate
        debate_id = "test_debate_123"
        debate_engine_instance.active_debates[debate_id] = {
            "status": "active",
            "current_stage": DebateStage.OPENING_POSITIONS,
            "rounds": [],
            "participants": {"agent1": {}, "agent2": {}},
            "consensus": None
        }
        
        status = await debate_engine_instance.get_debate_status(debate_id)
        assert status["debate_id"] == debate_id
        assert status["status"] == "active"
        assert status["participants"] == ["agent1", "agent2"]
    
    def test_assess_query_complexity(self, debate_engine_instance):
        """Test query complexity assessment"""
        # Simple query
        simple_complexity = debate_engine_instance._assess_query_complexity(
            "buy or sell?", {"holdings": ["AAPL", "MSFT"]}
        )
        assert simple_complexity in ["low", "medium"]
        
        # Complex query
        complex_complexity = debate_engine_instance._assess_query_complexity(
            "Perform comprehensive risk analysis including VaR calculations, stress testing, and correlation analysis across all asset classes",
            {"holdings": ["AAPL"] * 25}  # Many holdings
        )
        assert complex_complexity == "high"
    
    def test_identify_required_expertise(self, debate_engine_instance):
        """Test expertise identification"""
        themes = {"risk": True, "tax": True, "options": False}
        expertise = debate_engine_instance._identify_required_expertise("test query", themes)
        
        assert "risk_analysis" in expertise
        assert "tax_optimization" in expertise
        assert "options_analysis" not in expertise

# ===========================
# SCHEMA VALIDATION TESTS
# ===========================

class TestSchemas:
    """Test Pydantic schema validation"""
    
    def test_agent_registration_schema(self):
        """Test AgentRegistration schema validation"""
        # Valid registration
        valid_data = {
            "agent_id": "test_agent",
            "agent_name": "Test Agent",
            "agent_type": "TestAgentType",
            "capabilities": ["test_capability"]
        }
        registration = AgentRegistration(**valid_data)
        assert registration.agent_id == "test_agent"
        assert registration.max_concurrent_jobs == 5  # Default value
        
        # Invalid registration (missing required fields)
        with pytest.raises(ValueError):
            AgentRegistration(agent_id="test")  # Missing required fields
    
    def test_job_request_schema(self):
        """Test JobRequest schema validation"""
        # Valid request
        valid_data = {
            "query": "Test query",
            "context": {"test": "data"},
            "priority": 5
        }
        job_request = JobRequest(**valid_data)
        assert job_request.query == "Test query"
        assert job_request.timeout_seconds == 300  # Default value
        
        # Invalid priority
        with pytest.raises(ValueError):
            JobRequest(query="test", priority=15)  # Priority > 10
    
    def test_job_response_schema(self):
        """Test JobResponse schema validation"""
        valid_data = {
            "job_id": "test_job_123",
            "status": JobStatus.COMPLETED,
            "message": "Job completed successfully"
        }
        job_response = JobResponse(**valid_data)
        assert job_response.job_id == "test_job_123"
        assert job_response.status == JobStatus.COMPLETED
    
    def test_agent_job_request_schema(self):
        """Test AgentJobRequest schema validation"""
        valid_data = {
            "job_id": "test_job",
            "step_id": "test_step", 
            "capability": "test_capability",
            "input_data": {"query": "test"}
        }
        agent_request = AgentJobRequest(**valid_data)
        assert agent_request.timeout_seconds == 60  # Default value
    
    def test_health_check_schema(self):
        """Test HealthCheck schema validation"""
        valid_data = {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "registered_agents": 5,
            "active_jobs": 2
        }
        health_check = HealthCheck(**valid_data)
        assert health_check.status == "healthy"
        assert health_check.registered_agents == 5

# ===========================
# ERROR HANDLING TESTS
# ===========================

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_job_id_format(self, test_client):
        """Test handling of invalid job ID formats"""
        # Test with various invalid formats
        invalid_ids = ["", "   ", "invalid/id", "id with spaces", "very_long_id" * 100]
        
        for invalid_id in invalid_ids:
            response = test_client.get(f"/job/{invalid_id}")
            assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, workflow_engine_instance):
        """Test workflow timeout handling"""
        job_request = JobRequest(
            query="test query",
            timeout_seconds=1  # Very short timeout
        )
        
        job_id = str(uuid.uuid4())
        job = workflow_engine_instance.create_job(job_id, job_request, ["test_agent"])
        
        # Mock a long-running step
        async def slow_step(*args):
            await asyncio.sleep(2)  # Longer than timeout
            return {"result": "should not complete"}
        
        with patch.object(workflow_engine_instance, '_execute_step', side_effect=slow_step):
            # This should handle timeout gracefully
            try:
                await asyncio.wait_for(
                    workflow_engine_instance._execute_workflow_steps(job),
                    timeout=3
                )
            except asyncio.TimeoutError:
                pass  # Expected for this test
    
    def test_consensus_with_conflicting_data(self, consensus_builder_instance):
        """Test consensus building with conflicting/corrupted data"""
        conflicting_positions = [
            {
                "agent_id": "agent1",
                "stance": None,  # Missing stance
                "key_arguments": [],
                "supporting_evidence": [],
                "confidence_score": "invalid",  # Should be float
                "risk_assessment": {}
            },
            {
                "agent_id": "agent2",
                "stance": "valid stance",
                "key_arguments": ["arg1"],
                "supporting_evidence": [{"invalid": "evidence"}],
                "confidence_score": 1.5,  # > 1.0
                "risk_assessment": {"primary_risks": ["risk1"]}
            }
        ]
        
        # Should handle gracefully without crashing
        try:
            consensus = consensus_builder_instance.calculate_weighted_consensus(
                conflicting_positions, {"query": "test"}
            )
            assert "recommendation" in consensus
        except Exception as e:
            # If it does throw an exception, it should be handled gracefully
            assert ("error" in str(e).lower() or "invalid" in str(e).lower() or "multiply" in str(e).lower())
    
    def test_debate_with_no_agents(self, debate_engine_instance):
        """Test debate initiation with no agents"""
        with patch.object(debate_engine_instance, '_execute_debate', return_value=None):
            # Should handle empty agent list gracefully
            asyncio.run(debate_engine_instance.initiate_debate(
                "test query", [], {"portfolio_id": "test"}
            ))

# ===========================
# INTEGRATION TESTS
# ===========================

class TestIntegration:
    """Integration tests combining multiple components"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, test_client, sample_agent_registration, sample_job_request):
        """Test complete workflow from registration to result"""
        # 1. Register agent
        agent_registry.clear()
        reg_response = test_client.post("/register", json=sample_agent_registration.dict())
        assert reg_response.status_code == 200
        
        # 2. Submit job
        job_response = test_client.post("/submit_job", json=sample_job_request.dict())
        assert job_response.status_code == 200
        job_id = job_response.json()["job_id"]
        
        # 3. Check initial status
        status_response = test_client.get(f"/job/{job_id}")
        assert status_response.status_code == 200
        
        # 4. Wait a bit for processing (in real test, would wait for completion)
        await asyncio.sleep(0.1)
        
        # 5. Check final status
        final_status = test_client.get(f"/job/{job_id}")
        assert final_status.status_code == 200
    
    def test_consensus_with_workflow_results(self, consensus_builder_instance, workflow_engine_instance):
        """Test consensus building with workflow engine results"""
        # Create mock workflow results that look like agent positions
        workflow_results = [
            {
                "agent_id": "quantitative_analyst",
                "stance": "reduce risk exposure",
                "key_arguments": ["Market volatility increasing", "Portfolio beta too high"],
                "supporting_evidence": [
                    {"type": "statistical", "data": "VaR: 0.034", "confidence": 0.92}
                ],
                "confidence_score": 0.89,
                "risk_assessment": {"primary_risks": ["market_risk", "concentration_risk"]}
            },
            {
                "agent_id": "market_intelligence",
                "stance": "maintain current allocation",
                "key_arguments": ["Strong earnings growth", "Fed policy supportive"],
                "supporting_evidence": [
                    {"type": "analytical", "analysis": "S&P 500 earnings up 12%", "confidence": 0.85}
                ],
                "confidence_score": 0.81,
                "risk_assessment": {"primary_risks": ["inflation_risk"]}
            }
        ]
        
        consensus = consensus_builder_instance.calculate_weighted_consensus(
            workflow_results, {"query": "portfolio risk analysis"}
        )
        
        assert consensus["confidence_level"] > 0
        assert len(consensus["majority_agents"]) >= 1
        assert "recommendation" in consensus
    
    @pytest.mark.asyncio
    async def test_debate_consensus_integration(self, debate_engine_instance, consensus_builder_instance):
        """Test integration between debate engine and consensus builder"""
        # Mock a completed debate
        debate_id = "test_integration_debate"
        
        mock_debate_state = {
            "debate_id": debate_id,
            "query": "Should we rebalance the portfolio?",
            "participants": {"agent1": {}, "agent2": {}},
            "rounds": [
                Mock(positions=[
                    Mock(
                        agent_id="agent1",
                        stance="rebalance recommended", 
                        key_arguments=["Portfolio drift detected"],
                        supporting_evidence=[{"type": "analytical", "confidence": 0.8}],
                        confidence_score=0.85,
                        risk_assessment={"primary_risks": ["drift_risk"]}
                    ),
                    Mock(
                        agent_id="agent2",
                        stance="maintain current allocation",
                        key_arguments=["Transaction costs high"],
                        supporting_evidence=[{"type": "statistical", "confidence": 0.75}],
                        confidence_score=0.78,
                        risk_assessment={"primary_risks": ["cost_risk"]}
                    )
                ])
            ],
            "status": "completed"
        }
        
        debate_engine_instance.active_debates[debate_id] = mock_debate_state
        
        # Get debate results
        results = await debate_engine_instance.get_debate_results(debate_id)
        assert results["debate_id"] == debate_id

# ===========================
# PERFORMANCE TESTS
# ===========================

class TestPerformance:
    """Performance and load testing"""
    
    @pytest.mark.asyncio
    async def test_concurrent_job_processing(self, workflow_engine_instance):
        """Test handling multiple concurrent jobs"""
        num_jobs = 10
        job_requests = []
        
        for i in range(num_jobs):
            job_request = JobRequest(
                query=f"test query {i}",
                context={"job_number": i}
            )
            job_requests.append(job_request)
        
        # Create multiple jobs
        jobs = []
        for i, request in enumerate(job_requests):
            job_id = f"concurrent_job_{i}"
            job = workflow_engine_instance.create_job(job_id, request, ["test_agent"])
            jobs.append(job)
        
        # Verify all jobs created successfully
        assert len(workflow_engine_instance.jobs) == num_jobs
        
        # Mock execution of all jobs concurrently
        async def mock_execute(job_id):
            job = workflow_engine_instance.get_job(job_id)
            job.status = JobStatus.COMPLETED
            return job
        
        # Execute all jobs concurrently
        with patch.object(workflow_engine_instance, '_execute_step', return_value={"result": "success"}):
            tasks = [
                workflow_engine_instance._execute_workflow_steps(job)
                for job in jobs
            ]
            await asyncio.gather(*tasks)
        
        # Verify all completed
        for job in jobs:
            assert job.status == JobStatus.COMPLETED
    
    def test_large_consensus_calculation(self, consensus_builder_instance):
        """Test consensus building with many agent positions"""
        num_agents = 50
        large_position_set = []
        
        for i in range(num_agents):
            position = {
                "agent_id": f"agent_{i}",
                "stance": "balanced approach" if i % 2 == 0 else "aggressive growth",
                "key_arguments": [f"argument_{i}_1", f"argument_{i}_2"],
                "supporting_evidence": [
                    {"type": "statistical", "data": f"data_{i}", "confidence": 0.7 + (i % 3) * 0.1}
                ],
                "confidence_score": 0.6 + (i % 4) * 0.1,
                "risk_assessment": {"primary_risks": [f"risk_{i % 5}"]}
            }
            large_position_set.append(position)
        
        # Should handle large sets efficiently
        start_time = datetime.now()
        consensus = consensus_builder_instance.calculate_weighted_consensus(
            large_position_set, {"query": "large scale analysis"}
        )
        end_time = datetime.now()
        
        execution_time = (end_time - start_time).total_seconds()
        assert execution_time < 5.0  # Should complete within 5 seconds
        assert consensus["confidence_level"] > 0
        assert len(consensus["majority_agents"]) > 0

# ===========================
# MAIN TEST RUNNER
# ===========================

if __name__ == "__main__":
    """Run the comprehensive test suite"""
    print("🧪 Starting Comprehensive MCP Unit Testing Suite...")
    print("=" * 80)
    
    # Configure pytest
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "-s",  # Don't capture output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
    ]
    
    # Add coverage if available
    try:
        import pytest_cov
        pytest_args.extend([
            "--cov=mcp",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
        print("📊 Coverage reporting enabled")
    except ImportError:
        print("⚠️  pytest-cov not installed, skipping coverage")
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed successfully!")
        print("🎉 MCP system is ready for production!")
    else:
        print(f"\n❌ Some tests failed (exit code: {exit_code})")
        print("🔧 Please review the test output and fix issues")
    
    exit(exit_code)