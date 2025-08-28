"""
Enhanced Financial Orchestrator
Advanced orchestration with multi-stage workflows and contextual intelligence
"""

import re
import json
import uuid
from typing import Dict, Any, Set, Optional, List, Union
from datetime import datetime
from enum import Enum
import logging
import asyncio
import httpx

# Test imports individually
def test_agent_imports():
    """Test each agent import individually to identify failures"""
    import_results = {}
    
    agents_to_test = [
        ("QuantitativeAnalystAgent", "agents.quantitative_analyst"),
        ("StrategyArchitectAgent", "agents.strategy_architect"),
        ("StrategyRebalancingAgent", "agents.strategy_rebalancing"),
        ("HedgingStrategistAgent", "agents.hedging_strategist"),
        ("StrategyBacktesterAgent", "agents.strategy_backtester"),
        ("FinancialTutorAgent", "agents.financial_tutor"),
        ("RegimeForecastingAgent", "agents.regime_forecasting_agent"),
        ("BehavioralFinanceAgent", "agents.behavioral_finance_agent"),
        ("ScenarioSimulationAgent", "agents.scenario_simulation_agent"),
        ("SecurityScreenerAgent", "agents.security_screener_agent"),
    ]
    
    for agent_name, module_path in agents_to_test:
        try:
            module = __import__(module_path, fromlist=[agent_name])
            agent_class = getattr(module, agent_name)
            import_results[agent_name] = {"success": True, "class": agent_class}
            print(f"✓ {agent_name} import successful")
        except Exception as e:
            import_results[agent_name] = {"success": False, "error": str(e)}
            print(f"✗ {agent_name} import failed: {e}")
    
    return import_results

# Run import tests
import_results = test_agent_imports()

# Only import successfully tested agents
successful_imports = {}
for agent_name, result in import_results.items():
    if result["success"]:
        successful_imports[agent_name] = result["class"]

# Backend Imports
try:
    from db import crud
    print("✓ CRUD import successful")
except Exception as e:
    print(f"✗ CRUD import failed: {e}")

try:
    from core.data_handler import get_market_data_for_portfolio
    print("✓ Data handler import successful")
except Exception as e:
    print(f"✗ Data handler import failed: {e}")

try:
    import anthropic
    from core.config import settings
    print("✓ Anthropic and settings import successful")
except Exception as e:
    print(f"✗ Anthropic/settings import failed: {e}")

logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """Multi-stage workflow states"""
    INITIAL_ANALYSIS = "initial_analysis"
    SCREENING = "screening"
    DEEP_ANALYSIS = "deep_analysis"
    STRATEGY_FORMATION = "strategy_formation"
    RISK_ASSESSMENT = "risk_assessment"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    FINAL_SYNTHESIS = "final_synthesis"

class ContextMode(Enum):
    """Context awareness modes"""
    AUTO = "auto"
    PORTFOLIO = "portfolio"
    MARKET = "market"
    MINIMAL = "minimal"
    COMPREHENSIVE = "comprehensive"

class EnhancedFinancialOrchestrator:
    """Enhanced orchestrator with multi-stage workflows and contextual intelligence"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8001"):
        # Initialize basic attributes first
        self.agents = {}
        self.mcp_agents = {}
        self.agent_capabilities = {}
        self.active_workflows = {}
        self.conversation_history = {}
        self.mcp_server_url = mcp_server_url
        
        try:
            self.client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            print("✓ Anthropic client initialized")
        except Exception as e:
            print(f"✗ Anthropic client initialization failed: {e}")
            self.client = None
        
        # Initialize agents safely using successful imports
        self._initialize_agents_safely()
        
        # MCP-enabled agents mapping
        self.mcp_agents = {
            "security_screener": "security_screener",  # Maps to MCP agent ID
        }
        
        # Initialize agent capabilities
        self._initialize_agent_capabilities()
        
        logger.info(f"EnhancedFinancialOrchestrator initialized with {len(self.agents) + len(self.mcp_agents)} agents")

    def _initialize_agents_safely(self):
        """Initialize agents with error handling for each agent"""
        agent_mapping = {
            "quantitative_analyst": "QuantitativeAnalystAgent",
            "strategy_architect": "StrategyArchitectAgent", 
            "strategy_rebalancing": "StrategyRebalancingAgent",
            "hedging_strategist": "HedgingStrategistAgent",
            "strategy_backtester": "StrategyBacktesterAgent",
            "financial_tutor": "FinancialTutorAgent",
            "regime_forecasting": "RegimeForecastingAgent",
            "behavioral_finance": "BehavioralFinanceAgent",
            "scenario_simulation": "ScenarioSimulationAgent",
        }
        
        for agent_key, agent_class_name in agent_mapping.items():
            try:
                if agent_class_name in successful_imports:
                    agent_class = successful_imports[agent_class_name]
                    self.agents[agent_key] = agent_class()
                    print(f"✓ {agent_key} initialized successfully")
                else:
                    print(f"✗ {agent_key} skipped - class not available")
            except Exception as e:
                print(f"✗ {agent_key} initialization failed: {e}")
                logger.error(f"Failed to initialize {agent_key}: {str(e)}")

    def _initialize_agent_capabilities(self):
        """Initialize agent capabilities mapping"""
        self.agent_capabilities = {
            "quantitative_analyst": ["risk_analysis", "portfolio_analysis", "quantitative_analysis"],
            "strategy_architect": ["strategy_development", "investment_planning"],
            "strategy_rebalancing": ["portfolio_rebalancing", "allocation_optimization"],
            "hedging_strategist": ["risk_hedging", "options_analysis"],
            "strategy_backtester": ["backtesting", "performance_analysis"],
            "financial_tutor": ["education", "explanation"],
            "regime_forecasting": ["market_regime_analysis", "forecasting"],
            "behavioral_finance": ["behavioral_analysis", "sentiment_analysis"],
            "scenario_simulation": ["scenario_testing", "stress_testing"],
            # MCP agent capabilities
            "security_screener": [
                "security_screening", "factor_analysis", "fundamental_screening",
                "stock_selection", "portfolio_complement_analysis"
            ]
        }

    async def process_contextual_request(
        self,
        message: str,
        user: Any,
        db_session: Any,
        context_mode: str = "auto",
        analysis_depth: str = "standard",
        conversation_id: Optional[str] = None,
        include_market_data: bool = True,
        preferred_agents: Optional[List[str]] = None,
        max_response_time: int = 30
    ) -> Dict[str, Any]:
        """Process a contextual request with enhanced intelligence"""
        
        start_time = datetime.now()
        workflow_id = conversation_id or str(uuid.uuid4())
        
        try:
            # Step 1: Analyze request and build context
            context = await self._build_request_context(
                message, user, db_session, context_mode, include_market_data
            )
            
            # Step 2: Classify query and determine workflow
            classification = await self._classify_query(message, context, analysis_depth)
            
            # Step 3: Execute multi-stage workflow
            workflow_result = await self._execute_workflow(
                message, context, classification, preferred_agents, max_response_time
            )
            
            # Step 4: Synthesize final response
            final_response = await self._synthesize_response(
                workflow_result, context, classification
            )
            
            # Step 5: Extract insights and suggestions
            insights = self._extract_insights(workflow_result, context)
            suggestions = self._generate_smart_suggestions(workflow_result, context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "message": final_response,
                "conversation_id": workflow_id,
                "confidence_score": workflow_result.get("confidence", 0.8),
                "agents_used": workflow_result.get("agents_used", []),
                "context_sources": context.get("sources", []),
                "portfolio_insights": insights.get("portfolio", []),
                "market_insights": insights.get("market", []),
                "smart_suggestions": suggestions,
                "execution_time_seconds": execution_time,
                "workflow_metadata": {
                    "classification": classification,
                    "stages_executed": workflow_result.get("stages", []),
                    "context_mode": context_mode,
                    "analysis_depth": analysis_depth
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing contextual request: {str(e)}")
            return {
                "message": "I encountered an issue processing your request. Please try again.",
                "conversation_id": workflow_id,
                "confidence_score": 0.0,
                "agents_used": [],
                "context_sources": [],
                "error": str(e)
            }

    async def _build_request_context(
        self,
        message: str,
        user: Any,
        db_session: Any,
        context_mode: str,
        include_market_data: bool
    ) -> Dict[str, Any]:
        """Build comprehensive context for the request"""
        
        context = {
            "user_id": user.id,
            "timestamp": datetime.now(),
            "sources": [],
            "market_data": {},
            "portfolio_data": {},
            "user_preferences": {},
            "original_message": message
        }
        
        try:
            # Portfolio context
            if context_mode in ["auto", "portfolio", "comprehensive"]:
                portfolios = crud.get_user_portfolios(db_session, user.id)
                if portfolios:
                    context["portfolio_data"] = {
                        "portfolios": [p.to_dict() for p in portfolios],
                        "count": len(portfolios)
                    }
                    context["sources"].append("user_portfolios")
                    
                    # Add holdings with values for MCP agents
                    holdings_with_values = []
                    for portfolio in portfolios:
                        holdings = crud.get_portfolio_holdings(db_session, portfolio.id)
                        for holding in holdings:
                            holdings_with_values.append({
                                "ticker": holding.symbol,
                                "shares": holding.shares,
                                "market_value": holding.shares * holding.current_price if holding.current_price else 0,
                                "portfolio_id": portfolio.id
                            })
                    
                    context["holdings_with_values"] = holdings_with_values
                    
                    if include_market_data:
                        # Get market data for portfolio holdings
                        for portfolio in portfolios[:3]:  # Limit to first 3 portfolios
                            holdings = crud.get_portfolio_holdings(db_session, portfolio.id)
                            if holdings:
                                market_data = await get_market_data_for_portfolio(holdings)
                                context["market_data"][f"portfolio_{portfolio.id}"] = market_data
                                context["sources"].append(f"market_data_portfolio_{portfolio.id}")
            
            # Market context
            if context_mode in ["auto", "market", "comprehensive"] and include_market_data:
                # Add current market conditions
                context["market_data"]["market_overview"] = await self._get_market_overview()
                context["sources"].append("market_overview")
            
            # User preferences
            user_preferences = crud.get_user_preferences(db_session, user.id)
            if user_preferences:
                context["user_preferences"] = user_preferences.to_dict()
                context["sources"].append("user_preferences")
                
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            
        return context

    async def _classify_query(
        self,
        message: str,
        context: Dict[str, Any],
        analysis_depth: str
    ) -> Dict[str, Any]:
        """Classify the query to determine appropriate workflow"""
        
        # Enhanced pattern matching
        patterns = {
            "portfolio_analysis": [
                r"analyz.*portfolio", r"portfolio.*performance", r"my.*holdings",
                r"portfolio.*risk", r"diversification", r"allocation"
            ],
            "security_screening": [
                r"screen.*securities", r"find.*stocks", r"recommend.*investments",
                r"best.*stocks", r"investment.*opportunities", r"stock.*selection",
                r"complement.*portfolio", r"factor.*analysis", r"fundamental.*screening"
            ],
            "risk_management": [
                r"risk.*assessment", r"hedge.*portfolio", r"protect.*downside",
                r"value.*at.*risk", r"var", r"tail.*risk", r"correlation.*analysis"
            ],
            "strategy_development": [
                r"investment.*strategy", r"trading.*strategy", r"systematic.*approach",
                r"factor.*investing", r"momentum.*strategy", r"mean.*reversion"
            ],
            "market_analysis": [
                r"market.*conditions", r"economic.*outlook", r"sector.*analysis",
                r"regime.*detection", r"market.*forecast", r"technical.*analysis"
            ],
            "educational": [
                r"explain.*", r"how.*work", r"what.*is", r"difference.*between",
                r"learn.*about", r"understand.*", r"tutorial"
            ]
        }
        
        query_type = "general"
        confidence = 0.5
        
        message_lower = message.lower()
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, message_lower):
                    query_type = category
                    confidence = 0.8
                    break
            if confidence > 0.5:
                break
        
        # Determine complexity based on analysis depth and query type
        complexity_mapping = {
            "quick": "low",
            "standard": "medium", 
            "comprehensive": "high"
        }
        
        complexity = complexity_mapping.get(analysis_depth, "medium")
        
        # Determine required agents based on query type
        agent_mapping = {
            "portfolio_analysis": ["quantitative_analyst", "strategy_rebalancing"],
            "security_screening": ["security_screener", "quantitative_analyst"],  # MCP agent first
            "risk_management": ["quantitative_analyst", "hedging_strategist"],
            "strategy_development": ["strategy_architect", "strategy_backtester"],
            "market_analysis": ["regime_forecasting", "scenario_simulation"],
            "educational": ["financial_tutor"],
            "general": ["quantitative_analyst"]
        }
        
        required_agents = agent_mapping.get(query_type, ["quantitative_analyst"])
        
        return {
            "query_type": query_type,
            "complexity": complexity,
            "confidence": confidence,
            "required_agents": required_agents,
            "analysis_depth": analysis_depth
        }

    async def _execute_workflow(
        self,
        message: str,
        context: Dict[str, Any],
        classification: Dict[str, Any],
        preferred_agents: Optional[List[str]],
        max_response_time: int
    ) -> Dict[str, Any]:
        """Execute multi-stage workflow based on classification"""
        
        workflow_result = {
            "stages": [],
            "agents_used": [],
            "responses": {},
            "confidence": 0.8
        }
        
        # Determine agents to use
        agents_to_use = preferred_agents or classification["required_agents"]
        
        # Execute with timeout
        try:
            workflow_result = await asyncio.wait_for(
                self._run_agent_workflow(message, context, agents_to_use, classification),
                timeout=max_response_time
            )
        except asyncio.TimeoutError:
            logger.warning(f"Workflow timed out after {max_response_time} seconds")
            workflow_result["message"] = "Analysis timed out. Please try a simpler request."
            workflow_result["confidence"] = 0.3
        
        return workflow_result

    async def _run_agent_workflow(
        self,
        message: str,
        context: Dict[str, Any],
        agents_to_use: List[str],
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run the actual agent workflow"""
        
        workflow_result = {
            "stages": [],
            "agents_used": [],
            "responses": {},
            "confidence": 0.8
        }
        
        for agent_name in agents_to_use:
            try:
                # Check if this is an MCP agent
                if agent_name in self.mcp_agents:
                    response = await self._execute_mcp_agent(agent_name, message, context)
                elif agent_name in self.agents:
                    response = await self._execute_local_agent(agent_name, message, context, classification)
                else:
                    logger.warning(f"Unknown agent: {agent_name}")
                    continue
                
                workflow_result["responses"][agent_name] = response
                workflow_result["agents_used"].append(agent_name)
                workflow_result["stages"].append(f"executed_{agent_name}")
                
            except Exception as e:
                logger.error(f"Error executing agent {agent_name}: {str(e)}")
                continue
        
        return workflow_result

    async def _execute_mcp_agent(
        self,
        agent_name: str,
        message: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute an MCP agent through the MCP server"""
        
        try:
            async with httpx.AsyncClient() as client:
                # Submit job to MCP server
                job_request = {
                    "query": message,
                    "context": context,
                    "agent_preferences": [self.mcp_agents[agent_name]]
                }
                
                submit_response = await client.post(
                    f"{self.mcp_server_url}/submit_job",
                    json=job_request,
                    timeout=30.0
                )
                
                if submit_response.status_code != 200:
                    raise Exception(f"MCP job submission failed: {submit_response.status_code}")
                
                job_data = submit_response.json()
                job_id = job_data["job_id"]
                
                # Poll for completion
                max_polls = 20
                for _ in range(max_polls):
                    status_response = await client.get(
                        f"{self.mcp_server_url}/job/{job_id}",
                        timeout=5.0
                    )
                    
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        
                        if status_data["status"] == "completed":
                            return status_data.get("result", {})
                        elif status_data["status"] == "failed":
                            raise Exception(f"MCP job failed: {status_data.get('message', 'Unknown error')}")
                    
                    await asyncio.sleep(1.0)
                
                raise Exception("MCP job timed out")
                
        except Exception as e:
            logger.error(f"MCP agent execution failed: {str(e)}")
            return {
                "message": f"Unable to execute {agent_name} analysis at this time.",
                "error": str(e)
            }

    async def _execute_local_agent(
        self,
        agent_name: str,
        message: str,
        context: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a local agent"""
        
        agent = self.agents[agent_name]
        
        # Prepare agent-specific context
        agent_context = {
            "message": message,
            "user_context": context,
            "classification": classification
        }
        
        # Execute agent
        if hasattr(agent, 'process_request'):
            return await agent.process_request(agent_context)
        elif hasattr(agent, 'run'):
            return agent.run(message, context)
        else:
            # Fallback for agents without standard methods
            return {"message": f"{agent_name} processed your request", "agent": agent_name}

    async def _synthesize_response(
        self,
        workflow_result: Dict[str, Any],
        context: Dict[str, Any],
        classification: Dict[str, Any]
    ) -> str:
        """Synthesize final response from agent outputs"""
        
        if not workflow_result.get("responses"):
            return "I was unable to process your request. Please try again."
        
        # If only one agent response, return it directly
        if len(workflow_result["responses"]) == 1:
            agent_name = list(workflow_result["responses"].keys())[0]
            response = workflow_result["responses"][agent_name]
            
            if isinstance(response, dict):
                if "message" in response:
                    return response["message"]
                elif "summary" in response:
                    return response["summary"]
            return str(response)
        
        # Multi-agent synthesis
        responses_text = ""
        for agent_name, response in workflow_result["responses"].items():
            if isinstance(response, dict):
                response_text = response.get("message", response.get("summary", str(response)))
            else:
                response_text = str(response)
            
            responses_text += f"\n\n**{agent_name.replace('_', ' ').title()}:**\n{response_text}"
        
        # Use Claude to synthesize if multiple responses and client is available
        if self.client:
            try:
                synthesis_prompt = f"""
                Synthesize the following agent responses into a coherent, comprehensive answer:
                
                Original Query: {context.get('original_message', 'User query')}
                Query Type: {classification.get('query_type', 'general')}
                
                Agent Responses:{responses_text}
                
                Provide a unified, well-structured response that combines the key insights from all agents.
                """
                
                response = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": synthesis_prompt}]
                )
                
                return response.content[0].text if response.content else responses_text
                
            except Exception as e:
                logger.error(f"Error synthesizing response: {str(e)}")
        
        return responses_text

    def _extract_insights(
        self,
        workflow_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Extract key insights from workflow results"""
        
        insights = {
            "portfolio": [],
            "market": [],
            "strategy": [],
            "risk": []
        }
        
        # Extract insights from agent responses
        for agent_name, response in workflow_result.get("responses", {}).items():
            if isinstance(response, dict):
                if "insights" in response:
                    agent_insights = response["insights"]
                    if isinstance(agent_insights, list):
                        insights["portfolio"].extend(agent_insights)
                elif "recommendations" in response:
                    recommendations = response["recommendations"]
                    if isinstance(recommendations, list):
                        insights["strategy"].extend(recommendations)
        
        return insights

    def _generate_smart_suggestions(
        self,
        workflow_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate smart follow-up suggestions"""
        
        suggestions = []
        
        # Based on agents used, suggest related actions
        agents_used = workflow_result.get("agents_used", [])
        
        if "security_screener" in agents_used:
            suggestions.append({
                "suggestion_type": "analysis",
                "suggestion_text": "Would you like me to analyze the risk profile of these recommended securities?",
                "priority": 8,
                "estimated_time": 15
            })
        
        if "quantitative_analyst" in agents_used:
            suggestions.append({
                "suggestion_type": "action", 
                "suggestion_text": "I can help you backtest this analysis or run scenario simulations.",
                "priority": 7,
                "estimated_time": 30
            })
        
        if "strategy_architect" in agents_used:
            suggestions.append({
                "suggestion_type": "question",
                "suggestion_text": "What's your risk tolerance for implementing this strategy?",
                "priority": 9,
                "estimated_time": 5
            })
        
        return suggestions

    async def _get_market_overview(self) -> Dict[str, Any]:
        """Get current market overview"""
        # Placeholder for market data integration
        return {
            "status": "Retrieved market overview",
            "timestamp": datetime.now().isoformat()
        }

# Export the class
__all__ = ["EnhancedFinancialOrchestrator"]