# agents/orchestrator.py - Enhanced with Multi-Stage Workflow
import re
import json
import uuid
from typing import Dict, Any, Set, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

# Agent Imports
from agents.quantitative_analyst import QuantitativeAnalystAgent
from agents.strategy_architect import StrategyArchitectAgent
from agents.strategy_rebalancing import StrategyRebalancingAgent
from agents.hedging_strategist import HedgingStrategistAgent
from agents.strategy_backtester import StrategyBacktesterAgent
from agents.financial_tutor import FinancialTutorAgent
from agents.regime_forecasting_agent import RegimeForecastingAgent
from agents.behavioral_finance_agent import BehavioralFinanceAgent
from agents.scenario_simulation_agent import ScenarioSimulationAgent
from agents.security_screener_agent import SecurityScreenerAgent
from agents.conversation_manager import AutoGenConversationManager
from agents.tax_strategist_agent import TaxStrategistAgent
from agents.economic_data_agent import EconomicDataAgent 

# NEW: Import CrossAssetAnalyst
from agents.cross_asset_analyst import CrossAssetAnalyst

# Backend Imports
from db import crud
from core.data_handler import get_market_data_for_portfolio
import anthropic
from core.config import settings

# 🚀 NEW: Dynamic Agent Selection Components
class AgentTier(Enum):
    CORE = "core_agents"
    SPECIALIZED = "specialized_agents"
    RESEARCH = "research_agents"

@dataclass
class CommitteeConfig:
    """Configuration for different committee types"""
    name: str
    agents: List[str]
    min_agents: int
    max_agents: int
    required_capabilities: List[str]
    use_case: str

class QueryAnalyzer:
    """Analyzes queries to determine complexity and type"""
    
    def __init__(self):
        self.complexity_keywords = {
            "high": [
                "cross-asset", "correlation", "regime", "crisis", "breakdown",
                "systematic", "tail risk", "contagion", "stress test", "comprehensive"
            ],
            "medium": [
                "risk", "volatility", "diversification", "allocation", "optimization",
                "factor", "beta", "alpha", "sharpe", "var", "actionable"
            ],
            "low": [
                "balance", "simple", "basic", "overview", "summary", "quick"
            ]
        }
    
    def analyze_complexity(self, query: str) -> Tuple[float, str]:
        """Analyze query complexity and return score (0.0-1.0) and reasoning"""
        query_lower = query.lower()
        complexity_score = 0.0
        reasoning_parts = []
        
        # Check for high complexity keywords
        high_matches = sum(1 for keyword in self.complexity_keywords["high"] 
                           if keyword in query_lower)
        if high_matches > 0:
            complexity_score += 0.4 + (high_matches * 0.1)
            reasoning_parts.append(f"High complexity keywords: {high_matches}")
        
        # Check for medium complexity keywords
        medium_matches = sum(1 for keyword in self.complexity_keywords["medium"] 
                             if keyword in query_lower)
        if medium_matches > 0:
            complexity_score += 0.2 + (medium_matches * 0.05)
            reasoning_parts.append(f"Medium complexity keywords: {medium_matches}")
        
        # Check for workflow triggers
        workflow_triggers = ["actionable", "comprehensive", "end-to-end", "full analysis"]
        trigger_matches = sum(1 for trigger in workflow_triggers if trigger in query_lower)
        if trigger_matches > 0:
            complexity_score += 0.3
            reasoning_parts.append("Workflow trigger detected")
        
        # Normalize score
        complexity_score = max(0.0, min(1.0, complexity_score))
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Basic query"
        
        return complexity_score, reasoning

class WorkflowState(Enum):
    AWAITING_STRATEGY = "awaiting_strategy"
    AWAITING_SCREENING = "awaiting_screening" 
    AWAITING_DEEP_ANALYSIS = "awaiting_deep_analysis"
    AWAITING_FINAL_SYNTHESIS = "awaiting_final_synthesis"
    COMPLETE = "complete"
    ERROR = "error"

class WorkflowSession:
    """Manages state for multi-stage workflow execution"""
    
    def __init__(self, session_id: str, user_query: str, user_id: int):
        self.session_id = session_id
        self.user_query = user_query
        self.user_id = user_id
        self.state = WorkflowState.AWAITING_STRATEGY
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Workflow results storage
        self.strategy_result = None
        self.screening_result = None
        self.analysis_result = None
        self.final_synthesis = None
        
        # Execution metadata
        self.steps_completed = []
        self.current_step = 1
        self.total_steps = 4
        self.errors = []
        
    def update_state(self, new_state: WorkflowState, result: Dict = None):
        """Update workflow state and store results"""
        self.state = new_state
        self.updated_at = datetime.now()
        
        if result:
            if new_state == WorkflowState.AWAITING_SCREENING:
                self.strategy_result = result
                self.steps_completed.append("strategy")
                self.current_step = 2
            elif new_state == WorkflowState.AWAITING_DEEP_ANALYSIS:
                self.screening_result = result
                self.steps_completed.append("screening")
                self.current_step = 3
            elif new_state == WorkflowState.AWAITING_FINAL_SYNTHESIS:
                self.analysis_result = result
                self.steps_completed.append("analysis")
                self.current_step = 4
            elif new_state == WorkflowState.COMPLETE:
                self.final_synthesis = result
                self.steps_completed.append("synthesis")
                self.current_step = 4
    
    def get_status_summary(self) -> Dict:
        """Get current workflow status for frontend"""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "progress": {
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "percentage": int((self.current_step / self.total_steps) * 100)
            },
            "steps_completed": self.steps_completed,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

class FinancialOrchestrator:
    """
    Enhanced orchestrator with multi-stage workflow capabilities and dynamic agent selection
    """
    def __init__(self):
        self.roster = {
            "QuantitativeAnalystAgent": QuantitativeAnalystAgent(),
            "StrategyArchitectAgent": StrategyArchitectAgent(),
            "StrategyRebalancingAgent": StrategyRebalancingAgent(),
            "HedgingStrategistAgent": HedgingStrategistAgent(),
            "StrategyBacktesterAgent": StrategyBacktesterAgent(),
            "FinancialTutorAgent": FinancialTutorAgent(),
            "RegimeForecastingAgent": RegimeForecastingAgent(),
            "BehavioralFinanceAgent": BehavioralFinanceAgent(),
            "ScenarioSimulationAgent": ScenarioSimulationAgent(),
            "SecurityScreenerAgent": SecurityScreenerAgent(),
            # 🚀 NEW: Add CrossAssetAnalyst
            "CrossAssetAnalyst": CrossAssetAnalyst(),
            "TaxStrategistAgent": TaxStrategistAgent(),
            "EconomicDataAgent": EconomicDataAgent()
        }

        self.mcp_agents = {
            "QuantitativeAnalystAgent",
            "SecurityScreenerAgent",
            "FinancialTutorAgent",
            "CrossAssetAnalyst",
            "StrategyArchitectAgent",
            "RegimeForecastingAgent",
            "StrategyRebalancingAgent",
            "HedgingStrategistAgent",
            "StrategyBacktesterAgent",
            "ScenarioSimulationAgent",
            "BehavioralFinanceAgent",
            "TaxStrategistAgent",
            "EconomicDataAgent"
        }
        
        # 🚀 NEW: Dynamic capabilities
        self.query_analyzer = QueryAnalyzer()
        self.agent_capabilities = self._initialize_agent_capabilities()
        self.committee_templates = self._initialize_committee_templates()
        
        print(f"FinancialOrchestrator initialized with a team of {len(self.roster)} agents (including Cross-Asset Analyst).")
        self._setup_classification_patterns()
        
        # Initialize Anthropic client
        self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        
        llm_config = {
            "config_list": [{
                "model": "claude-3-5-sonnet-20240620",
                "api_key": settings.ANTHROPIC_API_KEY,
                "api_type": "anthropic",
            }],
            "temperature": 0.2,
        }
        self.conversation_manager = AutoGenConversationManager(llm_config=llm_config)
        
        # 🚀 EXISTING: Workflow session management
        self.active_workflows: Dict[str, WorkflowSession] = {}

    def _initialize_agent_capabilities(self) -> Dict[str, List[str]]:
        """Map agents to their capabilities for intelligent selection"""
        return {
            "QuantitativeAnalystAgent": ["risk_analysis", "statistical_modeling", "portfolio_optimization"],
            "StrategyArchitectAgent": ["strategy_formulation", "investment_planning", "market_analysis"],
            "StrategyRebalancingAgent": ["portfolio_rebalancing", "allocation_optimization", "risk_management"],
            "HedgingStrategistAgent": ["risk_hedging", "volatility_management", "protection_strategies"],
            "StrategyBacktesterAgent": ["strategy_testing", "historical_analysis", "performance_evaluation"],
            "FinancialTutorAgent": ["education", "explanation", "concept_clarification"],
            "RegimeForecastingAgent": ["regime_detection", "market_forecasting", "transition_analysis"],
            "BehavioralFinanceAgent": ["behavioral_analysis", "bias_detection", "psychological_factors"],
            "ScenarioSimulationAgent": {"scenario", "scenarios", "simulate", "simulation", "stress", "crash", "crisis", "resilience", "tail"},
            "SecurityScreenerAgent": ["stock_screening", "factor_analysis", "security_selection"],
            # 🚀 NEW: CrossAssetAnalyst capabilities
            "CrossAssetAnalyst": ["correlation_analysis", "regime_detection", "cross_asset_risk", "diversification_analysis"]
        }

    def _initialize_committee_templates(self) -> Dict[str, CommitteeConfig]:
        """Initialize committee configurations using existing agents"""
        return {
            "standard": CommitteeConfig(
                name="Standard Analysis Committee",
                agents=["QuantitativeAnalystAgent", "StrategyArchitectAgent", "SecurityScreenerAgent"],
                min_agents=3,
                max_agents=4,
                required_capabilities=["risk_analysis", "strategy_formulation", "security_selection"],
                use_case="Basic portfolio analysis and recommendations"
            ),
            "enhanced": CommitteeConfig(
                name="Enhanced Multi-Agent Committee",
                agents=[
                    "QuantitativeAnalystAgent", "StrategyArchitectAgent", "SecurityScreenerAgent",
                    "CrossAssetAnalyst", "RegimeForecastingAgent", "BehavioralFinanceAgent"
                ],
                min_agents=6,
                max_agents=7,
                required_capabilities=[
                    "risk_analysis", "strategy_formulation", "correlation_analysis",
                    "regime_detection", "behavioral_analysis"
                ],
                use_case="Complex multi-factor analysis with cross-asset and behavioral considerations"
            ),
            "crisis": CommitteeConfig(
                name="Crisis Response Committee",
                agents=list(self.roster.keys()), # All agents
                min_agents=8,
                max_agents=11,
                required_capabilities=["risk_analysis", "stress_testing", "scenario_modeling", "correlation_analysis"],
                use_case="Emergency analysis during market stress"
            )
        }
    

    def _setup_classification_patterns(self):
        """Enhanced routing with ALL agents including new MCP agents"""
        self.routing_map = {
            # SecurityScreener patterns
            "SecurityScreenerAgent": {
                frozenset(["find", "stocks"]), frozenset(["recommend", "stocks"]), 
                frozenset(["buy", "what"]), frozenset(["screen"]), frozenset(["complement"]),
                frozenset(["specific", "recommendations"]), frozenset(["which", "stocks"]),
                frozenset(["quality", "stocks"]), frozenset(["value", "stocks"]),
                frozenset(["growth", "stocks"]), frozenset(["diversify", "with"]),
                frozenset(["actionable", "advice"]), frozenset(["stocks", "to", "buy"])
            },
            
            "CrossAssetAnalyst": {
                frozenset(["cross-asset"]), frozenset(["correlation"]), frozenset(["regime"]),
                frozenset(["asset", "class"]), frozenset(["diversification"]), 
                frozenset(["cross", "asset"]), frozenset(["correlation", "breakdown"]),
                frozenset(["regime", "shift"]), frozenset(["asset", "allocation"])
            },
            
            "BehavioralFinanceAgent": {
                frozenset(["bias"]), frozenset(["biases"]), frozenset(["behavior"]), 
                frozenset(["behavioral"]), frozenset(["psychology"]), frozenset(["psychological"]),
                frozenset(["cognitive", "bias"]), frozenset(["investment", "behavior"]),
                frozenset(["decision", "making"]), frozenset(["emotional", "investing"]),
                frozenset(["investor", "psychology"]), frozenset(["behavioral", "patterns"]),
                frozenset(["analyze", "biases"]), frozenset(["identify", "biases"]),
                frozenset(["behavioral", "analysis"]), frozenset(["investment", "psychology"])
            },
            
            "EconomicDataAgent": {
                frozenset(["economic"]), frozenset(["economy"]), frozenset(["economics"]),
                frozenset(["fed", "policy"]), frozenset(["federal", "reserve"]), 
                frozenset(["interest", "rates"]), frozenset(["inflation"]),
                frozenset(["gdp"]), frozenset(["unemployment"]), frozenset(["yield", "curve"]),
                frozenset(["economic", "indicators"]), frozenset(["macro", "economic"]),
                frozenset(["geopolitical"]), frozenset(["monetary", "policy"]),
                frozenset(["economic", "outlook"]), frozenset(["recession"]),
                frozenset(["economic", "analysis"]), frozenset(["macro", "analysis"])
            },
            
            "TaxStrategistAgent": {
                frozenset(["tax"]), frozenset(["taxes"]), frozenset(["tax", "optimization"]),
                frozenset(["tax", "planning"]), frozenset(["tax", "loss"]), frozenset(["tax", "harvesting"]),
                frozenset(["asset", "location"]), frozenset(["tax", "efficient"]),
                frozenset(["after", "tax"]), frozenset(["tax", "strategy"]), 
                frozenset(["wash", "sale"]), frozenset(["tax", "drag"]),
                frozenset(["retirement", "contributions"]), frozenset(["tax", "benefits"]),
                frozenset(["optimize", "taxes"]), frozenset(["year", "end", "tax"]),
                frozenset(["tax", "efficiency"]), frozenset(["charitable", "giving"])
            },
            
            "StrategyRebalancingAgent": {
                frozenset(["rebalance"]), frozenset(["optimize"]), frozenset(["allocation"]), 
                frozenset(["risk", "parity"]), frozenset(["diversify", "risk"])
            },
            
            "StrategyBacktesterAgent": {
                frozenset(["backtest"]), frozenset(["test", "strategy"])
            },
            
            "HedgingStrategistAgent": {
                frozenset(["hedge"]), frozenset(["protect"]), frozenset(["target", "volatility"])
            },
            
            "StrategyArchitectAgent": {
                frozenset(["design"]), frozenset(["find", "strategy"]), frozenset(["recommend"]), frozenset(["momentum"])
            },
            
            "RegimeForecastingAgent": {
                frozenset(["forecast"]), frozenset(["regime", "change"]), frozenset(["transition"])
            },
            
            "ScenarioSimulationAgent": {
                frozenset(["scenario"]), frozenset(["simulate"]), frozenset(["what", "if"]), frozenset(["market", "crash"])
            },
            
            "FinancialTutorAgent": {
                frozenset(["explain"]), frozenset(["what", "is"]), frozenset(["define"]), frozenset(["learn"])
            },
            
            "QuantitativeAnalystAgent": {
                frozenset(["risk"]), frozenset(["report"]), frozenset(["factor"]), frozenset(["alpha"]), 
                frozenset(["cvar"]), frozenset(["analyze"]), frozenset(["stress", "test"]), frozenset(["tail", "risk"])
            }
        }

    def _should_trigger_workflow(self, user_query: str) -> Tuple[bool, str, float]:
        """Enhanced workflow triggering with complexity analysis"""
        
        # Analyze query complexity
        complexity_score, reasoning = self.query_analyzer.analyze_complexity(user_query)
        
        # Existing workflow triggers
        workflow_triggers = [
            "what should i do", "what stocks", "recommend", "advice",
            "portfolio improvement", "investment strategy", "rebalance my portfolio",
            "optimize my", "improve my", "help with my portfolio",
            "full analysis", "complete review", "thorough examination",
            "end-to-end", "comprehensive", "detailed analysis",
            "actionable recommendations", "specific steps", "concrete advice",
            "trading plan", "implementation", "execute"
        ]
        
        query_lower = user_query.lower()
        trigger_match = any(trigger in query_lower for trigger in workflow_triggers)
        
        # Crisis keywords that should trigger workflows
        crisis_keywords = ["crisis", "crash", "emergency", "stress test", "black swan"]
        crisis_match = any(keyword in query_lower for keyword in crisis_keywords)
        
        # Check for single cross-asset queries that should NOT trigger workflows
        single_cross_asset_queries = [
            "analyze cross-asset", "cross-asset correlation", "detect regime", 
            "regime detection", "regime transition", "diversification analysis",
            "correlation analysis", "asset allocation", "cross asset"
        ]
        is_single_cross_asset = any(phrase in query_lower for phrase in single_cross_asset_queries)
        
        # Enhanced logic: Only trigger workflow for truly complex multi-step queries
        should_trigger = (
            trigger_match or 
            (complexity_score > 0.7 and not is_single_cross_asset) or 
            crisis_match
        )
        
        # Determine workflow type based on multiple factors
        if crisis_match or complexity_score > 0.8:
            workflow_type = "crisis"
        elif complexity_score > 0.7 or trigger_match:
            workflow_type = "enhanced"
        else:
            workflow_type = "standard"
        
        print(f"🔍 Workflow analysis: trigger={should_trigger}, type={workflow_type}, complexity={complexity_score:.2f}")
        if complexity_score > 0.5:
            print(f"📊 Complexity reasoning: {reasoning}")

        return should_trigger, workflow_type, complexity_score
    
    async def start_workflow(self, user_query: str, db_session, current_user, workflow_type: str = "auto") -> Dict[str, Any]:
        """Enhanced workflow start with dynamic type selection"""
        
        # Auto-determine workflow type if not specified
        if workflow_type == "auto":
            should_trigger, detected_type, complexity_score = self._should_trigger_workflow(user_query)
            workflow_type = detected_type
        else:
            complexity_score, _ = self.query_analyzer.analyze_complexity(user_query)
        
        # FIX: Ensure a user exists for a workflow session
        if not current_user:
            return {"success": False, "error": "Workflow execution requires a logged-in user."}

        # Create enhanced workflow session
        session_id = str(uuid.uuid4())
        workflow = WorkflowSession(session_id, user_query, current_user.id)
        
        # 🚀 NEW: Add metadata about selection
        workflow.workflow_type = workflow_type
        workflow.complexity_score = complexity_score
        
        self.active_workflows[session_id] = workflow
        
        print(f"🚀 Starting {workflow_type} workflow session: {session_id}")
        print(f"📊 Complexity score: {complexity_score:.2f}")
        print(f"📝 Query: {user_query}")
        
        # Get portfolio context
        portfolio_context = {}
        # FIX: Check for db_session before using it
        if db_session:
            user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
            if user_portfolios:
                primary_portfolio = user_portfolios[0]
                portfolio_context = get_market_data_for_portfolio(primary_portfolio.holdings)
        
        # 🚀 ENHANCED: Choose workflow execution path based on type
        return await self._execute_enhanced_workflow_step(workflow, portfolio_context, db_session)
    
    async def _execute_workflow_step(self, workflow: WorkflowSession, context: Dict, db_session) -> Dict[str, Any]:
        """Execute current workflow step based on state"""
        
        try:
            if workflow.state == WorkflowState.AWAITING_STRATEGY:
                return await self._step_1_strategy_formulation(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_SCREENING:
                return await self._step_2_security_screening(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_DEEP_ANALYSIS:
                return await self._step_3_quantitative_analysis(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_FINAL_SYNTHESIS:
                return self._step_4_final_synthesis(workflow, context)
            
            else:
                raise ValueError(f"Unknown workflow state: {workflow.state}")
                
        except Exception as e:
            print(f"❌ Workflow step failed: {str(e)}")
            workflow.state = WorkflowState.ERROR
            workflow.errors.append(str(e))
            return {
                "success": False,
                "error": f"Workflow step failed: {str(e)}",
                "workflow_status": workflow.get_status_summary()
            }

    async def _step_1_strategy_formulation(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Step 1: Strategy Architecture - High-level strategy formulation"""
        
        print("🎯 Workflow Step 1: Strategy Formulation")
        
        # Use StrategyArchitect for high-level strategy
        strategy_agent = self.roster["StrategyArchitectAgent"]
        enhanced_query = f"Based on the user's request '{workflow.user_query}', formulate a comprehensive investment strategy that can guide specific stock selection."
        
        strategy_result = strategy_agent.run(enhanced_query, context=context)
        
        if strategy_result.get("success", True):
            # Move to next step
            workflow.update_state(WorkflowState.AWAITING_SCREENING, strategy_result)
            
            # Automatically trigger next step
            return await self._step_2_security_screening(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return strategy_result

    async def _step_2_security_screening(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Step 2: Security Screening - Find specific stocks that match strategy"""
        
        print("🔍 Workflow Step 2: Security Screening")
        
        # Extract strategy insights for screening
        strategy_summary = workflow.strategy_result.get("summary", "")
        strategy_focus = self._extract_strategy_focus(strategy_summary)
        
        # Enhance query for SecurityScreener
        screening_query = f"Find specific stocks that align with this strategy: {strategy_focus}. "
        
        # Add portfolio complement analysis if portfolio exists
        if context.get("holdings_with_values"):
            screening_query += "Focus on securities that complement the existing portfolio."
        else:
            screening_query += "Focus on high-quality opportunities across value, growth, and quality factors."
        
        screener_agent = self.roster["SecurityScreenerAgent"]
        screening_result = await screener_agent.run(screening_query, context=context)
        
        if screening_result.get("success", True):
            # Move to next step
            workflow.update_state(WorkflowState.AWAITING_DEEP_ANALYSIS, screening_result)
            
            # Automatically trigger next step
            return await self._step_3_quantitative_analysis(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return screening_result

    async def _step_3_quantitative_analysis(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Step 3: Deep Quantitative Analysis - Risk analysis of proposed changes"""
        
        print("📊 Workflow Step 3: Quantitative Analysis")
        
        # Extract recommended tickers from screening
        recommendations = workflow.screening_result.get("recommendations", [])
        if recommendations:
            tickers = [rec.get("ticker") for rec in recommendations[:3]] # Top 3
            
            # Enhanced query for risk analysis
            analysis_query = f"Perform comprehensive risk analysis for adding {', '.join(tickers)} to the portfolio. "
            analysis_query += "Focus on tail risk, correlation analysis, and portfolio impact assessment."
        else:
            analysis_query = "Perform comprehensive portfolio risk analysis based on current holdings."
        
        quant_agent = self.roster["QuantitativeAnalystAgent"]
        analysis_result = await quant_agent.run(analysis_query, context=context)
        
        if analysis_result.get("success", True):
            # Move to final step
            workflow.update_state(WorkflowState.AWAITING_FINAL_SYNTHESIS, analysis_result)
            
            # Automatically trigger final step
            return self._step_4_final_synthesis(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return analysis_result

    async def _execute_enhanced_workflow_step(self, workflow: WorkflowSession, context: Dict, db_session) -> Dict[str, Any]:
        """Enhanced workflow execution with dynamic agent selection"""
        
        # Determine if we should use enhanced committee
        if hasattr(workflow, 'workflow_type') and workflow.workflow_type in ["enhanced", "crisis"]:
            return await self._execute_enhanced_multi_agent_workflow(workflow, context, db_session)
        else:
            # Use existing workflow logic for standard cases
            return await self._execute_workflow_step(workflow, context, db_session)

    async def _execute_enhanced_multi_agent_workflow(self, workflow: WorkflowSession, context: Dict, db_session) -> Dict[str, Any]:
        """Execute enhanced workflow with expanded agent committee"""
        
        try:
            if workflow.state == WorkflowState.AWAITING_STRATEGY:
                return await self._enhanced_step_1_strategy_formulation(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_SCREENING:
                return await self._enhanced_step_2_security_screening(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_DEEP_ANALYSIS:
                return await self._enhanced_step_3_comprehensive_analysis(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_FINAL_SYNTHESIS:
                return self._enhanced_step_4_final_synthesis(workflow, context)
            
            else:
                raise ValueError(f"Unknown enhanced workflow state: {workflow.state}")
                
        except Exception as e:
            print(f"❌ Enhanced workflow step failed: {str(e)}")
            workflow.state = WorkflowState.ERROR
            workflow.errors.append(str(e))
            return {
                "success": False,
                "error": f"Enhanced workflow step failed: {str(e)}",
                "workflow_status": workflow.get_status_summary()
            }
        
    async def _enhanced_step_1_strategy_formulation(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Enhanced Step 1: Strategy formulation with complexity awareness"""
        
        print("🎯 Enhanced Workflow Step 1: Advanced Strategy Formulation")
        
        # Use StrategyArchitect with enhanced context
        strategy_agent = self.roster["StrategyArchitectAgent"]
        enhanced_query = f"Based on the user's {workflow.workflow_type} complexity request '{workflow.user_query}', formulate a comprehensive investment strategy."
        
        if workflow.workflow_type == "crisis":
            enhanced_query += " Focus on crisis-resistant strategies and defensive positioning."
        elif workflow.workflow_type == "enhanced":
            enhanced_query += " Consider cross-asset correlations and behavioral factors."
        
        strategy_result = strategy_agent.run(enhanced_query, context=context)
        
        if strategy_result.get("success", True):
            workflow.update_state(WorkflowState.AWAITING_SCREENING, strategy_result)
            return await self._enhanced_step_2_security_screening(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return strategy_result

    async def _enhanced_step_2_security_screening(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Enhanced Step 2: Security screening with advanced criteria"""
        
        print("🔍 Enhanced Workflow Step 2: Advanced Security Screening")
        
        # Extract strategy insights
        strategy_summary = workflow.strategy_result.get("summary", "")
        strategy_focus = self._extract_strategy_focus(strategy_summary)
        
        # Enhanced screening query based on workflow type
        if workflow.workflow_type == "crisis":
            screening_query = f"Find defensive, crisis-resistant stocks that align with: {strategy_focus}. "
            screening_query += "Prioritize low-beta, dividend-paying, and defensive sector securities."
        elif workflow.workflow_type == "enhanced":
            screening_query = f"Find stocks with strong cross-asset diversification benefits: {strategy_focus}. "
            screening_query += "Consider correlation benefits and regime resilience."
        else:
            screening_query = f"Find quality stocks that align with: {strategy_focus}."
        
        # Add portfolio context
        if context.get("holdings_with_values"):
            screening_query += " Focus on securities that complement the existing portfolio."
        
        screener_agent = self.roster["SecurityScreenerAgent"]
        screening_result = await screener_agent.run(screening_query, context=context)
        
        if screening_result.get("success", True):
            workflow.update_state(WorkflowState.AWAITING_DEEP_ANALYSIS, screening_result)
            return await self._enhanced_step_3_comprehensive_analysis(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return screening_result

    async def _enhanced_step_3_comprehensive_analysis(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Enhanced Step 3: Multi-agent comprehensive analysis"""
        
        print("📊 Enhanced Workflow Step 3: Multi-Agent Comprehensive Analysis")
        
        # Get recommended tickers from screening
        recommendations = workflow.screening_result.get("recommendations", [])
        
        # Parallel analysis with multiple agents
        analysis_results = {}
        
        # 1. Quantitative Risk Analysis
        if recommendations:
            tickers = [rec.get("ticker") for rec in recommendations[:3]]
            risk_query = f"Analyze portfolio risk impact of adding {', '.join(tickers)}"
        else:
            risk_query = "Perform comprehensive portfolio risk analysis"
        
        quant_agent = self.roster["QuantitativeAnalystAgent"]
        analysis_results["risk_analysis"] = await quant_agent.run(risk_query, context=context)
        
        # 2. Cross-Asset Analysis (NEW)
        cross_asset_query = f"Analyze cross-asset correlations and regime risks for portfolio"
        if "crisis" in workflow.user_query.lower():
            cross_asset_query += " with focus on crisis scenario correlations"
        
        cross_asset_agent = self.roster["CrossAssetAnalyst"]
        analysis_results["cross_asset"] = cross_asset_agent.run(cross_asset_query, context=context)
        
        # 3. Behavioral Analysis
        behavior_query = f"Identify behavioral factors and biases relevant to current market environment"
        behavioral_agent = self.roster["BehavioralFinanceAgent"]
        analysis_results["behavioral"] = behavioral_agent.run(behavior_query, context=context)
        
        # 4. Regime Forecasting
        regime_query = f"Assess market regime and potential transitions"
        regime_agent = self.roster["RegimeForecastingAgent"]
        analysis_results["regime"] = regime_agent.run(regime_query, context=context)
        
        # Synthesize multi-agent analysis
        synthesized_analysis = self._synthesize_multi_agent_analysis(analysis_results)
        
        if synthesized_analysis.get("success", True):
            workflow.update_state(WorkflowState.AWAITING_FINAL_SYNTHESIS, synthesized_analysis)
            return self._enhanced_step_4_final_synthesis(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return synthesized_analysis
        
    def _enhanced_step_4_final_synthesis(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Enhanced Step 4: Advanced final synthesis"""
        
        print("🎯 Enhanced Workflow Step 4: Advanced Final Synthesis")
        
        # Create enhanced synthesis based on workflow type
        if workflow.workflow_type in ["enhanced", "crisis"]:
            final_synthesis = self._synthesize_enhanced_workflow_results(workflow)
        else:
            final_synthesis = self._synthesize_workflow_results(workflow)
        
        workflow.update_state(WorkflowState.COMPLETE, final_synthesis)
        
        return {
            "success": True,
            "summary": final_synthesis["summary"],
            "workflow_type": f"{workflow.workflow_type}_multi_agent_analysis",
            "agent_used": f"Orchestrator ({workflow.workflow_type.title()} Multi-Agent Workflow)",
            "workflow_status": workflow.get_status_summary(),
            "detailed_results": {
                "strategy": workflow.strategy_result,
                "screening": workflow.screening_result, 
                "analysis": workflow.analysis_result,
                "synthesis": final_synthesis
            },
            "recommendations": final_synthesis["recommendations"],
            "implementation_plan": final_synthesis["implementation_plan"]
        }

    def _synthesize_enhanced_workflow_results(self, workflow: WorkflowSession) -> Dict[str, Any]:
        """Synthesize enhanced workflow results with multi-agent insights"""
        
        synthesis_summary = f"### 🎯 {workflow.workflow_type.title()} Multi-Agent Analysis & Recommendations\n\n"
        
        # Strategy Foundation
        strategy_summary = workflow.strategy_result.get("summary", "Strategy analysis completed")
        synthesis_summary += f"**📋 {workflow.workflow_type.title()} Investment Strategy:**\n{strategy_summary}\n\n"
        
        # Screening Results
        screening_recs = workflow.screening_result.get("recommendations", [])
        if screening_recs:
            synthesis_summary += f"**🔍 Advanced Security Recommendations:**\n"
            for i, rec in enumerate(screening_recs[:3], 1):
                ticker = rec.get("ticker", "N/A")
                score = rec.get("overall_score", 0)
                rationale = rec.get("rationale", "High-quality opportunity")
                synthesis_summary += f"{i}. **{ticker}** (Score: {score:.2f}) - {rationale}\n"
            synthesis_summary += "\n"
        
        # Multi-Agent Analysis Results
        if hasattr(workflow, 'analysis_result') and workflow.analysis_result:
            analysis_summary = workflow.analysis_result.get("summary", "Multi-agent analysis completed")
            synthesis_summary += f"**🤖 Multi-Agent Intelligence Assessment:**\n{analysis_summary}\n\n"
            
            # Extract component analyses if available
            component_analyses = workflow.analysis_result.get("component_analyses", {})
            if component_analyses:
                if "cross_asset" in component_analyses:
                    ca_result = component_analyses["cross_asset"]
                    synthesis_summary += f"**🌐 Cross-Asset Risk:** {ca_result.get('regime_status', 'Normal')}\n"
                    synthesis_summary += f"**📊 Diversification Score:** {ca_result.get('diversification_score', 'N/A')}/10\n\n"
        
        # Implementation Guidance
        if workflow.workflow_type == "crisis":
            synthesis_summary += f"**🚨 Crisis Implementation Plan:**\n"
            synthesis_summary += f"1. Prioritize defensive positioning immediately\n"
            synthesis_summary += f"2. Implement gradually with strict risk controls\n"
            synthesis_summary += f"3. Monitor correlations and regime indicators daily\n"
            synthesis_summary += f"4. Maintain higher cash reserves for opportunities\n\n"
        elif workflow.workflow_type == "enhanced":
            synthesis_summary += f"**🚀 Enhanced Implementation Plan:**\n"
            synthesis_summary += f"1. Review cross-asset correlation implications\n"
            synthesis_summary += f"2. Consider behavioral and regime factors\n"
            synthesis_summary += f"3. Implement with enhanced risk monitoring\n"
            synthesis_summary += f"4. Regular multi-factor reassessment\n\n"
        
        synthesis_summary += f"**✅ Conclusion:** This {workflow.workflow_type} multi-agent analysis provides comprehensive insights from {len(self.committee_templates[workflow.workflow_type].agents)} specialized experts."
        
        # Extract recommendations
        recommendations = []
        for rec in screening_recs[:3]:
            ticker = rec.get("ticker")
            if ticker:
                recommendations.append(f"Consider adding {ticker} to portfolio ({workflow.workflow_type} analysis)")
        
        # Enhanced implementation plan
        implementation_plan = [
            {
                "step": 1,
                "action": f"Review {workflow.workflow_type} recommendations",
                "details": f"Analyze insights from {len(self.committee_templates.get(workflow.workflow_type, {}).get('agents', []))} agent committee",
                "timeline": "1-2 days"
            },
            {
                "step": 2, 
                "action": "Validate cross-asset implications",
                "details": "Assess correlation and regime factors identified",
                "timeline": "1 day"
            },
            {
                "step": 3,
                "action": f"Execute {workflow.workflow_type} strategy",
                "details": "Implement with appropriate risk controls",
                "timeline": "1-2 weeks"
            },
            {
                "step": 4,
                "action": "Monitor multi-factor metrics", 
                "details": "Track performance across all analysis dimensions",
                "timeline": "Ongoing"
            }
        ]
        
        return {
            "summary": synthesis_summary,
            "recommendations": recommendations,
            "implementation_plan": implementation_plan,
            "confidence_score": 0.95, # Higher confidence due to multi-agent analysis
            "analysis_depth": f"{workflow.workflow_type}_multi_agent_comprehensive",
            "workflow_id": workflow.session_id,
            "agents_involved": len(self.committee_templates.get(workflow.workflow_type, {}).get('agents', []))
        }

    def _synthesize_multi_agent_analysis(self, analysis_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        
        synthesis_summary = "### 🎯 Multi-Agent Comprehensive Analysis\n\n"
        
        # Risk Analysis Summary
        risk_summary = analysis_results.get("risk_analysis", {}).get("summary", "Risk analysis completed")
        synthesis_summary += f"**📊 Quantitative Risk Assessment:**\n{risk_summary}\n\n"
        
        # Cross-Asset Analysis Summary
        cross_asset_summary = analysis_results.get("cross_asset", {}).get("summary", "Cross-asset analysis completed")
        synthesis_summary += f"**🌐 Cross-Asset Correlation Analysis:**\n{cross_asset_summary}\n\n"
        
        # Behavioral Analysis Summary
        behavioral_summary = analysis_results.get("behavioral", {}).get("summary", "Behavioral analysis completed")
        synthesis_summary += f"**🧠 Behavioral Finance Assessment:**\n{behavioral_summary}\n\n"
        
        # Regime Analysis Summary
        regime_summary = analysis_results.get("regime", {}).get("summary", "Regime analysis completed")
        synthesis_summary += f"**📈 Market Regime Forecast:**\n{regime_summary}\n\n"
        
        # Generate integrated insights
        integrated_insights = self._generate_integrated_insights(analysis_results)
        synthesis_summary += f"**🔗 Integrated Multi-Agent Insights:**\n{integrated_insights}"
        
        return {
            "success": True,
            "summary": synthesis_summary,
            "analysis_type": "multi_agent_comprehensive",
            "component_analyses": analysis_results,
            "confidence_score": 0.95 # Higher confidence due to multi-agent validation
        }

    def _generate_integrated_insights(self, analysis_results: Dict[str, Dict]) -> str:
        """Generate insights from multi-agent analysis"""
        
        insights = []
        
        # Check for regime warnings
        regime_result = analysis_results.get("regime", {})
        if "transition" in regime_result.get("summary", "").lower():
            insights.append("• Market regime transition detected - adjust portfolio positioning")
        
        # Check cross-asset risks
        cross_asset_result = analysis_results.get("cross_asset", {})
        if cross_asset_result.get("diversification_score", 10) < 6:
            insights.append("• Low diversification score - increase cross-asset allocation")
        
        # Check behavioral factors
        behavioral_result = analysis_results.get("behavioral", {})
        if "bias" in behavioral_result.get("summary", "").lower():
            insights.append("• Behavioral biases detected - implement systematic decision-making")
        
        # Default insight
        if not insights:
            insights.append("• Multi-agent analysis shows portfolio is well-positioned")
            insights.append("• Continue monitoring across risk, correlation, behavioral, and regime factors")
        
        return "\n".join(insights)

    def _step_4_final_synthesis(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Step 4: Final Synthesis - Combine all analysis into actionable plan"""
        
        print("🎯 Workflow Step 4: Final Synthesis")
        
        # Synthesize all results
        final_synthesis = self._synthesize_workflow_results(workflow)
        
        # Mark workflow as complete
        workflow.update_state(WorkflowState.COMPLETE, final_synthesis)
        
        return {
            "success": True,
            "summary": final_synthesis["summary"],
            "workflow_type": "multi_stage_analysis",
            "agent_used": "Orchestrator (Multi-Agent Workflow)",
            "workflow_status": workflow.get_status_summary(),
            "detailed_results": {
                "strategy": workflow.strategy_result,
                "screening": workflow.screening_result, 
                "analysis": workflow.analysis_result,
                "synthesis": final_synthesis
            },
            "recommendations": final_synthesis["recommendations"],
            "implementation_plan": final_synthesis["implementation_plan"]
        }

    def _extract_strategy_focus(self, strategy_summary: str) -> str:
        """Extract key focus areas from strategy summary"""
        
        # Simple keyword extraction - can be enhanced with NLP
        focus_areas = []
        
        if "value" in strategy_summary.lower():
            focus_areas.append("value investing")
        if "growth" in strategy_summary.lower():
            focus_areas.append("growth opportunities")
        if "quality" in strategy_summary.lower():
            focus_areas.append("high-quality companies")
        if "dividend" in strategy_summary.lower():
            focus_areas.append("dividend-paying stocks")
        if "defensive" in strategy_summary.lower():
            focus_areas.append("defensive positioning")
        if "momentum" in strategy_summary.lower():
            focus_areas.append("momentum strategies")
        
        if not focus_areas:
            focus_areas.append("balanced quality investing")
        
        return ", ".join(focus_areas)

    def _synthesize_workflow_results(self, workflow: WorkflowSession) -> Dict[str, Any]:
        """Synthesize all workflow steps into final actionable recommendations"""
        
        synthesis_summary = "### 🎯 Comprehensive Investment Analysis & Recommendations\n\n"
        
        # Strategy Foundation
        strategy_summary = workflow.strategy_result.get("summary", "Strategy analysis completed")
        synthesis_summary += f"**📋 Investment Strategy Foundation:**\n{strategy_summary}\n\n"
        
        # Specific Stock Recommendations
        screening_recs = workflow.screening_result.get("recommendations", [])
        if screening_recs:
            synthesis_summary += f"**🔍 Specific Stock Recommendations:**\n"
            for i, rec in enumerate(screening_recs[:3], 1):
                ticker = rec.get("ticker", "N/A")
                score = rec.get("overall_score", 0)
                rationale = rec.get("rationale", "High-quality opportunity")
                synthesis_summary += f"{i}. **{ticker}** (Score: {score:.2f}) - {rationale}\n"
            synthesis_summary += "\n"
        
        # Risk Assessment
        if workflow.analysis_result:
            risk_summary = workflow.analysis_result.get("summary", "Risk analysis completed")
            synthesis_summary += f"**📊 Risk Assessment:**\n{risk_summary}\n\n"
        
        # Implementation Guidance
        synthesis_summary += f"**🚀 Implementation Plan:**\n"
        synthesis_summary += f"1. Review the recommended securities in detail\n"
        synthesis_summary += f"2. Consider position sizing based on risk tolerance\n"
        synthesis_summary += f"3. Implement gradually to manage market timing risk\n"
        synthesis_summary += f"4. Monitor portfolio risk metrics post-implementation\n\n"
        
        synthesis_summary += f"**✅ Conclusion:** This multi-agent analysis provides a comprehensive path from strategy to specific actionable recommendations."
        
        # Extract actionable recommendations
        recommendations = []
        for rec in screening_recs[:3]:
            ticker = rec.get("ticker")
            if ticker:
                recommendations.append(f"Consider adding {ticker} to portfolio")
        
        # Implementation plan
        implementation_plan = [
            {
                "step": 1,
                "action": "Review recommended securities",
                "details": f"Analyze {len(screening_recs)} recommended stocks",
                "timeline": "1-2 days"
            },
            {
                "step": 2, 
                "action": "Determine position sizes",
                "details": "Calculate appropriate allocation based on risk analysis",
                "timeline": "1 day"
            },
            {
                "step": 3,
                "action": "Execute trades",
                "details": "Implement recommendations gradually",
                "timeline": "1-2 weeks"
            },
            {
                "step": 4,
                "action": "Monitor and adjust", 
                "details": "Track performance and risk metrics",
                "timeline": "Ongoing"
            }
        ]
        
        return {
            "summary": synthesis_summary,
            "recommendations": recommendations,
            "implementation_plan": implementation_plan,
            "confidence_score": 0.92,
            "analysis_depth": "comprehensive_multi_agent",
            "workflow_id": workflow.session_id
        }

    def get_workflow_status(self, session_id: str) -> Optional[Dict]:
        """Get current status of workflow session"""
        workflow = self.active_workflows.get(session_id)
        if workflow:
            return workflow.get_status_summary()
        return None

    async def route_query(self, user_query: str, db_session, current_user) -> Dict[str, Any]:
        """
        Enhanced routing with dynamic workflow capabilities and intelligent agent selection
        """
        
        # 🚀 ENHANCED: Check if query should trigger workflow with complexity analysis
        # FIX: Check for current_user before starting a workflow that might need it
        should_trigger, workflow_type, complexity_score = self._should_trigger_workflow(user_query)
        
        if should_trigger:
            if not current_user or not db_session:
                 return {"success": False, "error": "A comprehensive workflow requires a logged-in user."}
            print(f"🔥 Triggering {workflow_type} workflow for comprehensive analysis (complexity: {complexity_score:.2f})")
            return await self.start_workflow(user_query, db_session, current_user, workflow_type)
        
        # EXISTING: Multi-step plan logic (enhanced with dynamic agent selection)
        clean_query = re.sub(r'[^\w\s]', '', user_query.lower())
        query_words = set(clean_query.split())
        
        connector_words = {"and", "then", "after", "first", "next"}
        is_complex = len(query_words.intersection(connector_words)) > 0

        plan = None
        if is_complex:
            plan = self._create_execution_plan(user_query)

        if plan:
            # Execute multi-step plan with enhanced agent selection
            step_results = []
            current_context = {}
            if current_user and db_session:
                user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
                if user_portfolios:
                    current_context = get_market_data_for_portfolio(user_portfolios[0].holdings)
            
            for i, step in enumerate(plan):
                agent_name_from_plan = step.get("agent")
                step_query = step.get("query")
                
                # 🚀 ENHANCED: Try intelligent agent selection if plan agent not found
                agent_to_run = None
                selected_agent_name = None
                if agent_name_from_plan:
                    normalized_plan_agent_name = agent_name_from_plan.lower().replace("agent", "")
                    
                    for roster_key, agent_instance in self.roster.items():
                        normalized_roster_key = roster_key.lower().replace("agent", "")
                        if normalized_plan_agent_name == normalized_roster_key:
                            agent_to_run = agent_instance
                            selected_agent_name = roster_key
                            break
                    
                    if not agent_to_run:
                        fallback_agent_name = self._find_intelligent_fallback(agent_name_from_plan, step_query)
                        if fallback_agent_name and fallback_agent_name in self.roster:
                            agent_to_run = self.roster[fallback_agent_name]
                            selected_agent_name = fallback_agent_name
                            print(f"⚡ Using intelligent fallback: {fallback_agent_name} for planned agent: {agent_name_from_plan}")
                
                if not agent_to_run or not step_query:
                    step_results.append({"error": f"Invalid plan at step {i+1}. Could not find agent '{agent_name_from_plan}' or query is missing."})
                    continue

                print(f"Executing Plan Step {i+1}: Agent -> {agent_to_run.name}, Query -> {step_query}")
                
                result = None
                if selected_agent_name in self.mcp_agents:
                    result = await agent_to_run.run(user_query, context=portfolio_context)
                else:
                    result = agent_to_run.run(user_query, context=portfolio_context)
                
                step_results.append({agent_to_run.name: result})
                
                if isinstance(result, dict) and result.get("success"):
                    current_context.update(result)

            final_summary = "I have completed the multi-step analysis as requested.\n\n"
            for i, res in enumerate(step_results):
                agent_name = list(res.keys())[0] if res else "UnknownStep"
                step_result = res.get(agent_name, {})
                
                summary = "Step completed."
                if isinstance(step_result, dict):
                    if "error" in step_result:
                        summary = f"Step failed with error: {step_result['error']}"
                    else:
                        summary = step_result.get('summary', 'Step completed successfully.')
                elif isinstance(step_result, str):
                    summary = step_result
                
                final_summary += f"**Step {i+1} ({agent_name}):**\n{summary}\n\n"
            return {
                "success": True,
                "summary": final_summary,
                "agent_used": "Orchestrator (Sequential Workflow)",
                "steps": step_results
            }
        else:
            # 🚀 ENHANCED: Single-agent routing with intelligent selection
            selected_agent_name = self._classify_query_enhanced(query_words, user_query)
            if not selected_agent_name:
                print("Orchestrator could not determine a route for a simple query.")
                return {"success": False, "error": "I'm not sure which specialist can handle that request."}
            
            agent_to_run = self.roster[selected_agent_name]
            
            # --- START OF MODIFIED LOGIC ---
            portfolio_context = {}
            user_portfolios = None

            # Attempt to get portfolio context only if a user and db session are available
            if current_user and db_session:
                user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
                if user_portfolios:
                    primary_portfolio = user_portfolios[0]
                    print(f"Fetching market data for portfolio ID: {primary_portfolio.id}")
                    portfolio_context = get_market_data_for_portfolio(primary_portfolio.holdings)

            # Check if a portfolio is strictly required by the agent
            portfolio_required_agents = self._get_portfolio_required_agents()
            if selected_agent_name in portfolio_required_agents and not user_portfolios:
                error_msg = f"The '{agent_to_run.name}' requires a portfolio, but you don't have one yet."
                if not current_user:
                    error_msg = f"The '{agent_to_run.name}' requires a logged-in user with a portfolio."
                return {"success": False, "error": error_msg}

            # At this point, the agent can either run with or without a portfolio,
            # and we've fetched the context if it exists.
            print(f"Orchestrator routing query to: {agent_to_run.name}")
            
            result = None
            if selected_agent_name in self.mcp_agents:
                result = await agent_to_run.run(user_query, context=portfolio_context)
            else:
                result = agent_to_run.run(user_query, context=portfolio_context)
            # --- END OF MODIFIED LOGIC ---
            
            # 🚀 ENHANCED: Dynamic follow-up suggestions based on agent and complexity
            result = self._enhance_result_with_follow_ups(result, selected_agent_name, user_query, complexity_score)
            
            return result

    def _classify_query_enhanced(self, query_words: Set[str], full_query: str) -> Optional[str]:
        """Enhanced query classification with ALL agents including new MCP agents"""
        
        # NEW: Check for cross-asset patterns first (highest priority for new capability)
        cross_asset_keywords = {
            "cross-asset", "correlation", "correlations", "regime", "asset", "class", "diversification",
            "cross", "breakdown", "shift", "allocation"
        }
        
        if query_words.intersection(cross_asset_keywords):
            print("🌐 Routing to CrossAssetAnalyst for cross-asset analysis")
            return "CrossAssetAnalyst"
        
        # ENHANCED: Complete routing with new agents
        expanded_routing = {
            "SecurityScreenerAgent": {
                "find", "stocks", "recommend", "recommendations", "buy", "purchase",
                "screen", "screening", "complement", "complementary", "specific",
                "which", "what", "securities", "equities", "companies",
                "quality", "value", "growth", "diversify", "balance", "improve",
                "candidates", "options", "choices", "select", "selection"
            },
            
            # NEW: BehavioralFinanceAgent
            "BehavioralFinanceAgent": {
                "bias", "biases", "behavior", "behavioral", "psychology", "psychological",
                "cognitive", "decision", "emotional", "investor", "patterns", "analyze",
                "identify", "investment"
            },
            
            # NEW: EconomicDataAgent  
            "EconomicDataAgent": {
                "economic", "economy", "economics", "fed", "federal", "reserve", "policy",
                "interest", "rates", "inflation", "gdp", "unemployment", "yield", "curve",
                "indicators", "macro", "geopolitical", "monetary", "outlook", "recession",
                "analysis"
            },
            
            # NEW: TaxStrategistAgent
            "TaxStrategistAgent": {
                "tax", "taxes", "optimization", "planning", "loss", "harvesting",
                "location", "efficient", "after", "strategy", "wash", "sale", "drag",
                "retirement", "contributions", "benefits", "optimize", "year", "end",
                "efficiency", "charitable", "giving"
            },
            
            "QuantitativeAnalystAgent": {
                "analyze", "analysis", "analyzing", "analytical",
                "risk", "risks", "risky", 
                "report", "reports", "reporting",
                "factor", "factors",
                "alpha", "beta",
                "cvar", "var",
                "stress", "test", "testing",
                "tail", "performance", "metrics"
            },
            "StrategyRebalancingAgent": {
                "rebalance", "rebalancing", "rebalanced",
                "optimize", "optimization", "optimizing",
                "allocation", "allocate", "allocating",
                "diversify", "diversification", "diversifying"
            },
            "FinancialTutorAgent": {
                "explain", "explaining", "explained",
                "what", "is", "define", "definition",
                "learn", "learning", "teach", "tutorial",
                "help", "how"
            },
            "StrategyArchitectAgent": {
                "design", "designing", "create", "build",
                "strategy", "strategies", "strategic",
                "recommend", "recommendation", "suggest",
                "momentum", "trend", "find"
            },
            "HedgingStrategistAgent": {
                "hedge", "hedging", "protect", "protection",
                "volatility", "target"
            },
            "RegimeForecastingAgent": {
                "forecast", "forecasting", "predict",
                "regime", "transition", "change"
            },
            "ScenarioSimulationAgent": {
                "scenario", "scenarios", "simulate",
                "simulation", "what-if", "crash"
            },
            "StrategyBacktesterAgent": {
                "backtest", "backtesting", "test", "testing"
            }
        }
        
        # Calculate scores for each agent
        agent_scores = {}
        for agent_name, keywords in expanded_routing.items():
            score = len(query_words.intersection(keywords))
            if score > 0:
                agent_scores[agent_name] = score
        
        # ENHANCED: Add context-based scoring
        if agent_scores:
            # Boost SecurityScreener for actionable queries
            if any(word in full_query.lower() for word in ["actionable", "buy", "purchase", "recommend stocks"]):
                if "SecurityScreenerAgent" in agent_scores:
                    agent_scores["SecurityScreenerAgent"] += 2
            
            # Boost specific agents for domain queries
            if any(word in full_query.lower() for word in ["tax planning", "tax optimization", "tax strategy"]):
                if "TaxStrategistAgent" in agent_scores:
                    agent_scores["TaxStrategistAgent"] += 2
            
            if any(word in full_query.lower() for word in ["economic outlook", "fed policy", "macro analysis"]):
                if "EconomicDataAgent" in agent_scores:
                    agent_scores["EconomicDataAgent"] += 2
            
            if any(word in full_query.lower() for word in ["behavioral analysis", "investment psychology", "bias identification"]):
                if "BehavioralFinanceAgent" in agent_scores:
                    agent_scores["BehavioralFinanceAgent"] += 2
            
            # Boost QuantitativeAnalyst for risk queries
            if any(word in full_query.lower() for word in ["risk analysis", "portfolio risk", "stress test"]):
                if "QuantitativeAnalystAgent" in agent_scores:
                    agent_scores["QuantitativeAnalystAgent"] += 1
            
            best_agent = max(agent_scores.items(), key=lambda x: x[1])
            print(f"🎯 Enhanced routing to {best_agent[0]} (score: {best_agent[1]})")
            return best_agent[0]
        
        # EXISTING: Fallback logic
        portfolio_words = {"portfolio", "portfolios", "holdings", "stocks", "investments"}
        if query_words.intersection(portfolio_words):
            print("Defaulting to QuantitativeAnalystAgent for portfolio query")
            return "QuantitativeAnalystAgent"
        
        print("No specific routing found, defaulting to FinancialTutorAgent")
        return "FinancialTutorAgent"

    def _find_intelligent_fallback(self, planned_agent: str, step_query: str) -> Optional[str]:
        """Find intelligent fallback agent when planned agent not available"""
        
        # Map common plan agent names to roster agent names
        agent_mapping = {
            "quantitative": "QuantitativeAnalystAgent",
            "strategy": "StrategyArchitectAgent", 
            "security": "SecurityScreenerAgent",
            "screener": "SecurityScreenerAgent",
            "risk": "QuantitativeAnalystAgent",
            "rebalancing": "StrategyRebalancingAgent",
            "hedging": "HedgingStrategistAgent",
            "backtest": "StrategyBacktesterAgent",
            "tutor": "FinancialTutorAgent",
            "regime": "RegimeForecastingAgent",
            "behavioral": "BehavioralFinanceAgent",
            "scenario": "ScenarioSimulationAgent",
            "cross_asset": "CrossAssetAnalyst",
            "crossasset": "CrossAssetAnalyst"
        }
        
        planned_lower = planned_agent.lower().replace("agent", "")
        
        for key, roster_agent in agent_mapping.items():
            if key in planned_lower and roster_agent in self.roster:
                return roster_agent
        
        # If no direct mapping, analyze step query for best fit
        step_words = set(step_query.lower().split())
        return self._classify_query_enhanced(step_words, step_query)

    def _get_portfolio_required_agents(self) -> List[str]:
        """Get agents that require portfolio context"""
        return [
            "QuantitativeAnalystAgent", "StrategyRebalancingAgent", 
            "HedgingStrategistAgent", "RegimeForecastingAgent",
            "BehavioralFinanceAgent", "ScenarioSimulationAgent",
            "StrategyBacktesterAgent", 
        ]

    def _get_portfolio_optional_agents(self) -> List[str]:
        """Get agents that don't require portfolio context"""
        return ["StrategyArchitectAgent", "FinancialTutorAgent"]

    def _enhance_result_with_follow_ups(self, result: Dict, agent_name: str, query: str, complexity_score: float) -> Dict:
        """Enhanced result with intelligent follow-up suggestions"""
        
        if not result or not result.get("success"):
            return result
        
        # EXISTING: SecurityScreener follow-up logic
        if (agent_name == "SecurityScreenerAgent" and 
            result.get("success") and 
            "complement" not in query.lower()):
            
            if "summary" in result:
                result["summary"] += "\n\n💡 **Would you like me to assemble the quantitative team to perform deeper risk analysis on these recommendations?**"
                result["follow_up_available"] = True
                result["follow_up_type"] = "risk_analysis"
        
        # 🚀 NEW: CrossAssetAnalyst follow-up suggestions
        elif agent_name == "CrossAssetAnalyst":
            if complexity_score > 0.5:
                result["summary"] += "\n\n🔗 **Would you like me to trigger a comprehensive multi-agent workflow to analyze these cross-asset insights further?**"
                result["follow_up_available"] = True
                result["follow_up_type"] = "enhanced_workflow"
        
        # 🚀 NEW: QuantitativeAnalyst follow-up for high complexity
        elif agent_name == "QuantitativeAnalystAgent" and complexity_score > 0.7:
            result["summary"] += "\n\n📊 **This analysis could benefit from cross-asset correlation analysis. Would you like me to include regime and behavioral factors?**"
            result["follow_up_available"] = True
            result["follow_up_type"] = "enhanced_analysis"
        
        # 🚀 NEW: General high-complexity follow-up
        elif complexity_score > 0.8 and not result.get("follow_up_available"):
            result["summary"] += "\n\n🚀 **This query shows high complexity. Would you like me to assemble the full committee for comprehensive analysis?**"
            result["follow_up_available"] = True
            result["follow_up_type"] = "crisis_workflow"
        
        return result

    async def execute_follow_up_analysis(self, original_result: Dict, db_session, current_user) -> Dict[str, Any]:
        """Execute follow-up risk analysis on SecurityScreener results"""
        if not original_result.get("follow_up_available"):
            return {"success": False, "error": "No follow-up analysis available"}
        
        if original_result.get("follow_up_type") != "risk_analysis":
            return {"success": False, "error": "Unknown follow-up type"}
        
        user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
        if not user_portfolios:
            return {"success": False, "error": "Portfolio required for risk analysis"}
        
        portfolio_context = get_market_data_for_portfolio(user_portfolios[0].holdings)
        
        recommendations = original_result.get("recommendations", [])
        if not recommendations:
            return {"success": False, "error": "No recommendations found for analysis"}
        
        tickers = [r["ticker"] for r in recommendations[:3]]
        analysis_query = f"Analyze the risk implications of adding {', '.join(tickers)} to the portfolio"
        
        quant_agent = self.roster["QuantitativeAnalystAgent"]
        risk_analysis = await quant_agent.run(analysis_query, context=portfolio_context)
        
        combined_summary = f"### Security Screening + Risk Analysis\n\n"
        combined_summary += f"**Original Recommendations:**\n{original_result['summary']}\n\n"
        combined_summary += f"**Risk Analysis:**\n{risk_analysis.get('summary', 'Risk analysis completed')}\n\n"
        
        return {
            "success": True,
            "summary": combined_summary,
            "agent_used": "Orchestrator (SecurityScreener + QuantitativeAnalyst)",
            "original_screening": original_result,
            "risk_analysis": risk_analysis
        }

    def _create_execution_plan(self, user_query: str) -> Optional[List[Dict[str, str]]]:
        """Enhanced LLM planning with SecurityScreener awareness (existing method)"""
        print("Orchestrator: Query is complex. Engaging LLM to create an execution plan...")
        
        agent_descriptions = "\n".join([f"- **{agent.name}**: {agent.purpose}" for agent in self.roster.values()])
        
        prompt = f"""
        You are an expert financial orchestrator. Your job is to decompose a user's complex financial query into a logical, sequential plan of discrete steps. Each step must be assigned to the single best specialist agent from the available team.

        Here is the team of specialist agents you can use. You MUST use these exact names:
        {agent_descriptions}

        Here is the user's query: "{user_query}"

        IMPORTANT: If the user is asking for specific stock recommendations, consider using SecurityScreenerAgent to bridge strategy to actionable advice.

        Analyze the query and create a JSON array of steps. Each step in the array should be a JSON object with two keys: "agent" and "query".

        CRITICAL RULES:
        - You MUST respond with ONLY a valid JSON object containing a single key "plan".
        - The "agent" value in each step MUST EXACTLY MATCH one of the names from the list provided (e.g., 'QuantitativeAnalystAgent', 'SecurityScreenerAgent'). Do NOT shorten the names.
        - The "query" value for each step should be a clear, self-contained instruction for that agent.

        Now, create the plan for the user's actual query.
        """

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620", 
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text
            # Attempt to extract JSON from a markdown block if the model wraps it
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()

            plan_json = json.loads(response_text)
            print(f"Orchestrator: LLM Planner generated plan: {plan_json}")
            return plan_json.get("plan")
        except (json.JSONDecodeError, KeyError, IndexError, Exception) as e:
            print(f"Orchestrator: LLM planning failed. Response was: '{response_text}'. Error: {e}")
            return None

    def _classify_query_simple(self, query_words: Set[str]) -> Optional[str]:
        """Enhanced keyword-based routing with SecurityScreener (existing method)"""
        
        expanded_routing = {
            "SecurityScreenerAgent": {
                "find", "stocks", "recommend", "recommendations", "buy", "purchase",
                "screen", "screening", "complement", "complementary", "specific",
                "which", "what", "securities", "equities", "companies",
                "quality", "value", "growth", "diversify", "balance", "improve",
                "candidates", "options", "choices", "select", "selection"
            },
            
            "QuantitativeAnalystAgent": {
                "analyze", "analysis", "analyzing", "analytical",
                "risk", "risks", "risky", 
                "report", "reports", "reporting",
                "factor", "factors",
                "alpha", "beta",
                "cvar", "var",
                "stress", "test", "testing",
                "tail", "performance", "metrics"
            },
            "StrategyRebalancingAgent": {
                "rebalance", "rebalancing", "rebalanced",
                "optimize", "optimization", "optimizing",
                "allocation", "allocate", "allocating",
                "diversify", "diversification", "diversifying"
            },
            "FinancialTutorAgent": {
                "explain", "explaining", "explained",
                "what", "is", "define", "definition",
                "learn", "learning", "teach", "tutorial",
                "help", "how"
            },
            "StrategyArchitectAgent": {
                "design", "designing", "create", "build",
                "strategy", "strategies", "strategic",
                "recommend", "recommendation", "suggest",
                "momentum", "trend", "find"
            },
            "HedgingStrategistAgent": {
                "hedge", "hedging", "protect", "protection",
                "volatility", "target"
            },
            "RegimeForecastingAgent": {
                "forecast", "forecasting", "predict",
                "regime", "transition", "change"
            },
            "BehavioralFinanceAgent": {
                "bias", "biases", "behavior", "psychology",
                "behavioral", "psychological"
            },
            "ScenarioSimulationAgent": {
                "scenario", "scenarios", "simulate",
                "simulation", "what-if", "crash"
            },
            "StrategyBacktesterAgent": {
                "backtest", "backtesting", "test", "testing"
            }
        }
        
        agent_scores = {}
        for agent_name, keywords in expanded_routing.items():
            score = len(query_words.intersection(keywords))
            if score > 0:
                agent_scores[agent_name] = score
        
        if agent_scores:
            best_agent = max(agent_scores.items(), key=lambda x: x[1])
            print(f"Routing to {best_agent[0]} (score: {best_agent[1]})")
            return best_agent[0]
        
        portfolio_words = {"portfolio", "portfolios", "holdings", "stocks", "investments"}
        if query_words.intersection(portfolio_words):
            print("Defaulting to QuantitativeAnalystAgent for portfolio query")
            return "QuantitativeAnalystAgent"
        
        print("No specific routing found, defaulting to FinancialTutorAgent")
        return "FinancialTutorAgent"