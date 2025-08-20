# agents/orchestrator.py - Enhanced with Multi-Stage Workflow
import re
import json
import uuid
from typing import Dict, Any, Set, Optional, List
from datetime import datetime
from enum import Enum

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

# Backend Imports
from db import crud
from core.data_handler import get_market_data_for_portfolio
import anthropic
from core.config import settings

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
    Enhanced orchestrator with multi-stage workflow capabilities and SecurityScreener integration
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
        }
        print(f"FinancialOrchestrator initialized with a team of {len(self.roster)} agents.")
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
        
        # 🚀 NEW: Workflow session management
        self.active_workflows: Dict[str, WorkflowSession] = {}

    def _setup_classification_patterns(self):
        """Enhanced routing with SecurityScreener patterns"""
        self.routing_map = {
            # SecurityScreener patterns - highest priority for actionable queries
            "SecurityScreenerAgent": {
                frozenset(["find", "stocks"]), frozenset(["recommend", "stocks"]), 
                frozenset(["buy", "what"]), frozenset(["screen"]), frozenset(["complement"]),
                frozenset(["specific", "recommendations"]), frozenset(["which", "stocks"]),
                frozenset(["quality", "stocks"]), frozenset(["value", "stocks"]),
                frozenset(["growth", "stocks"]), frozenset(["diversify", "with"]),
                frozenset(["actionable", "advice"]), frozenset(["stocks", "to", "buy"])
            },
            
            "StrategyRebalancingAgent": {frozenset(["rebalance"]), frozenset(["optimize"]), frozenset(["allocation"]), frozenset(["risk", "parity"]), frozenset(["diversify", "risk"])},
            "StrategyBacktesterAgent": {frozenset(["backtest"]), frozenset(["test", "strategy"])},
            "HedgingStrategistAgent": {frozenset(["hedge"]), frozenset(["protect"]), frozenset(["target", "volatility"])},
            "StrategyArchitectAgent": {frozenset(["design"]), frozenset(["find", "strategy"]), frozenset(["recommend"]), frozenset(["momentum"])},
            "RegimeForecastingAgent": {frozenset(["forecast"]), frozenset(["regime", "change"]), frozenset(["transition"])},
            "BehavioralFinanceAgent": {frozenset(["bias"]), frozenset(["biases"]), frozenset(["behavior"]), frozenset(["psychology"])},
            "ScenarioSimulationAgent": {frozenset(["scenario"]), frozenset(["simulate"]), frozenset(["what", "if"]), frozenset(["market", "crash"])},
            "FinancialTutorAgent": {frozenset(["explain"]), frozenset(["what", "is"]), frozenset(["define"]), frozenset(["learn"])},
            "QuantitativeAnalystAgent": {frozenset(["risk"]), frozenset(["report"]), frozenset(["factor"]), frozenset(["alpha"]), frozenset(["cvar"]), frozenset(["analyze"]), frozenset(["stress", "test"]), frozenset(["tail", "risk"])}
        }

    def _should_trigger_workflow(self, user_query: str) -> bool:
        """Determine if query should trigger multi-stage workflow"""
        workflow_triggers = [
            # Actionable investment queries
            "what should i do", "what stocks", "recommend", "advice",
            "portfolio improvement", "investment strategy", "rebalance my portfolio",
            "optimize my", "improve my", "help with my portfolio",
            # Comprehensive analysis requests
            "full analysis", "complete review", "thorough examination",
            "end-to-end", "comprehensive", "detailed analysis",
            # Specific action requests
            "actionable recommendations", "specific steps", "concrete advice",
            "trading plan", "implementation", "execute"
        ]
        
        query_lower = user_query.lower()
        return any(trigger in query_lower for trigger in workflow_triggers)

    def start_workflow(self, user_query: str, db_session, current_user) -> Dict[str, Any]:
        """🚀 NEW: Start multi-stage workflow execution"""
        
        # Create new workflow session
        session_id = str(uuid.uuid4())
        workflow = WorkflowSession(session_id, user_query, current_user.id)
        self.active_workflows[session_id] = workflow
        
        print(f"🚀 Starting new workflow session: {session_id}")
        print(f"📝 Query: {user_query}")
        
        # Get portfolio context
        user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
        portfolio_context = {}
        if user_portfolios:
            primary_portfolio = user_portfolios[0]
            portfolio_context = get_market_data_for_portfolio(primary_portfolio.holdings)
        
        # Start with strategy formulation
        return self._execute_workflow_step(workflow, portfolio_context, db_session)

    def _execute_workflow_step(self, workflow: WorkflowSession, context: Dict, db_session) -> Dict[str, Any]:
        """Execute current workflow step based on state"""
        
        try:
            if workflow.state == WorkflowState.AWAITING_STRATEGY:
                return self._step_1_strategy_formulation(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_SCREENING:
                return self._step_2_security_screening(workflow, context)
            
            elif workflow.state == WorkflowState.AWAITING_DEEP_ANALYSIS:
                return self._step_3_quantitative_analysis(workflow, context)
            
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

    def _step_1_strategy_formulation(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
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
            return self._step_2_security_screening(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return strategy_result

    def _step_2_security_screening(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
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
        screening_result = screener_agent.run(screening_query, context=context)
        
        if screening_result.get("success", True):
            # Move to next step
            workflow.update_state(WorkflowState.AWAITING_DEEP_ANALYSIS, screening_result)
            
            # Automatically trigger next step
            return self._step_3_quantitative_analysis(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return screening_result

    def _step_3_quantitative_analysis(self, workflow: WorkflowSession, context: Dict) -> Dict[str, Any]:
        """Step 3: Deep Quantitative Analysis - Risk analysis of proposed changes"""
        
        print("📊 Workflow Step 3: Quantitative Analysis")
        
        # Extract recommended tickers from screening
        recommendations = workflow.screening_result.get("recommendations", [])
        if recommendations:
            tickers = [rec.get("ticker") for rec in recommendations[:3]]  # Top 3
            
            # Enhanced query for risk analysis
            analysis_query = f"Perform comprehensive risk analysis for adding {', '.join(tickers)} to the portfolio. "
            analysis_query += "Focus on tail risk, correlation analysis, and portfolio impact assessment."
        else:
            analysis_query = "Perform comprehensive portfolio risk analysis based on current holdings."
        
        quant_agent = self.roster["QuantitativeAnalystAgent"]
        analysis_result = quant_agent.run(analysis_query, context=context)
        
        if analysis_result.get("success", True):
            # Move to final step
            workflow.update_state(WorkflowState.AWAITING_FINAL_SYNTHESIS, analysis_result)
            
            # Automatically trigger final step
            return self._step_4_final_synthesis(workflow, context)
        else:
            workflow.state = WorkflowState.ERROR
            return analysis_result

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

    def route_query(self, user_query: str, db_session, current_user) -> Dict[str, Any]:
        """
        Enhanced routing with workflow capabilities
        """
        
        # 🚀 NEW: Check if query should trigger multi-stage workflow
        if self._should_trigger_workflow(user_query):
            print("🔥 Triggering multi-stage workflow for comprehensive analysis")
            return self.start_workflow(user_query, db_session, current_user)
        
        # Otherwise use existing single-agent routing
        clean_query = re.sub(r'[^\w\s]', '', user_query.lower())
        query_words = set(clean_query.split())
        
        connector_words = {"and", "then", "after", "first", "next"}
        is_complex = len(query_words.intersection(connector_words)) > 0

        plan = None
        if is_complex:
            plan = self._create_execution_plan(user_query)

        if plan:
            # Execute multi-step plan (existing logic)
            step_results = []
            user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
            current_context = get_market_data_for_portfolio(user_portfolios[0].holdings) if user_portfolios else {}
            
            for i, step in enumerate(plan):
                agent_name_from_plan = step.get("agent")
                step_query = step.get("query")
                
                agent_to_run = None
                if agent_name_from_plan:
                    normalized_plan_agent_name = agent_name_from_plan.lower().replace("agent", "")
                    
                    for roster_key, agent_instance in self.roster.items():
                        normalized_roster_key = roster_key.lower().replace("agent", "")
                        if normalized_plan_agent_name == normalized_roster_key:
                            agent_to_run = agent_instance
                            break
                
                if not agent_to_run or not step_query:
                    step_results.append({"error": f"Invalid plan at step {i+1}. Could not find agent '{agent_name_from_plan}' or query is missing."})
                    continue

                print(f"Executing Plan Step {i+1}: Agent -> {agent_to_run.name}, Query -> {step_query}")
                
                result = agent_to_run.run(step_query, context=current_context)
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
            # Single-agent routing (existing logic)
            selected_agent_name = self._classify_query_simple(query_words)
            if not selected_agent_name:
                print("Orchestrator could not determine a route for a simple query.")
                return {"success": False, "error": "I'm not sure which specialist can handle that request."}
            
            agent_to_run = self.roster[selected_agent_name]
            
            agents_that_dont_need_portfolio = ["StrategyArchitectAgent", "FinancialTutorAgent"]
            agents_that_need_portfolio = [
                "QuantitativeAnalystAgent", "StrategyRebalancingAgent", 
                "HedgingStrategistAgent", "RegimeForecastingAgent",
                "BehavioralFinanceAgent", "ScenarioSimulationAgent",
                "SecurityScreenerAgent"
            ]
            
            if selected_agent_name in agents_that_dont_need_portfolio:
                print(f"Orchestrator routing query to: {agent_to_run.name} (no portfolio context needed)")
                return agent_to_run.run(user_query, context={})
            else:
                user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
                if not user_portfolios:
                    if selected_agent_name == "SecurityScreenerAgent":
                        print(f"Orchestrator routing query to: {agent_to_run.name} (general screening mode)")
                        return agent_to_run.run(user_query, context={})
                    else:
                        return {"success": False, "error": f"The '{agent_to_run.name}' requires a portfolio, but you don't have one yet."}

                primary_portfolio = user_portfolios[0]
                print(f"Fetching market data for portfolio ID: {primary_portfolio.id}")
                portfolio_context = get_market_data_for_portfolio(primary_portfolio.holdings)
                
                print(f"Orchestrator routing query to: {agent_to_run.name}")
                result = agent_to_run.run(user_query, context=portfolio_context)
                
                # Enhanced result with "assemble team" capability
                if (selected_agent_name == "SecurityScreenerAgent" and 
                    result.get("success") and 
                    "complement" not in user_query.lower()):
                    
                    if "summary" in result:
                        result["summary"] += "\n\n💡 **Would you like me to assemble the quantitative team to perform deeper risk analysis on these recommendations?**"
                        result["follow_up_available"] = True
                        result["follow_up_type"] = "risk_analysis"
                
                return result

    def execute_follow_up_analysis(self, original_result: Dict, db_session, current_user) -> Dict[str, Any]:
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
        risk_analysis = quant_agent.run(analysis_query, context=portfolio_context)
        
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
            plan_json = json.loads(response_text)
            print(f"Orchestrator: LLM Planner generated plan: {plan_json}")
            return plan_json.get("plan")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Orchestrator: LLM planning failed. {e}")
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