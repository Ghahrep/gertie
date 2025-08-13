# in agents/orchestrator.py
import re
import json
from typing import Dict, Any, Set, Optional, List

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
from agents.conversation_manager import AutoGenConversationManager

# Backend Imports
from db import crud
from core.data_handler import get_market_data_for_portfolio
import anthropic
from core.config import settings


class FinancialOrchestrator:
    """
    The central command hub for the agent team. Can create and execute 
    multi-step plans for complex queries or route simple queries directly.
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
        }
        print(f"FinancialOrchestrator initialized with a team of {len(self.roster)} agents.")
        self._setup_classification_patterns()
        llm_config = {
            "config_list": [{
                "model": "claude-3-5-sonnet-20240620",
                "api_key": settings.ANTHROPIC_API_KEY,
                "api_type": "anthropic",
            }],
            "temperature": 0.2,
        }
        self.conversation_manager = AutoGenConversationManager(llm_config=llm_config)

    def _setup_classification_patterns(self):
        """Setup prioritized keyword sets for routing."""
        self.routing_map = {
            
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
    
    def _create_execution_plan(self, user_query: str) -> Optional[List[Dict[str, str]]]:
        """Uses an LLM to decompose a complex query into a sequence of agent tasks."""
        print("Orchestrator: Query is complex. Engaging LLM to create an execution plan...")
        
        agent_descriptions = "\n".join([f"- **{agent.name}**: {agent.purpose}" for agent in self.roster.values()])
        
        prompt = f"""
        You are an expert financial orchestrator. Your job is to decompose a user's complex financial query into a logical, sequential plan of discrete steps. Each step must be assigned to the single best specialist agent from the available team.

        Here is the team of specialist agents you can use. You MUST use these exact names:
        {agent_descriptions}

        Here is the user's query: "{user_query}"

        Analyze the query and create a JSON array of steps. Each step in the array should be a JSON object with two keys: "agent" and "query".

        CRITICAL RULES:
        - You MUST respond with ONLY a valid JSON object containing a single key "plan".
        - The "agent" value in each step MUST EXACTLY MATCH one of the names from the list provided (e.g., 'QuantitativeAnalystAgent', 'StrategyRebalancingAgent'). Do NOT shorten the names.
        - The "query" value for each step should be a clear, self-contained instruction for that agent.

        Now, create the plan for the user's actual query.
        """

        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620", max_tokens=1024,
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
        """Enhanced keyword-based routing with stemming and partial matches."""
        
        # Create expanded keyword mappings with variations
        expanded_routing = {
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
            "StrategyRebalancingAgent": {
                "rebalance", "optimize", "allocation"
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
        
        # Score each agent based on keyword matches
        agent_scores = {}
        for agent_name, keywords in expanded_routing.items():
            score = len(query_words.intersection(keywords))
            if score > 0:
                agent_scores[agent_name] = score
        
        # Return the agent with the highest score
        if agent_scores:
            best_agent = max(agent_scores.items(), key=lambda x: x[1])
            print(f"Routing to {best_agent[0]} (score: {best_agent[1]})")
            return best_agent[0]
        
        # Default fallback for portfolio-related queries
        portfolio_words = {"portfolio", "portfolios", "holdings", "stocks", "investments"}
        if query_words.intersection(portfolio_words):
            print("Defaulting to QuantitativeAnalystAgent for portfolio query")
            return "QuantitativeAnalystAgent"
        
        # Final fallback
        print("No specific routing found, defaulting to FinancialTutorAgent")
        return "FinancialTutorAgent"

    def route_query(self, user_query: str, db_session, current_user) -> Dict[str, Any]:
        """
        ### FINAL VERSION: With robust, flexible plan execution logic.
        """
        clean_query = re.sub(r'[^\w\s]', '', user_query.lower())
        query_words = set(clean_query.split())
        
        connector_words = {"and", "then", "after", "first", "next"}
        is_complex = len(query_words.intersection(connector_words)) > 0

        plan = None
        if is_complex:
            plan = self._create_execution_plan(user_query)

        if plan:
            # --- Execute a Multi-Step Plan ---
            step_results = []
            user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
            current_context = get_market_data_for_portfolio(user_portfolios[0].holdings) if user_portfolios else {}
            
            for i, step in enumerate(plan):
                agent_name_from_plan = step.get("agent")
                step_query = step.get("query")
                
                # ### THE FIX: Implement a flexible agent lookup ###
                agent_to_run = None
                if agent_name_from_plan:
                    # Normalize the name from the plan for matching (e.g., "QuantitativeAnalyst" -> "quantitativeanalyst")
                    normalized_plan_agent_name = agent_name_from_plan.lower().replace("agent", "")
                    
                    for roster_key, agent_instance in self.roster.items():
                        # Normalize the key from our roster (e.g., "QuantitativeAnalystAgent" -> "quantitativeanalyst")
                        normalized_roster_key = roster_key.lower().replace("agent", "")
                        if normalized_plan_agent_name == normalized_roster_key:
                            agent_to_run = agent_instance
                            break # Found the match, stop searching
                
                if not agent_to_run or not step_query:
                    step_results.append({"error": f"Invalid plan at step {i+1}. Could not find agent '{agent_name_from_plan}' or query is missing."})
                    continue

                print(f"Executing Plan Step {i+1}: Agent -> {agent_to_run.name}, Query -> {step_query}")
                
                result = agent_to_run.run(step_query, context=current_context)
                step_results.append({agent_to_run.name: result})
                
                if isinstance(result, dict) and result.get("success"):
                    current_context.update(result)

            # Synthesize final response from all steps
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
            # --- ### THIS IS THE FULL SINGLE-AGENT QUERY LOGIC ### ---
            selected_agent_name = self._classify_query_simple(query_words)
            if not selected_agent_name:
                print("Orchestrator could not determine a route for a simple query.")
                return {"success": False, "error": "I'm not sure which specialist can handle that request."}
            
            agent_to_run = self.roster[selected_agent_name]
            
            agents_that_dont_need_portfolio = ["StrategyArchitectAgent", "FinancialTutorAgent"]
            
            if selected_agent_name in agents_that_dont_need_portfolio:
                print(f"Orchestrator routing query to: {agent_to_run.name} (no portfolio context needed)")
                return agent_to_run.run(user_query, context={})
            else:
                user_portfolios = crud.get_user_portfolios(db=db_session, user_id=current_user.id)
                if not user_portfolios:
                    return {"success": False, "error": f"The '{agent_to_run.name}' requires a portfolio, but you don't have one yet."}

                primary_portfolio = user_portfolios[0]
                print(f"Fetching market data for portfolio ID: {primary_portfolio.id}")
                portfolio_context = get_market_data_for_portfolio(primary_portfolio.holdings)
                
                print(f"Orchestrator routing query to: {agent_to_run.name}")
                return agent_to_run.run(user_query, context=portfolio_context)