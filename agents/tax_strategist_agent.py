# agents/tax_strategist_agent.py
"""
TaxStrategistAgent - Revolutionary AI Tax Optimization
====================================================
Intelligent tax strategist agent that provides sophisticated tax optimization
with MCP integration and debate capabilities. Specializes in:
- Tax-loss harvesting with wash sale compliance
- Asset location optimization across account types
- Year-end tax planning automation
- After-tax return analysis and optimization
"""

from agents.mcp_base_agent import MCPBaseAgent
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

class DebatePerspective(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    SPECIALIST = "specialist"

class TaxStrategy(Enum):
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    OPPORTUNISTIC = "opportunistic"

@dataclass
class TaxLossOpportunity:
    """Represents a tax-loss harvesting opportunity"""
    symbol: str
    current_price: float
    cost_basis: float
    unrealized_loss: float
    tax_benefit: float
    wash_sale_risk: str
    replacement_securities: List[str]
    optimal_timing: str
    priority_score: float

@dataclass
class AssetLocationRecommendation:
    """Asset location optimization recommendation"""
    asset_type: str
    current_account: str
    recommended_account: str
    tax_savings: float
    reasoning: str
    implementation_complexity: str

class TaxStrategistAgent(MCPBaseAgent):
    """Revolutionary AI Tax Strategist with MCP + Debate Capabilities"""
    
    def __init__(self, agent_id: str = "tax_strategist"):
        super().__init__(
            agent_id=agent_id,
            agent_name="Tax Strategist Agent",
            capabilities=[
                "tax_loss_harvesting",
                "asset_location_optimization", 
                "year_end_tax_planning",
                "after_tax_analysis",
                "tax_optimization",
                "wash_sale_compliance",
                "retirement_planning",
                "debate_participation"
            ]
        )
        
        # Tax optimization configuration
        # Using 2025 brackets for forward-looking analysis
        self.tax_brackets = self._initialize_tax_brackets()
        self.wash_sale_period = 30  # Days before/after for wash sale rule
        self.capital_gains_rates = self._initialize_capital_gains_rates()
        self.account_types = self._initialize_account_types()
        
        # Debate system properties
        self.perspective = DebatePerspective.SPECIALIST
        self.specialization = "tax_optimization_and_efficiency"
        self.debate_strengths = ["after_tax_analysis", "tax_efficiency", "regulatory_compliance", "long_term_planning"]
        
        # Debate personality configuration  
        self.debate_config = {
            "focus": "tax_efficiency",
            "evidence_preference": "regulatory_analysis",
            "bias": "after_tax_optimization",
            "argument_style": "compliance_focused",
            "challenge_approach": "efficiency_maximization",
            "confidence_threshold": 0.8
        }
    
    def _initialize_tax_brackets(self) -> Dict[str, List[Tuple[float, float]]]:
        """Initialize projected 2025 tax brackets for a single filer"""
        return {
            "ordinary_income": [
                (0, 0.10),      # 10%: $0 - $11,600
                (11600, 0.12),  # 12%: $11,601 - $47,150
                (47150, 0.22),  # 22%: $47,151 - $100,525
                (100525, 0.24), # 24%: $100,526 - $191,950
                (191950, 0.32), # 32%: $191,951 - $243,725
                (243725, 0.35), # 35%: $243,726 - $609,350
                (609350, 0.37)  # 37%: $609,351+
            ],
            "long_term_capital_gains": [
                (0, 0.0),       # 0%: $0 - $47,025
                (47025, 0.15),  # 15%: $47,026 - $518,900
                (518900, 0.20)  # 20%: $518,901+
            ]
        }
    
    def _initialize_capital_gains_rates(self) -> Dict[str, float]:
        """Initialize capital gains tax rates"""
        return {
            "short_term": 0.37,  # Treated as ordinary income (max rate used for general calc)
            "long_term_0": 0.0,
            "long_term_15": 0.15,
            "long_term_20": 0.20
        }
    
    def _initialize_account_types(self) -> Dict[str, Dict]:
        """Initialize account type characteristics for asset location"""
        return {
            "taxable": {
                "tax_treatment": "immediate",
                "best_assets": ["tax_efficient_funds", "individual_stocks", "municipal_bonds"],
                "avoid_assets": ["bonds", "reits", "high_turnover_funds"],
                "characteristics": ["foreign_tax_credit", "capital_gains_flexibility"]
            },
            "traditional_401k": {
                "tax_treatment": "deferred",
                "best_assets": ["bonds", "reits", "high_dividend_stocks"],
                "avoid_assets": ["tax_efficient_funds", "municipal_bonds"],
                "characteristics": ["ordinary_income_on_withdrawal", "rmd_required"]
            },
            "roth_ira": {
                "tax_treatment": "tax_free",
                "best_assets": ["growth_stocks", "aggressive_funds", "high_potential_assets"],
                "avoid_assets": ["bonds", "conservative_income"],
                "characteristics": ["tax_free_growth", "no_rmd", "contribution_limits"]
            },
            "traditional_ira": {
                "tax_treatment": "deferred",
                "best_assets": ["bonds", "dividend_stocks", "reits"],
                "avoid_assets": ["growth_stocks", "tax_efficient_funds"],
                "characteristics": ["ordinary_income_on_withdrawal", "rmd_required"]
            }
        }
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute tax optimization capabilities"""
        logger.info(f"Executing tax capability: {capability}")
        
        capability_map = {
            "tax_loss_harvesting": self.analyze_tax_loss_harvesting,
            "asset_location_optimization": self.optimize_asset_location,
            "year_end_tax_planning": self.generate_year_end_strategy,
            "after_tax_analysis": self.analyze_after_tax_returns,
            "tax_optimization": self.comprehensive_tax_optimization,
            "comprehensive_tax_optimization": self.comprehensive_tax_optimization,
            "wash_sale_compliance": self.check_wash_sale_compliance,
            "retirement_planning": self.optimize_retirement_contributions,
            "debate_participation": self.participate_in_debate
        }
        
        if capability not in capability_map:
            raise ValueError(f"Capability {capability} not supported by TaxStrategist")
        
        return await capability_map[capability](data, context)
    
    # ==========================================
    # CORE TAX OPTIMIZATION METHODS
    # ==========================================
    
    async def analyze_tax_loss_harvesting(self, data: Dict, context: Dict) -> Dict:
        """Comprehensive tax-loss harvesting analysis with wash sale compliance"""
        try:
            logger.info("Starting tax-loss harvesting analysis")
            portfolio_data = data.get("portfolio_data", {})
            tax_context = data.get("tax_context", {})
            if not portfolio_data or not portfolio_data.get("holdings"):
                return {"error": "No portfolio data available for tax-loss harvesting analysis"}
            
            holdings = portfolio_data.get("holdings", [])
            opportunities = []
            for holding in holdings:
                opportunity = await self._analyze_holding_for_tax_loss(holding, tax_context)
                if opportunity and opportunity.unrealized_loss < 0:
                    opportunities.append(opportunity)
            
            opportunities.sort(key=lambda x: x.priority_score, reverse=True)
            total_tax_benefit = sum(opp.tax_benefit for opp in opportunities)
            implementation_plan = self._create_tax_loss_implementation_plan(opportunities)
            
            return {
                "opportunities": [self._serialize_tax_opportunity(opp) for opp in opportunities],
                "total_potential_benefit": float(total_tax_benefit),
                "opportunities_count": len(opportunities),
                "implementation_plan": implementation_plan,
                "wash_sale_warnings": self._identify_wash_sale_risks(opportunities),
                "optimal_timing": self._determine_optimal_harvest_timing(opportunities),
                "confidence_score": 0.92,
                "methodology": "Comprehensive tax-loss harvesting with wash sale compliance"
            }
        except Exception as e:
            logger.error(f"Tax-loss harvesting analysis failed: {str(e)}")
            return {"error": f"Tax-loss harvesting failed: {str(e)}"}
    
    async def optimize_asset_location(self, data: Dict, context: Dict) -> Dict:
        """Optimize asset placement across different account types"""
        try:
            logger.info("Starting asset location optimization")
            accounts = data.get("accounts", [])
            tax_context = data.get("tax_context", {})
            if not accounts:
                return {"error": "No account data available for asset location optimization"}
            
            current_efficiency = self._calculate_current_tax_efficiency(accounts)
            recommendations = []
            total_projected_savings = 0.0
            for account in accounts:
                account_recommendations = await self._optimize_account_allocation(account, accounts, tax_context)
                recommendations.extend(account_recommendations)
                total_projected_savings += sum(rec.tax_savings for rec in account_recommendations)
            
            implementation_roadmap = self._create_asset_location_roadmap(recommendations)
            
            return {
                "current_efficiency_score": current_efficiency,
                "recommendations": [self._serialize_location_recommendation(rec) for rec in recommendations],
                "projected_annual_savings": float(total_projected_savings),
                "implementation_roadmap": implementation_roadmap,
                "account_optimization_scores": self._score_account_optimizations(accounts),
                "tax_drag_analysis": self._calculate_tax_drag_by_account(accounts, tax_context),
                "confidence_score": 0.88,
                "methodology": "Tax-efficient asset location optimization"
            }
        except Exception as e:
            logger.error(f"Asset location optimization failed: {str(e)}")
            return {"error": f"Asset location optimization failed: {str(e)}"}
    
    async def generate_year_end_strategy(self, data: Dict, context: Dict) -> Dict:
        """Generate comprehensive year-end tax planning strategy"""
        try:
            logger.info("Generating year-end tax strategy")
            portfolio_data = data.get("portfolio_data", {})
            tax_context = data.get("tax_context", {})
            income_data = data.get("income_data", {})
            
            strategy = {"tax_year": datetime.now().year, "strategies": {}}
            
            tlh_analysis = await self.analyze_tax_loss_harvesting(data, context)
            if "error" not in tlh_analysis: strategy["strategies"]["tax_loss_harvesting"] = tlh_analysis
            
            gains_strategy = await self._analyze_capital_gains_realization(portfolio_data, tax_context)
            strategy["strategies"]["capital_gains_realization"] = gains_strategy
            
            retirement_strategy = await self.optimize_retirement_contributions(income_data, tax_context)
            strategy["strategies"]["retirement_contributions"] = retirement_strategy
            
            charitable_strategy = await self._analyze_charitable_strategies(portfolio_data, tax_context)
            strategy["strategies"]["charitable_giving"] = charitable_strategy
            
            strategy["estimated_total_savings"] = self._calculate_total_strategy_impact(strategy["strategies"])
            strategy["implementation_timeline"] = self._create_year_end_timeline(strategy["strategies"])
            strategy["priority_actions"] = self._identify_priority_year_end_actions(strategy["strategies"])
            strategy["confidence_score"] = 0.90
            
            return strategy
        except Exception as e:
            logger.error(f"Year-end strategy generation failed: {str(e)}")
            return {"error": f"Year-end strategy failed: {str(e)}"}
    
    async def analyze_after_tax_returns(self, data: Dict, context: Dict) -> Dict:
        """Analyze after-tax returns and efficiency"""
        try:
            portfolio_data = data.get("portfolio_data", {})
            tax_context = data.get("tax_context", {})
            holdings = portfolio_data.get("holdings", [])
            analysis = {"holdings_analysis": [], "portfolio_summary": {}}
            
            total_pre_tax_return = 0.0
            total_after_tax_return = 0.0
            total_value = 0.0
            
            for holding in holdings:
                holding_analysis = self._analyze_holding_after_tax_return(holding, tax_context)
                analysis["holdings_analysis"].append(holding_analysis)
                value = holding.get("current_value", 0)
                total_value += value
                total_pre_tax_return += holding_analysis.get("pre_tax_return_amount", 0)
                total_after_tax_return += holding_analysis.get("after_tax_return_amount", 0)
            
            if total_value > 0:
                portfolio_pre_tax = total_pre_tax_return / total_value
                portfolio_after_tax = total_after_tax_return / total_value
                tax_drag = portfolio_pre_tax - portfolio_after_tax
                efficiency = (portfolio_after_tax / portfolio_pre_tax) if portfolio_pre_tax > 0 else 1.0
                
                analysis["portfolio_summary"] = {
                    "pre_tax_return": float(portfolio_pre_tax),
                    "after_tax_return": float(portfolio_after_tax),
                    "tax_drag": float(tax_drag),
                    "tax_efficiency": float(efficiency)
                }
                analysis["tax_efficiency_score"] = min(efficiency * 100, 100.0)
            
            analysis["improvement_opportunities"] = self._identify_tax_efficiency_improvements(analysis["holdings_analysis"])
            analysis["confidence_score"] = 0.85
            return analysis
        except Exception as e:
            logger.error(f"After-tax analysis failed: {str(e)}")
            return {"error": f"After-tax analysis failed: {str(e)}"}
    
    async def comprehensive_tax_optimization(self, data: Dict, context: Dict) -> Dict:
        """Execute comprehensive tax optimization across all strategies"""
        try:
            logger.info("Starting comprehensive tax optimization")
            results = {
                "tax_loss_harvesting": await self.analyze_tax_loss_harvesting(data, context),
                "asset_location": await self.optimize_asset_location(data, context),
                "after_tax_analysis": await self.analyze_after_tax_returns(data, context),
                "year_end_strategy": await self.generate_year_end_strategy(data, context)
            }
            comprehensive_analysis = self._synthesize_tax_optimization_results(results)
            return {
                "detailed_analyses": results,
                "comprehensive_recommendations": comprehensive_analysis["recommendations"],
                "total_estimated_savings": comprehensive_analysis["total_savings"],
                "priority_actions": comprehensive_analysis["priority_actions"],
                "implementation_timeline": comprehensive_analysis["timeline"],
                "confidence_score": 0.91,
                "methodology": "Comprehensive multi-strategy tax optimization"
            }
        except Exception as e:
            logger.error(f"Comprehensive tax optimization failed: {str(e)}")
            return {"error": f"Comprehensive tax optimization failed: {str(e)}"}
    
    async def check_wash_sale_compliance(self, data: Dict, context: Dict) -> Dict:
        """Check wash sale compliance for planned transactions"""
        try:
            transactions = data.get("planned_transactions", [])
            portfolio_data = data.get("portfolio_data", {})
            report = {"compliant_transactions": [], "violations": [], "warnings": []}
            for transaction in transactions:
                check = self._check_transaction_wash_sale(transaction, portfolio_data)
                if check["status"] == "compliant": report["compliant_transactions"].append(check)
                elif check["status"] == "violation": report["violations"].append(check)
                else: report["warnings"].append(check)
            return report
        except Exception as e:
            logger.error(f"Wash sale compliance check failed: {str(e)}")
            return {"error": f"Wash sale compliance check failed: {str(e)}"}
    
    async def optimize_retirement_contributions(self, data: Dict, context: Dict) -> Dict:
        """Optimize retirement account contributions"""
        try:
            income_data = data.get("income_data", {})
            tax_context = data.get("tax_context", {})
            current_income = income_data.get("annual_income", 100000)
            age = income_data.get("age", 35)
            return self._calculate_optimal_retirement_contributions(current_income, age, tax_context)
        except Exception as e:
            logger.error(f"Retirement contribution optimization failed: {str(e)}")
            return {"error": f"Retirement optimization failed: {str(e)}"}

    # ==========================================
    # DEBATE SYSTEM METHODS
    # ==========================================
    
    async def formulate_debate_position(self, query: str, context: Dict, debate_context: Dict = None) -> Dict:
        """Generate tax-focused debate position"""
        themes = self._extract_tax_themes(query)
        stance_map = {
            "risk": "prioritize tax-efficient risk management strategies",
            "growth": "maximize after-tax growth through tax-efficient strategies",
            "allocation": "implement tax-aware allocation and rebalancing strategies"
        }
        stance = stance_map.get(themes[0] if themes else "default", "integrate comprehensive tax planning into all investment decisions")
        arguments = self._get_arguments_for_stance(stance)
        
        portfolio_data = context.get("portfolio_data", {})
        if portfolio_data:
            tax_analysis = await self.comprehensive_tax_optimization({"portfolio_data": portfolio_data, "tax_context": context.get("tax_context", {})}, context)
            evidence = self._convert_tax_analysis_to_evidence(tax_analysis)
        else:
            evidence = self._get_default_tax_evidence()
        
        return {
            "stance": stance, "key_arguments": arguments, "supporting_evidence": evidence,
            "risk_assessment": {"primary_risks": ["Tax inefficiency", "Regulatory changes"], "mitigation_strategies": ["Diversified tax strategies", "Compliance monitoring"]},
            "confidence_score": 0.90, "perspective_bias": self.debate_config["bias"], "argument_style": self.debate_config["argument_style"]
        }

    async def respond_to_challenge(self, challenge: str, original_position: Dict, challenge_context: Dict = None) -> Dict:
        """Respond to challenges with tax expertise"""
        challenge_type = self._classify_tax_challenge(challenge)
        response_map = {
            "complexity": ("complexity_management", ["Tax optimization complexity is manageable with systematic approaches", "Automated strategies reduce implementation complexity"]),
            "risk": ("risk_mitigation", ["Tax strategies are based on established regulatory frameworks", "Conservative planning minimizes audit risk"]),
            "effectiveness": ("evidence_based_defense", ["Tax efficiency is a proven method for enhancing returns", "After-tax analysis provides more accurate evaluation"])
        }
        strategy, arguments = response_map.get(challenge_type, ("evidence_based_defense", ["Systematic tax optimization compounds benefits over time"]))
        
        return {
            "response_strategy": strategy, "counter_arguments": arguments,
            "supporting_evidence": original_position.get("supporting_evidence", []),
            "acknowledgments": ["Tax planning requires careful consideration of individual circumstances"],
            "updated_position": original_position, "confidence_change": 0.0, "rebuttal_strength": 0.85
        }

    async def participate_in_debate(self, data: Dict, context: Dict) -> Dict:
        """Participate in MCP-orchestrated debate"""
        debate_request = data.get("debate_request", {})
        query, role = debate_request.get("query", ""), debate_request.get("role", "participant")
        if role == "position": return await self.formulate_debate_position(query, context, debate_request)
        elif role == "challenge": return await self.respond_to_challenge(debate_request.get("challenge", ""), debate_request.get("original_position", {}), debate_request)
        else: return {"error": f"Unknown debate role: {role}"}
    
    # ==========================================
    # HELPER & IMPLEMENTED MISSING METHODS
    # ==========================================

    # --- SERIALIZATION HELPERS ---
    def _serialize_tax_opportunity(self, opp: TaxLossOpportunity) -> Dict:
        """Converts a TaxLossOpportunity dataclass object to a dictionary."""
        return asdict(opp)

    def _serialize_location_recommendation(self, rec: AssetLocationRecommendation) -> Dict:
        """Converts an AssetLocationRecommendation dataclass object to a dictionary."""
        return asdict(rec)

    # --- TAX-LOSS HARVESTING HELPERS ---
    async def _analyze_holding_for_tax_loss(self, holding: Dict, tax_context: Dict) -> Optional[TaxLossOpportunity]:
        """Analyze individual holding for tax-loss harvesting opportunity"""
        current_price, cost_basis, shares = holding.get("current_price", 0), holding.get("cost_basis", 0), holding.get("shares", 0)
        unrealized_gain_loss = (current_price - cost_basis) * shares
        if unrealized_gain_loss >= 0: return None
        
        tax_rate = tax_context.get("marginal_tax_rate", 0.24)
        tax_benefit = abs(min(unrealized_gain_loss, 3000)) * tax_rate
        
        wash_sale_risk = await self._check_wash_sale_risk(holding, tax_context)
        replacement_securities = await self._find_replacement_securities(holding)
        optimal_timing = self._determine_harvest_timing(holding, tax_context)
        priority_score = self._calculate_tax_opportunity_priority(tax_benefit, unrealized_gain_loss, wash_sale_risk)
        
        return TaxLossOpportunity(
            symbol=holding.get("symbol", ""), current_price=current_price, cost_basis=cost_basis,
            unrealized_loss=unrealized_gain_loss, tax_benefit=tax_benefit, wash_sale_risk=wash_sale_risk,
            replacement_securities=replacement_securities, optimal_timing=optimal_timing, priority_score=priority_score
        )

    async def _check_wash_sale_risk(self, holding: Dict, tax_context: Dict) -> str:
        """Check for wash sale rule violations (simplified)."""
        # In a real system, this would check transaction history.
        high_risk = {"SPY", "QQQ", "VTI", "AAPL", "MSFT"}
        return "medium" if holding.get("symbol") in high_risk else "low"

    async def _find_replacement_securities(self, holding: Dict) -> List[str]:
        """Find suitable replacement securities for tax-loss harvesting."""
        replacements = {"SPY": ["IVV", "VOO"], "QQQ": ["QQQM", "ONEQ"], "VTI": ["ITOT", "SCHB"]}
        return replacements.get(holding.get("symbol"), ["Similar sector ETF"])

    def _determine_harvest_timing(self, holding: Dict, tax_context: Dict) -> str:
        """Determine optimal timing for tax-loss harvesting."""
        days_to_year_end = (datetime(datetime.now().year, 12, 31) - datetime.now()).days
        if days_to_year_end < 60: return "before_year_end"
        return "q4_optimal" if days_to_year_end < 120 else "flexible_timing"

    def _calculate_tax_opportunity_priority(self, tax_benefit: float, unrealized_loss: float, wash_sale_risk: str) -> float:
        """Calculate priority score for tax-loss opportunity."""
        base_score = min(tax_benefit / 500, 10.0)
        risk_multiplier = {"low": 1.0, "medium": 0.8, "high": 0.5}
        return (base_score * risk_multiplier.get(wash_sale_risk, 0.7)) + min(abs(unrealized_loss) / 5000, 2.0)

    def _create_tax_loss_implementation_plan(self, opportunities: List[TaxLossOpportunity]) -> Dict:
        """Create implementation plan for tax-loss harvesting."""
        plan = {"immediate_actions": [], "q4_actions": [], "ongoing_monitoring": []}
        for opp in opportunities:
            action_item = {
                "symbol": opp.symbol, "action": f"Sell to realize ${opp.unrealized_loss:,.2f} loss",
                "replacement": f"Consider {opp.replacement_securities[0]}", "tax_benefit": f"${opp.tax_benefit:,.2f}"
            }
            if opp.optimal_timing == "before_year_end": plan["immediate_actions"].append(action_item)
            elif opp.optimal_timing == "q4_optimal": plan["q4_actions"].append(action_item)
            else: plan["ongoing_monitoring"].append({"symbol": opp.symbol})
        return plan

    def _identify_wash_sale_risks(self, opportunities: List[TaxLossOpportunity]) -> List[Dict]:
        """Identify and explain wash sale risks."""
        return [{
            "symbol": opp.symbol, "risk_level": opp.wash_sale_risk,
            "mitigation": f"Use {opp.replacement_securities[0]} as replacement and wait 31 days before repurchasing {opp.symbol}."
        } for opp in opportunities if opp.wash_sale_risk in ["medium", "high"]]
    
    def _determine_optimal_harvest_timing(self, opportunities: List[TaxLossOpportunity]) -> Dict:
        """Determine overall optimal timing for harvest execution."""
        counts = {"before_year_end": 0, "q4_optimal": 0, "flexible_timing": 0}
        for opp in opportunities: counts[opp.optimal_timing] += 1
        return counts

    # --- ASSET LOCATION HELPERS ---
    def _calculate_current_tax_efficiency(self, accounts: List[Dict]) -> float:
        """Calculates a tax-efficiency score from 0-100 for the current asset location."""
        score, total_value = 0, 0
        for acc in accounts:
            acc_type = acc.get("type")
            if not acc_type or acc_type not in self.account_types: continue
            
            ideal_assets = set(self.account_types[acc_type]["best_assets"])
            for holding in acc.get("holdings", []):
                value = holding.get("current_value", 0)
                asset_class = holding.get("asset_class", "individual_stocks") # Default
                
                # Simple scoring: 1 if ideal, 0.2 if not.
                holding_score = 1.0 if asset_class in ideal_assets else 0.2
                score += holding_score * value
                total_value += value
        
        return (score / total_value * 100) if total_value > 0 else 100.0

    async def _optimize_account_allocation(self, account: Dict, all_accounts: List[Dict], tax_context: Dict) -> List[AssetLocationRecommendation]:
        """Generates recommendations for moving assets to more tax-efficient accounts."""
        recommendations = []
        current_acc_type = account.get("type")
        if not current_acc_type: return []

        avoid_assets = set(self.account_types[current_acc_type].get("avoid_assets", []))
        
        for holding in account.get("holdings", []):
            asset_class = holding.get("asset_class")
            if asset_class in avoid_assets:
                # Find a better home
                for potential_account in all_accounts:
                    potential_acc_type = potential_account.get("type")
                    if potential_acc_type and asset_class in self.account_types[potential_acc_type].get("best_assets", []):
                        tax_savings = holding.get("current_value", 0) * 0.0075 # Estimate 0.75% annual savings
                        recommendations.append(AssetLocationRecommendation(
                            asset_type=asset_class, current_account=current_acc_type,
                            recommended_account=potential_acc_type, tax_savings=tax_savings,
                            reasoning=f"{asset_class} is better suited for a {potential_acc_type} account.",
                            implementation_complexity="Medium"
                        ))
                        break # Found a home, move to next holding
        return recommendations
        
    def _create_asset_location_roadmap(self, recommendations: List[AssetLocationRecommendation]) -> Dict:
        """Creates a step-by-step roadmap for asset location changes."""
        if not recommendations: return {"message": "Current asset location is optimal."}
        roadmap = {"priority_actions": [], "notes": []}
        for i, rec in enumerate(recommendations):
            action = f"Step {i+1}: Move assets of type '{rec.asset_type}' from your {rec.current_account} account to your {rec.recommended_account} account."
            roadmap["priority_actions"].append(action)
        roadmap["notes"].append("Warning: Selling assets in a taxable account may trigger capital gains. Plan accordingly.")
        return roadmap

    def _score_account_optimizations(self, accounts: List[Dict]) -> Dict:
        """Scores each account on how well its assets are placed."""
        scores = {}
        for acc in accounts:
            acc_type = acc.get("type")
            if not acc_type: continue
            
            ideal_assets = set(self.account_types[acc_type]["best_assets"])
            value_in_ideal = sum(h.get("current_value",0) for h in acc.get("holdings", []) if h.get("asset_class") in ideal_assets)
            total_value = sum(h.get("current_value",0) for h in acc.get("holdings", []))
            scores[acc_type] = (value_in_ideal / total_value * 100) if total_value > 0 else 100.0
        return scores

    def _calculate_tax_drag_by_account(self, accounts: List[Dict], tax_context: Dict) -> Dict:
        """Estimates the annual cost of taxes ('tax drag') for each account."""
        drag = {}
        tax_rate = tax_context.get("marginal_tax_rate", 0.24)
        for acc in accounts:
            acc_type = acc.get("type")
            if acc_type == "taxable":
                acc_drag = 0
                for h in acc.get("holdings",[]):
                    # Simplified drag: 1% for inefficient assets, 0.2% for efficient
                    asset_class = h.get("asset_class")
                    multiplier = 0.01 if asset_class in self.account_types["taxable"]["avoid_assets"] else 0.002
                    acc_drag += h.get("current_value", 0) * multiplier
                drag[acc_type] = acc_drag
            else: # Tax-advantaged accounts have no drag
                drag[acc_type] = 0.0
        return drag

    # --- YEAR-END & COMPREHENSIVE HELPERS ---
    async def _analyze_capital_gains_realization(self, portfolio_data: Dict, tax_context: Dict) -> Dict:
        """Analyzes opportunities for 'gain harvesting' at low tax rates."""
        income = tax_context.get("annual_income", 100000)
        ltcg_0_bracket_limit = self.tax_brackets["long_term_capital_gains"][1][0]
        
        harvestable_gains = ltcg_0_bracket_limit - income
        if harvestable_gains <= 0:
            return {"message": "Income level too high for 0% LTCG harvesting."}
        
        opportunities = []
        for h in portfolio_data.get("holdings", []):
            gain = (h.get("current_price", 0) - h.get("cost_basis", 0)) * h.get("shares", 0)
            if gain > 0 and h.get("holding_period", 366) > 365: # Is long term
                opportunities.append({"symbol": h.get("symbol"), "realizable_gain": gain})
        
        opportunities.sort(key=lambda x: x['realizable_gain'])
        return {
            "message": f"Opportunity to realize up to ${harvestable_gains:,.2f} in long-term capital gains at a 0% federal tax rate.",
            "potential_opportunities": opportunities
        }

    async def _analyze_charitable_strategies(self, portfolio_data: Dict, tax_context: Dict) -> Dict:
        """Identifies highly appreciated assets suitable for charitable donation."""
        recommendations = []
        for h in portfolio_data.get("holdings",[]):
            gain = (h.get("current_price", 0) - h.get("cost_basis", 0)) * h.get("shares", 0)
            if gain > 5000 and h.get("holding_period", 366) > 365: # LTCG > $5k
                recommendations.append(f"Consider donating shares of {h.get('symbol')} directly to charity to avoid capital gains tax on ${gain:,.2f} of appreciation.")
        
        if not recommendations: return {"message": "No standout opportunities for donating appreciated stock."}
        return {"recommendations": recommendations}

    def _calculate_total_strategy_impact(self, strategies: Dict) -> float:
        """Sums the estimated savings from all substrategies."""
        total_savings = 0.0
        total_savings += strategies.get("tax_loss_harvesting", {}).get("total_potential_benefit", 0.0)
        # Add other potential savings sources here as they are quantified
        return total_savings

    def _create_year_end_timeline(self, strategies: Dict) -> List[Dict]:
        """Creates a chronological action plan for year-end."""
        timeline = []
        if strategies.get("tax_loss_harvesting", {}).get("opportunities_count", 0) > 0:
            timeline.append({"action": "Execute final tax-loss harvesting trades.", "deadline": "December 28", "priority": "High"})
        if strategies.get("charitable_giving", {}).get("recommendations"):
            timeline.append({"action": "Complete donations of appreciated stock.", "deadline": "December 20", "priority": "Medium"})
        if strategies.get("retirement_contributions"):
            timeline.append({"action": "Ensure all retirement accounts are maxed out for the year.", "deadline": "December 31", "priority": "High"})
        return timeline
        
    def _identify_priority_year_end_actions(self, strategies: Dict) -> List[str]:
        """Extracts the most impactful recommendations."""
        actions = []
        tlh_benefit = strategies.get("tax_loss_harvesting", {}).get("total_potential_benefit", 0)
        if tlh_benefit > 100:
             actions.append(f"Prioritize tax-loss harvesting to capture an estimated ${tlh_benefit:,.2f} tax benefit.")
        if strategies.get("capital_gains_realization", {}).get("potential_opportunities"):
            actions.append("Consider harvesting long-term capital gains at a 0% tax rate.")
        return actions if actions else ["Review portfolio and finalize any remaining trades for the year."]

    def _synthesize_tax_optimization_results(self, results: Dict) -> Dict:
        """Combines all analyses into a single, comprehensive recommendation object."""
        return {
            "recommendations": "A synthesized list of top recommendations.",
            "total_savings": self._calculate_total_strategy_impact(results),
            "priority_actions": self._identify_priority_year_end_actions(results.get("year_end_strategy", {})),
            "timeline": self._create_year_end_timeline(results.get("year_end_strategy", {}))
        }

    # --- AFTER-TAX RETURN HELPERS ---
    def _analyze_holding_after_tax_return(self, holding: Dict, tax_context: Dict) -> Dict:
        """Analyzes the tax drag on a single holding."""
        value = holding.get("current_value", 0)
        pre_tax_return = holding.get("annual_return_pct", 0.08) # Assume 8% if not given
        dividend_yield = holding.get("dividend_yield", 0.02) # Assume 2%
        
        tax_rate = tax_context.get("marginal_tax_rate", 0.24)
        dividend_tax = dividend_yield * tax_rate
        
        # Assume tax on capital gains from turnover is 0.5%
        turnover_tax_drag = 0.005
        
        total_tax_drag_pct = dividend_tax + turnover_tax_drag
        after_tax_return_pct = pre_tax_return - total_tax_drag_pct
        
        return {
            "symbol": holding.get("symbol"),
            "pre_tax_return_pct": pre_tax_return,
            "after_tax_return_pct": after_tax_return_pct,
            "tax_drag_pct": total_tax_drag_pct,
            "pre_tax_return_amount": pre_tax_return * value,
            "after_tax_return_amount": after_tax_return_pct * value
        }

    def _identify_tax_efficiency_improvements(self, holdings_analysis: List[Dict]) -> List[str]:
        """Suggests improvements for holdings with the highest tax drag."""
        holdings_analysis.sort(key=lambda x: x.get('tax_drag_pct', 0), reverse=True)
        recommendations = []
        for holding in holdings_analysis[:2]: # Top 2
            if holding['tax_drag_pct'] > 0.01: # Drag > 1%
                recommendations.append(f"Consider replacing {holding['symbol']} with a more tax-efficient alternative to reduce its {holding['tax_drag_pct']:.2%} annual tax drag.")
        return recommendations

    # --- RETIREMENT & WASH SALE HELPERS ---
    def _calculate_optimal_retirement_contributions(self, current_income: float, age: int, tax_context: Dict) -> Dict:
        """Recommends a split between Traditional and Roth contributions."""
        # Rule: If in 24% bracket or higher, favor Traditional. Otherwise, Roth.
        if current_income > self.tax_brackets["ordinary_income"][3][0]: # > $100,525
            recommendation = "Prioritize Traditional (pre-tax) contributions to lower your current taxable income."
            split = {"traditional_pct": 80, "roth_pct": 20}
        else:
            recommendation = "Prioritize Roth (post-tax) contributions to pay taxes now while in a lower bracket and enjoy tax-free growth."
            split = {"traditional_pct": 20, "roth_pct": 80}
            
        if age < 35: # Youth bonus for Roth
            split["roth_pct"] = min(100, split["roth_pct"] + 15)
            split["traditional_pct"] = 100 - split["roth_pct"]

        return {"recommendation": recommendation, "suggested_split": split, "contribution_limit_2025": 23000}

    def _check_transaction_wash_sale(self, transaction: Dict, portfolio_data: Dict) -> Dict:
        """Checks a single planned transaction for wash sale violations."""
        # This is a simplified simulation
        if transaction.get("type") != "sell" or transaction.get("gain_loss", 0) >= 0:
            return {"symbol": transaction.get("symbol"), "status": "compliant", "reason": "Not a sale at a loss."}
            
        symbol = transaction.get("symbol")
        # Assume transaction_history exists in portfolio_data
        history = portfolio_data.get("transaction_history", [])
        
        sale_date = datetime.now()
        thirty_days_ago = sale_date - timedelta(days=31)
        
        for past_txn in history:
            txn_date = datetime.fromisoformat(past_txn.get("date"))
            if past_txn.get("symbol") == symbol and past_txn.get("type") == "buy" and txn_date > thirty_days_ago:
                return {"symbol": symbol, "status": "violation", "reason": f"A purchase of {symbol} was made on {txn_date.date()}, which is within the 30-day window."}
                
        return {"symbol": symbol, "status": "compliant", "reason": "No recent purchases of this security found."}

    # --- DEBATE SYSTEM HELPERS ---
    def _extract_tax_themes(self, query: str) -> List[str]:
        """Extracts key investment themes from a debate query."""
        query = query.lower()
        themes = []
        if any(w in query for w in ["risk", "danger", "safe", "protect"]): themes.append("risk")
        if any(w in query for w in ["growth", "opportunity", "return", "gain"]): themes.append("growth")
        if any(w in query for w in ["allocate", "balance", "rebalance", "diversify"]): themes.append("allocation")
        return themes
        
    def _get_arguments_for_stance(self, stance: str) -> List[str]:
        """Provides default arguments for a given debate stance."""
        if "risk management" in stance: return ["After-tax risk-adjusted returns provide more accurate risk assessment.", "Tax-loss harvesting can offset portfolio volatility impacts."]
        if "after-tax growth" in stance: return ["Tax-efficient growth vehicles like Roth accounts provide superior long-term outcomes.", "Strategic tax-loss harvesting enables greater risk-taking capacity."]
        return ["Tax optimization can add 1-2% annually to investment returns.", "After-tax analysis reveals true investment performance."]

    def _convert_tax_analysis_to_evidence(self, tax_analysis: Dict) -> List[str]:
        """Converts numerical analysis results into qualitative debate evidence."""
        evidence = []
        savings = tax_analysis.get("total_estimated_savings", 0)
        if savings > 100:
            evidence.append(f"My comprehensive analysis of this portfolio shows potential tax savings of approximately ${savings:,.2f} through strategies like tax-loss harvesting.")
        
        asset_location = tax_analysis.get("detailed_analyses", {}).get("asset_location", {})
        efficiency = asset_location.get("current_efficiency_score")
        if efficiency and efficiency < 80:
            evidence.append(f"The current asset location has a tax-efficiency score of only {efficiency:.0f}/100, indicating significant room for improvement by optimizing asset placement.")
        
        return evidence if evidence else self._get_default_tax_evidence()

    def _get_default_tax_evidence(self) -> List[str]:
        """Provides generic, universally applicable tax arguments."""
        return [
            "Industry studies consistently show that systematic tax optimization can add 1-2% in alpha to a portfolio annually.",
            "Failing to manage taxes is equivalent to accepting a guaranteed, permanent loss on your investments each year.",
            "Asset location optimization is one of the most effective, yet often overlooked, strategies for long-term wealth accumulation."
        ]

    def _classify_tax_challenge(self, challenge: str) -> str:
        """Classifies the nature of a challenge to the agent's position."""
        challenge = challenge.lower()
        if any(w in challenge for w in ["complex", "hard", "confusing", "difficult"]): return "complexity"
        if any(w in challenge for w in ["risk", "danger", "audit", "unsafe"]): return "risk"
        if any(w in challenge for w in ["worth it", "effective", "impact", "matter"]): return "effectiveness"
        return "general"