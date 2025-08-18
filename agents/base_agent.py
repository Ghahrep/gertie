# agents/base_agent.py (Enhanced for Debate Capabilities)
"""
Enhanced Base Agent with Debate Capabilities
===========================================
Transforms agents into specialized debaters with distinct investment philosophies
and sophisticated argument formulation capabilities.
"""

from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
import json

class DebatePerspective(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive" 
    BALANCED = "balanced"
    SPECIALIST = "specialist"

class DebateStyle(Enum):
    EVIDENCE_DRIVEN = "evidence_driven"
    RISK_FOCUSED = "risk_focused"
    OPPORTUNITY_FOCUSED = "opportunity_focused"
    ANALYTICAL = "analytical"
    PRACTICAL = "practical"

class BaseAgent(ABC):
    """Enhanced base agent with sophisticated debate capabilities"""
    
    def __init__(self, agent_id: str, perspective: DebatePerspective = DebatePerspective.BALANCED):
        self.agent_id = agent_id
        self.perspective = perspective
        self.debate_style = self._determine_debate_style()
        self.specialization = self._get_specialization()
        self.debate_strengths = self._get_debate_strengths()
        
        # Debate personality configuration
        self.debate_config = self._configure_debate_personality()
        
        # Performance tracking
        self.debate_history = []
        self.performance_metrics = {
            "arguments_accepted": 0,
            "positions_defended": 0,
            "consensus_contributions": 0,
            "accuracy_score": 0.5
        }
    
    def _determine_debate_style(self) -> DebateStyle:
        """Determine debate style based on agent type and perspective"""
        style_mapping = {
            "quantitative_analyst": DebateStyle.EVIDENCE_DRIVEN,
            "market_intelligence": DebateStyle.OPPORTUNITY_FOCUSED,
            "tax_strategist": DebateStyle.ANALYTICAL,
            "options_analyst": DebateStyle.RISK_FOCUSED,
            "economic_data": DebateStyle.ANALYTICAL
        }
        return style_mapping.get(self.agent_id, DebateStyle.EVIDENCE_DRIVEN)
    
    @abstractmethod
    def _get_specialization(self) -> str:
        """Get agent's primary specialization"""
        pass
    
    @abstractmethod 
    def _get_debate_strengths(self) -> List[str]:
        """Get agent's debate strengths and focus areas"""
        pass
    
    def _configure_debate_personality(self) -> Dict:
        """Configure agent's debate personality based on perspective"""
        
        base_configs = {
            DebatePerspective.CONSERVATIVE: {
                "focus": "risk_mitigation",
                "evidence_preference": "historical_data",
                "bias": "downside_protection",
                "argument_style": "cautious_analytical",
                "challenge_approach": "risk_highlighting",
                "confidence_threshold": 0.7  # Requires high confidence to take strong positions
            },
            DebatePerspective.AGGRESSIVE: {
                "focus": "growth_opportunities",
                "evidence_preference": "forward_looking",
                "bias": "upside_potential", 
                "argument_style": "opportunity_driven",
                "challenge_approach": "opportunity_cost_focus",
                "confidence_threshold": 0.5  # More willing to take positions with moderate confidence
            },
            DebatePerspective.BALANCED: {
                "focus": "risk_adjusted_returns",
                "evidence_preference": "comprehensive_analysis",
                "bias": "optimal_allocation",
                "argument_style": "balanced_synthesis",
                "challenge_approach": "holistic_consideration",
                "confidence_threshold": 0.6
            },
            DebatePerspective.SPECIALIST: {
                "focus": "domain_expertise",
                "evidence_preference": "specialized_analysis",
                "bias": "technical_accuracy",
                "argument_style": "expert_technical",
                "challenge_approach": "technical_precision",
                "confidence_threshold": 0.8  # High confidence in specialized domain
            }
        }
        
        return base_configs.get(self.perspective, base_configs[DebatePerspective.BALANCED])
    
    async def formulate_debate_position(self, query: str, context: Dict, debate_context: Dict) -> Dict:
        """Generate agent's initial position for debate"""
        
        # Analyze query through agent's perspective lens
        query_analysis = await self._analyze_query_from_perspective(query, context)
        
        # Gather supporting evidence aligned with agent style
        evidence = await self._gather_perspective_evidence(query_analysis, context)
        
        # Formulate clear, defendable position
        position = await self._formulate_position(query_analysis, evidence, context)
        
        # Prepare counter-arguments to expected opposition
        counter_prep = await self._prepare_counter_arguments(position, debate_context)
        
        # Calculate confidence in position
        confidence = self._calculate_position_confidence(position, evidence)
        
        debate_position = {
            "stance": position["stance"],
            "key_arguments": position["arguments"],
            "supporting_evidence": evidence,
            "risk_assessment": position["risk_assessment"],
            "confidence_score": confidence,
            "expected_counterarguments": counter_prep["expected_challenges"],
            "prepared_responses": counter_prep["prepared_responses"],
            "perspective_bias": self.debate_config["bias"],
            "argument_style": self.debate_config["argument_style"]
        }
        
        return debate_position
    
    async def respond_to_challenge(self, challenge: str, original_position: Dict, challenge_context: Dict) -> Dict:
        """Respond to challenges from other agents"""
        
        # Analyze the challenge and identify key points
        challenge_analysis = await self._analyze_challenge(challenge, challenge_context)
        
        # Determine response strategy based on challenge type
        response_strategy = self._determine_response_strategy(challenge_analysis, original_position)
        
        # Formulate evidence-based counter-response
        counter_response = await self._formulate_counter_response(
            challenge_analysis, original_position, response_strategy
        )
        
        # Acknowledge valid opposing points when appropriate
        acknowledgments = self._identify_valid_points(challenge_analysis)
        
        # Strengthen or modify position based on new evidence
        updated_position = await self._update_position_from_challenge(
            original_position, challenge_analysis, acknowledgments
        )
        
        response = {
            "response_strategy": response_strategy,
            "counter_arguments": counter_response["arguments"],
            "supporting_evidence": counter_response["evidence"],
            "acknowledgments": acknowledgments,
            "updated_position": updated_position,
            "confidence_change": updated_position["confidence"] - original_position["confidence_score"],
            "rebuttal_strength": counter_response["strength_score"]
        }
        
        return response
    
    async def _analyze_query_from_perspective(self, query: str, context: Dict) -> Dict:
        """Analyze query through agent's specialized perspective"""
        
        # Extract key themes relevant to this agent
        relevant_themes = self._extract_relevant_themes(query)
        
        # Assess query from perspective bias
        perspective_analysis = self._apply_perspective_filter(query, relevant_themes)
        
        # Identify primary concerns for this agent type
        primary_concerns = self._identify_primary_concerns(query, context)
        
        return {
            "relevant_themes": relevant_themes,
            "perspective_focus": perspective_analysis,
            "primary_concerns": primary_concerns,
            "complexity_assessment": self._assess_query_complexity(query, context),
            "confidence_factors": self._identify_confidence_factors(query, context)
        }
    
    def _extract_relevant_themes(self, query: str) -> List[str]:
        """Extract themes relevant to this agent's specialization"""
        
        query_lower = query.lower()
        
        # Base themes all agents consider
        base_themes = {
            "risk": ["risk", "safe", "conservative", "protect", "downside", "loss"],
            "return": ["return", "profit", "gain", "growth", "performance"],
            "timing": ["when", "timing", "now", "soon", "immediate"],
            "allocation": ["allocate", "balance", "weight", "distribute"],
            "market": ["market", "economic", "environment", "conditions"]
        }
        
        # Add agent-specific themes
        specialized_themes = self._get_specialized_themes()
        all_themes = {**base_themes, **specialized_themes}
        
        relevant = []
        for theme, keywords in all_themes.items():
            if any(keyword in query_lower for keyword in keywords):
                relevant.append(theme)
        
        return relevant
    
    @abstractmethod
    def _get_specialized_themes(self) -> Dict[str, List[str]]:
        """Get themes specific to this agent type"""
        pass
    
    def _apply_perspective_filter(self, query: str, themes: List[str]) -> Dict:
        """Apply agent's perspective bias to query analysis"""
        
        perspective_filters = {
            DebatePerspective.CONSERVATIVE: {
                "primary_lens": "risk_assessment",
                "secondary_concerns": ["capital_preservation", "downside_protection"],
                "opportunity_weighting": 0.3,
                "risk_weighting": 0.7
            },
            DebatePerspective.AGGRESSIVE: {
                "primary_lens": "opportunity_identification",
                "secondary_concerns": ["growth_potential", "market_timing"],
                "opportunity_weighting": 0.7,
                "risk_weighting": 0.3
            },
            DebatePerspective.BALANCED: {
                "primary_lens": "risk_adjusted_optimization",
                "secondary_concerns": ["efficiency", "diversification"],
                "opportunity_weighting": 0.5,
                "risk_weighting": 0.5
            },
            DebatePerspective.SPECIALIST: {
                "primary_lens": "technical_analysis",
                "secondary_concerns": ["domain_expertise", "precision"],
                "opportunity_weighting": 0.4,
                "risk_weighting": 0.6
            }
        }
        
        filter_config = perspective_filters.get(self.perspective, perspective_filters[DebatePerspective.BALANCED])
        
        return {
            "primary_lens": filter_config["primary_lens"],
            "focus_areas": self._map_themes_to_focus(themes, filter_config),
            "weighting": {
                "opportunity": filter_config["opportunity_weighting"],
                "risk": filter_config["risk_weighting"]
            }
        }
    
    def _map_themes_to_focus(self, themes: List[str], filter_config: Dict) -> List[str]:
        """Map query themes to agent's focus areas"""
        
        focus_mapping = {
            "risk": ["risk_analysis", "downside_scenarios", "protection_strategies"],
            "return": ["return_optimization", "performance_analysis", "growth_strategies"],
            "timing": ["market_timing", "entry_exit_points", "temporal_analysis"],
            "allocation": ["portfolio_optimization", "asset_allocation", "balance_strategies"],
            "market": ["market_analysis", "economic_assessment", "environmental_factors"]
        }
        
        mapped_focus = []
        for theme in themes:
            if theme in focus_mapping:
                mapped_focus.extend(focus_mapping[theme])
        
        # Add agent-specific focus areas
        mapped_focus.extend(self.debate_strengths)
        
        return list(set(mapped_focus))  # Remove duplicates
    
    async def _gather_perspective_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Gather evidence aligned with agent's perspective and style"""
        
        evidence = []
        
        # Use agent's specialized analysis capabilities
        specialized_evidence = await self._gather_specialized_evidence(analysis, context)
        evidence.extend(specialized_evidence)
        
        # Apply perspective bias to evidence selection
        filtered_evidence = self._filter_evidence_by_perspective(evidence)
        
        # Enhance evidence quality based on agent's strengths
        enhanced_evidence = self._enhance_evidence_quality(filtered_evidence)
        
        return enhanced_evidence[:5]  # Limit to top 5 pieces of evidence
    
    @abstractmethod
    async def _gather_specialized_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Gather evidence using agent's specialized capabilities"""
        pass
    
    def _filter_evidence_by_perspective(self, evidence: List[Dict]) -> List[Dict]:
        """Filter and prioritize evidence based on agent perspective"""
        
        perspective_priorities = {
            DebatePerspective.CONSERVATIVE: ["historical", "statistical", "risk_analysis"],
            DebatePerspective.AGGRESSIVE: ["forward_looking", "growth", "opportunity"],
            DebatePerspective.BALANCED: ["comprehensive", "analytical", "balanced"],
            DebatePerspective.SPECIALIST: ["technical", "specialized", "domain_specific"]
        }
        
        priorities = perspective_priorities.get(self.perspective, ["analytical"])
        
        # Score evidence based on alignment with perspective
        scored_evidence = []
        for ev in evidence:
            score = 0.0
            ev_type = ev.get("type", "").lower()
            
            for priority in priorities:
                if priority in ev_type or priority in ev.get("analysis", "").lower():
                    score += 1.0
            
            # Boost score for high-confidence evidence
            if ev.get("confidence", 0) > 0.7:
                score += 0.5
            
            scored_evidence.append((ev, score))
        
        # Sort by score and return top evidence
        scored_evidence.sort(key=lambda x: x[1], reverse=True)
        return [ev for ev, score in scored_evidence]
    
    def _enhance_evidence_quality(self, evidence: List[Dict]) -> List[Dict]:
        """Enhance evidence quality based on agent capabilities"""
        
        enhanced = []
        
        for ev in evidence:
            enhanced_ev = ev.copy()
            
            # Add agent-specific analysis
            enhanced_ev["agent_analysis"] = self._add_agent_perspective_to_evidence(ev)
            
            # Add confidence assessment
            enhanced_ev["agent_confidence"] = self._assess_evidence_confidence(ev)
            
            # Add relevance score
            enhanced_ev["relevance_score"] = self._calculate_evidence_relevance(ev)
            
            enhanced.append(enhanced_ev)
        
        return enhanced
    
    def _add_agent_perspective_to_evidence(self, evidence: Dict) -> str:
        """Add agent's perspective analysis to evidence"""
        
        perspective_analysis = {
            DebatePerspective.CONSERVATIVE: f"From a risk management perspective: {evidence.get('analysis', 'Standard analysis')}",
            DebatePerspective.AGGRESSIVE: f"From a growth opportunity perspective: {evidence.get('analysis', 'Standard analysis')}",
            DebatePerspective.BALANCED: f"From a balanced optimization perspective: {evidence.get('analysis', 'Standard analysis')}",
            DebatePerspective.SPECIALIST: f"From a technical {self.specialization} perspective: {evidence.get('analysis', 'Standard analysis')}"
        }
        
        return perspective_analysis.get(self.perspective, evidence.get("analysis", ""))
    
    async def _formulate_position(self, analysis: Dict, evidence: List[Dict], context: Dict) -> Dict:
        """Formulate clear, defendable position based on analysis and evidence"""
        
        # Generate stance based on perspective and evidence
        stance = await self._generate_stance(analysis, evidence)
        
        # Create supporting arguments
        arguments = self._create_supporting_arguments(analysis, evidence)
        
        # Assess risks from perspective
        risk_assessment = await self._assess_risks_from_perspective(analysis, context)
        
        # Generate implementation recommendations
        implementation = self._generate_implementation_recommendations(stance, arguments, risk_assessment)
        
        return {
            "stance": stance,
            "arguments": arguments,
            "risk_assessment": risk_assessment,
            "implementation": implementation,
            "rationale": self._create_position_rationale(stance, arguments, evidence)
        }
    
    @abstractmethod
    async def _generate_stance(self, analysis: Dict, evidence: List[Dict]) -> str:
        """Generate agent's stance based on specialized analysis"""
        pass
    
    def _create_supporting_arguments(self, analysis: Dict, evidence: List[Dict]) -> List[str]:
        """Create arguments supporting the position"""
        
        arguments = []
        
        # Convert evidence into arguments
        for ev in evidence[:3]:  # Top 3 pieces of evidence
            if ev.get("data"):
                arg = f"Analysis shows {ev.get('analysis', 'that')} based on {ev.get('type', 'evidence')}"
                arguments.append(arg)
        
        # Add perspective-specific arguments
        perspective_args = self._generate_perspective_arguments(analysis)
        arguments.extend(perspective_args)
        
        return arguments[:5]  # Limit to 5 key arguments
    
    def _generate_perspective_arguments(self, analysis: Dict) -> List[str]:
        """Generate arguments specific to agent's perspective"""
        
        perspective_arguments = {
            DebatePerspective.CONSERVATIVE: [
                "Risk management should be the primary consideration",
                "Historical data suggests caution is warranted",
                "Downside protection outweighs potential upside"
            ],
            DebatePerspective.AGGRESSIVE: [
                "Market opportunities should be maximized",
                "Growth potential justifies increased risk",
                "Timing favors aggressive positioning"
            ],
            DebatePerspective.BALANCED: [
                "Risk-adjusted returns should guide decisions",
                "Diversification provides optimal balance",
                "Multiple factors must be considered"
            ],
            DebatePerspective.SPECIALIST: [
                f"Technical {self.specialization} analysis indicates",
                "Specialized knowledge provides unique insights",
                "Domain expertise reveals critical factors"
            ]
        }
        
        base_args = perspective_arguments.get(self.perspective, [])
        return [arg for arg in base_args if self._argument_relevant_to_analysis(arg, analysis)]
    
    def _argument_relevant_to_analysis(self, argument: str, analysis: Dict) -> bool:
        """Check if argument is relevant to current analysis"""
        
        themes = analysis.get("relevant_themes", [])
        focus_areas = analysis.get("perspective_focus", {}).get("focus_areas", [])
        
        # Simple relevance check - can be enhanced
        arg_lower = argument.lower()
        for theme in themes + focus_areas:
            if any(word in arg_lower for word in theme.split("_")):
                return True
        
        return True  # Default to relevant for now
    
    async def _assess_risks_from_perspective(self, analysis: Dict, context: Dict) -> Dict:
        """Assess risks from agent's perspective"""
        
        # Get general risk factors
        general_risks = await self._identify_general_risks(context)
        
        # Apply perspective filter to risks
        perspective_risks = self._filter_risks_by_perspective(general_risks)
        
        # Add agent-specific risk insights
        specialized_risks = await self._identify_specialized_risks(analysis, context)
        
        return {
            "primary_risks": perspective_risks[:3],
            "specialized_risks": specialized_risks,
            "mitigation_strategies": self._suggest_risk_mitigation(perspective_risks + specialized_risks),
            "risk_tolerance": self.debate_config["bias"]
        }
    
    @abstractmethod
    async def _identify_general_risks(self, context: Dict) -> List[str]:
        """Identify general risks relevant to the query"""
        pass
    
    @abstractmethod
    async def _identify_specialized_risks(self, analysis: Dict, context: Dict) -> List[str]:
        """Identify risks specific to agent's domain"""
        pass
    
    def _filter_risks_by_perspective(self, risks: List[str]) -> List[str]:
        """Filter risks based on agent's perspective bias"""
        
        if self.perspective == DebatePerspective.CONSERVATIVE:
            # Conservative agents emphasize all risks
            return risks
        elif self.perspective == DebatePerspective.AGGRESSIVE:
            challenges.extend([
                "Excessive risk-taking may lead to significant losses",
                "Growth projections may be overly optimistic",
                "Market timing risks are underestimated"
            ])
        elif self.perspective == DebatePerspective.BALANCED:
            challenges.extend([
                "Balanced approach may lack clear direction",
                "Compromise solutions may be suboptimal",
                "Risk-return optimization may miss extremes"
            ])
        else:  # SPECIALIST
            challenges.extend([
                "Technical analysis may miss broader market factors",
                "Specialized focus may overlook important considerations",
                "Domain expertise may not apply in current conditions"
            ])
        
        return challenges[:3]  # Top 3 expected challenges
    
    async def _prepare_response_to_challenge(self, challenge: str, position: Dict) -> str:
        """Prepare response to a specific challenge"""
        
        # Create defense based on agent's evidence and perspective
        stance = position.get("stance", "")
        arguments = position.get("arguments", [])
        
        # Simple response generation - can be enhanced with NLP
        if "risk" in challenge.lower() and self.perspective == DebatePerspective.CONSERVATIVE:
            return f"Risk management is essential because {arguments[0] if arguments else 'historical evidence supports caution'}"
        elif "opportunity" in challenge.lower() and self.perspective == DebatePerspective.AGGRESSIVE:
            return f"The opportunity cost of inaction outweighs risks because {arguments[0] if arguments else 'market conditions favor growth'}"
        else:
            return f"My analysis shows {stance} based on {arguments[0] if arguments else 'comprehensive evaluation'}"
    
    async def _analyze_challenge(self, challenge: str, context: Dict) -> Dict:
        """Analyze incoming challenge from other agents"""
        
        return {
            "challenge_type": self._classify_challenge_type(challenge),
            "key_points": self._extract_challenge_points(challenge),
            "evidence_mentioned": self._extract_challenge_evidence(challenge),
            "strength_assessment": self._assess_challenge_strength(challenge),
            "response_required": self._determine_response_requirements(challenge)
        }
    
    def _classify_challenge_type(self, challenge: str) -> str:
        """Classify the type of challenge being made"""
        
        challenge_lower = challenge.lower()
        
        if any(word in challenge_lower for word in ["evidence", "data", "proof"]):
            return "evidence_challenge"
        elif any(word in challenge_lower for word in ["risk", "dangerous", "loss"]):
            return "risk_challenge"
        elif any(word in challenge_lower for word in ["opportunity", "miss", "cost"]):
            return "opportunity_challenge"
        elif any(word in challenge_lower for word in ["logic", "reasoning", "flawed"]):
            return "logic_challenge"
        else:
            return "general_disagreement"
    
    def _extract_challenge_points(self, challenge: str) -> List[str]:
        """Extract key points from the challenge"""
        
        # Simple extraction - can be enhanced with NLP
        sentences = challenge.split('.')
        key_points = [s.strip() for s in sentences if len(s.strip()) > 10]
        return key_points[:3]  # Top 3 points
    
    def _extract_challenge_evidence(self, challenge: str) -> List[str]:
        """Extract evidence mentioned in the challenge"""
        
        # Look for evidence indicators
        evidence_indicators = ["shows", "indicates", "proves", "demonstrates", "analysis", "data"]
        evidence_mentions = []
        
        for indicator in evidence_indicators:
            if indicator in challenge.lower():
                # Extract surrounding context
                words = challenge.split()
                for i, word in enumerate(words):
                    if indicator in word.lower():
                        # Get context around the evidence mention
                        start = max(0, i-3)
                        end = min(len(words), i+4)
                        context = ' '.join(words[start:end])
                        evidence_mentions.append(context)
        
        return evidence_mentions
    
    def _assess_challenge_strength(self, challenge: str) -> float:
        """Assess the strength of the incoming challenge"""
        
        strength_score = 0.5  # Base score
        
        # Indicators of strong challenges
        if any(word in challenge.lower() for word in ["data", "evidence", "proof"]):
            strength_score += 0.2
        if any(word in challenge.lower() for word in ["significantly", "major", "critical"]):
            strength_score += 0.15
        if len(challenge.split()) > 50:  # Detailed challenge
            strength_score += 0.1
        
        # Indicators of weak challenges
        if any(word in challenge.lower() for word in ["maybe", "perhaps", "might"]):
            strength_score -= 0.1
        if len(challenge.split()) < 20:  # Too brief
            strength_score -= 0.1
        
        return min(max(strength_score, 0.1), 1.0)
    
    def _determine_response_requirements(self, challenge: str) -> Dict:
        """Determine what type of response is required"""
        
        challenge_type = self._classify_challenge_type(challenge)
        
        response_requirements = {
            "evidence_challenge": {
                "evidence_needed": True,
                "logic_defense": False,
                "counter_evidence": True,
                "acknowledgment": False
            },
            "risk_challenge": {
                "evidence_needed": True,
                "logic_defense": True,
                "counter_evidence": False,
                "acknowledgment": True
            },
            "opportunity_challenge": {
                "evidence_needed": True,
                "logic_defense": True,
                "counter_evidence": True,
                "acknowledgment": False
            },
            "logic_challenge": {
                "evidence_needed": False,
                "logic_defense": True,
                "counter_evidence": False,
                "acknowledgment": True
            },
            "general_disagreement": {
                "evidence_needed": True,
                "logic_defense": True,
                "counter_evidence": False,
                "acknowledgment": True
            }
        }
        
        return response_requirements.get(challenge_type, response_requirements["general_disagreement"])
    
    def _determine_response_strategy(self, challenge_analysis: Dict, original_position: Dict) -> str:
        """Determine strategy for responding to challenge"""
        
        challenge_strength = challenge_analysis["strength_assessment"]
        challenge_type = challenge_analysis["challenge_type"]
        
        # Strategic response based on agent perspective and challenge strength
        if challenge_strength > 0.7:  # Strong challenge
            if self.perspective == DebatePerspective.CONSERVATIVE:
                return "defensive_with_evidence"
            elif self.perspective == DebatePerspective.AGGRESSIVE:
                return "counter_attack_with_opportunity_cost"
            else:
                return "acknowledge_and_refine"
        else:  # Weak challenge
            return "confident_rebuttal"
    
    async def _formulate_counter_response(self, challenge_analysis: Dict, original_position: Dict, strategy: str) -> Dict:
        """Formulate counter-response based on strategy"""
        
        if strategy == "defensive_with_evidence":
            return await self._create_defensive_response(challenge_analysis, original_position)
        elif strategy == "counter_attack_with_opportunity_cost":
            return await self._create_aggressive_response(challenge_analysis, original_position)
        elif strategy == "acknowledge_and_refine":
            return await self._create_balanced_response(challenge_analysis, original_position)
        else:  # confident_rebuttal
            return await self._create_confident_response(challenge_analysis, original_position)
    
    async def _create_defensive_response(self, challenge_analysis: Dict, position: Dict) -> Dict:
        """Create defensive response with strong evidence"""
        
        arguments = [
            f"The evidence strongly supports my position because {position['arguments'][0] if position['arguments'] else 'analysis is comprehensive'}",
            "Risk management principles require this cautious approach",
            "Historical precedents validate this conservative stance"
        ]
        
        return {
            "arguments": arguments,
            "evidence": position.get("supporting_evidence", [])[:2],
            "strength_score": 0.7
        }
    
    async def _create_aggressive_response(self, challenge_analysis: Dict, position: Dict) -> Dict:
        """Create aggressive counter-response focusing on opportunity cost"""
        
        arguments = [
            f"The opportunity cost of inaction far exceeds the risks mentioned",
            f"Market conditions strongly favor the {position.get('stance', 'recommended approach')}",
            "Conservative approaches often miss significant value creation opportunities"
        ]
        
        return {
            "arguments": arguments,
            "evidence": position.get("supporting_evidence", [])[:2],
            "strength_score": 0.8
        }
    
    async def _create_balanced_response(self, challenge_analysis: Dict, position: Dict) -> Dict:
        """Create balanced response that acknowledges valid points"""
        
        arguments = [
            "While the challenge raises valid concerns, the overall analysis supports my position",
            f"The {position.get('stance', 'recommended approach')} balances these competing factors",
            "Multiple perspectives strengthen the final decision"
        ]
        
        return {
            "arguments": arguments,
            "evidence": position.get("supporting_evidence", [])[:2],
            "strength_score": 0.6
        }
    
    async def _create_confident_response(self, challenge_analysis: Dict, position: Dict) -> Dict:
        """Create confident rebuttal to weak challenges"""
        
        arguments = [
            f"My analysis comprehensively addresses the concerns raised",
            f"The {position.get('stance', 'position')} is well-supported by multiple evidence sources",
            "The challenge does not materially change the fundamental analysis"
        ]
        
        return {
            "arguments": arguments,
            "evidence": position.get("supporting_evidence", [])[:3],
            "strength_score": 0.9
        }
    
    def _identify_valid_points(self, challenge_analysis: Dict) -> List[str]:
        """Identify valid points in the challenge that should be acknowledged"""
        
        acknowledgments = []
        challenge_points = challenge_analysis.get("key_points", [])
        
        # Look for points that align with agent's perspective
        for point in challenge_points:
            if self._point_has_merit(point):
                acknowledgments.append(f"Valid concern: {point}")
        
        return acknowledgments[:2]  # Limit acknowledgments
    
    def _point_has_merit(self, point: str) -> bool:
        """Determine if a challenge point has merit"""
        
        point_lower = point.lower()
        
        # Conservative agents acknowledge risk-related points
        if self.perspective == DebatePerspective.CONSERVATIVE:
            return any(word in point_lower for word in ["risk", "caution", "uncertainty"])
        
        # Aggressive agents acknowledge opportunity-related points
        elif self.perspective == DebatePerspective.AGGRESSIVE:
            return any(word in point_lower for word in ["opportunity", "growth", "timing"])
        
        # Balanced agents acknowledge most reasonable points
        elif self.perspective == DebatePerspective.BALANCED:
            return any(word in point_lower for word in ["consider", "factor", "important"])
        
        # Specialists acknowledge technical points
        else:
            return any(word in point_lower for word in ["technical", "analysis", "data"])
    
    async def _update_position_from_challenge(self, original_position: Dict, challenge_analysis: Dict, acknowledgments: List[str]) -> Dict:
        """Update position based on valid challenges and acknowledgments"""
        
        updated_position = original_position.copy()
        
        # Adjust confidence based on challenge strength
        challenge_strength = challenge_analysis["strength_assessment"]
        confidence_adjustment = 0.0
        
        if challenge_strength > 0.7 and acknowledgments:
            confidence_adjustment = -0.1  # Reduce confidence for strong, valid challenges
        elif challenge_strength < 0.4:
            confidence_adjustment = 0.05  # Increase confidence for weak challenges
        
        updated_position["confidence"] = min(max(
            original_position["confidence_score"] + confidence_adjustment, 0.1), 0.95)
        
        # Add refinements if acknowledging valid points
        if acknowledgments:
            refinements = [f"Refined considering: {ack}" for ack in acknowledgments]
            updated_position["refinements"] = refinements
        
        return updated_position
    
    def update_performance_metrics(self, debate_result: Dict):
        """Update agent's performance metrics based on debate outcome"""
        
        if debate_result.get("position_in_consensus"):
            self.performance_metrics["consensus_contributions"] += 1
        
        if debate_result.get("arguments_accepted", 0) > 0:
            self.performance_metrics["arguments_accepted"] += debate_result["arguments_accepted"]
        
        if debate_result.get("position_defended"):
            self.performance_metrics["positions_defended"] += 1
        
        # Update accuracy score (simplified)
        if debate_result.get("outcome_accuracy"):
            current_accuracy = self.performance_metrics["accuracy_score"]
            new_accuracy = debate_result["outcome_accuracy"]
            self.performance_metrics["accuracy_score"] = (current_accuracy * 0.8) + (new_accuracy * 0.2)
    
    def get_debate_summary(self) -> Dict:
        """Get summary of agent's debate capabilities and performance"""
        
        return {
            "agent_id": self.agent_id,
            "perspective": self.perspective.value,
            "debate_style": self.debate_style.value,
            "specialization": self.specialization,
            "debate_strengths": self.debate_strengths,
            "performance_metrics": self.performance_metrics,
            "debate_config": self.debate_config
        }
    
    # Abstract methods to be implemented by specific agents
    
    @abstractmethod
    async def execute_specialized_analysis(self, query: str, context: Dict) -> Dict:
        """Execute agent's specialized analysis (non-debate)"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict:
        """Perform health check for MCP monitoring"""
        pass


# Example Implementation for QuantitativeAnalystAgent
class QuantitativeAnalystAgent(BaseAgent):
    """Conservative, risk-focused quantitative analyst with debate capabilities"""
    
    def __init__(self):
        super().__init__("quantitative_analyst", DebatePerspective.CONSERVATIVE)
    
    def _get_specialization(self) -> str:
        return "risk_analysis_and_portfolio_optimization"
    
    def _get_debate_strengths(self) -> List[str]:
        return ["statistical_evidence", "risk_assessment", "downside_scenarios", "historical_analysis"]
    
    def _get_specialized_themes(self) -> Dict[str, List[str]]:
        return {
            "volatility": ["volatility", "variance", "deviation", "fluctuation"],
            "correlation": ["correlation", "relationship", "dependency", "connection"],
            "statistics": ["probability", "confidence", "significance", "statistical"],
            "risk_metrics": ["var", "sharpe", "beta", "alpha", "tracking_error"]
        }
    
    async def _gather_specialized_evidence(self, analysis: Dict, context: Dict) -> List[Dict]:
        """Gather quantitative evidence using statistical analysis"""
        
        evidence = []
        
        # Portfolio risk analysis
        if "risk" in analysis.get("relevant_themes", []):
            evidence.append({
                "type": "statistical",
                "analysis": "Value at Risk analysis shows 95% confidence level risk",
                "data": "Portfolio VaR: 2.3% daily loss potential",
                "confidence": 0.85,
                "source": "Monte Carlo simulation"
            })
        
        # Correlation analysis
        if "allocation" in analysis.get("relevant_themes", []):
            evidence.append({
                "type": "analytical",
                "analysis": "Correlation matrix reveals portfolio concentration risks",
                "data": "Average correlation: 0.67 (above diversification threshold)",
                "confidence": 0.9,
                "source": "Historical correlation analysis"
            })
        
        # Performance attribution
        evidence.append({
            "type": "historical",
            "analysis": "Historical performance attribution analysis",
            "data": "Risk-adjusted returns: Sharpe ratio 1.23",
            "confidence": 0.8,
            "source": "3-year performance history"
        })
        
        return evidence
    
    async def _generate_stance(self, analysis: Dict, evidence: List[Dict]) -> str:
        """Generate conservative, risk-focused stance"""
        
        themes = analysis.get("relevant_themes", [])
        
        if "risk" in themes:
            return "recommend risk reduction through diversification and defensive positioning"
        elif "allocation" in themes:
            return "suggest conservative rebalancing to reduce portfolio concentration"
        elif "timing" in themes:
            return "advise caution and gradual position adjustments"
        else:
            return "recommend thorough risk assessment before any major changes"
    
    async def _identify_general_risks(self, context: Dict) -> List[str]:
        """Identify general portfolio risks"""
        return [
            "Market volatility risk",
            "Concentration risk",
            "Liquidity risk",
            "Interest rate risk",
            "Inflation risk"
        ]
    
    async def _identify_specialized_risks(self, analysis: Dict, context: Dict) -> List[str]:
        """Identify quantitative analysis specific risks"""
        return [
            "Model risk in quantitative analysis",
            "Historical data limitations",
            "Correlation breakdown during stress periods"
        ]
    
    async def execute_specialized_analysis(self, query: str, context: Dict) -> Dict:
        """Execute quantitative analysis"""
        # Implementation would include VaR, correlation analysis, etc.
        return {"analysis": "quantitative_analysis_results"}
    
    async def health_check(self) -> Dict:
        """Health check for quantitative analyst"""
        return {
            "status": "healthy",
            "response_time": 0.5,
            "memory_usage": "normal",
            "active_jobs": 0,
            "capabilities": self.debate_strengths
        }
            # Aggressive agents focus on opportunity cost risks
            return [r for r in risks if "opportunity" in r.lower() or "timing" in r.lower()]
        else:
            # Balanced/Specialist agents consider systematic risks
            return [r for r in risks if any(word in r.lower() for word in ["systematic", "market", "economic"])]
    
    def _suggest_risk_mitigation(self, risks: List[str]) -> List[str]:
        """Suggest risk mitigation strategies"""
        
        mitigation_strategies = []
        
        for risk in risks[:3]:  # Top 3 risks
            if "market" in risk.lower():
                mitigation_strategies.append("Diversification across asset classes")
            elif "timing" in risk.lower():
                mitigation_strategies.append("Dollar-cost averaging approach")
            elif "volatility" in risk.lower():
                mitigation_strategies.append("Options hedging strategies")
            else:
                mitigation_strategies.append(f"Monitor and adjust for {risk.lower()}")
        
        return mitigation_strategies
    
    # Additional helper methods for debate functionality
    
    def _calculate_position_confidence(self, position: Dict, evidence: List[Dict]) -> float:
        """Calculate confidence in the formulated position"""
        
        # Base confidence from evidence quality
        evidence_confidence = sum(ev.get("confidence", 0.5) for ev in evidence) / len(evidence) if evidence else 0.5
        
        # Adjust for agent's confidence threshold
        threshold_adjustment = min(evidence_confidence / self.debate_config["confidence_threshold"], 1.0)
        
        # Adjust for position strength
        argument_strength = len(position.get("arguments", [])) / 5.0  # Normalize to 5 arguments
        
        # Combine factors
        final_confidence = (evidence_confidence * 0.6 + threshold_adjustment * 0.3 + argument_strength * 0.1)
        
        return min(max(final_confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    async def _prepare_counter_arguments(self, position: Dict, debate_context: Dict) -> Dict:
        """Prepare responses to expected counter-arguments"""
        
        expected_challenges = self._predict_likely_challenges(position, debate_context)
        prepared_responses = []
        
        for challenge in expected_challenges:
            response = await self._prepare_response_to_challenge(challenge, position)
            prepared_responses.append({
                "challenge": challenge,
                "response": response
            })
        
        return {
            "expected_challenges": expected_challenges,
            "prepared_responses": prepared_responses
        }
    
    def _predict_likely_challenges(self, position: Dict, debate_context: Dict) -> List[str]:
        """Predict likely challenges from other agents"""
        
        other_agents = debate_context.get("other_participants", [])
        challenges = []
        
        # Predict challenges based on opposing perspectives
        if self.perspective == DebatePerspective.CONSERVATIVE:
            challenges.extend([
                "Overly cautious approach may miss opportunities",
                "Risk assessment may be too pessimistic",
                "Historical data may not predict future performance"
            ])
        elif self