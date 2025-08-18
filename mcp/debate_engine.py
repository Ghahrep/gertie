# mcp/debate_engine.py
"""
Revolutionary Multi-Agent Debate Engine
========================================
The world's first autonomous AI investment debate system where specialized
agents collaborate through structured debates to reach optimal decisions.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import json
import uuid
from enum import Enum

class DebateStage(Enum):
    INITIALIZATION = "initialization"
    OPENING_POSITIONS = "opening_positions"
    CROSS_EXAMINATION = "cross_examination"
    CONSENSUS_BUILDING = "consensus_building"
    FINAL_SYNTHESIS = "final_synthesis"
    COMPLETED = "completed"

class AgentPerspective(Enum):
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    SPECIALIST = "specialist"

@dataclass
class DebatePosition:
    """Represents an agent's position in the debate"""
    agent_id: str
    stance: str
    key_arguments: List[str]
    supporting_evidence: List[Dict]
    confidence_score: float
    risk_assessment: Dict
    counterarguments_addressed: List[str]
    timestamp: datetime

@dataclass
class DebateRound:
    """Represents a single round of debate"""
    round_number: int
    stage: DebateStage
    positions: List[DebatePosition]
    challenges: List[Dict]
    responses: List[Dict]
    round_summary: str
    duration: float

class DebateEngine:
    """Revolutionary multi-agent debate orchestrator"""
    
    def __init__(self, mcp_server):
        self.mcp = mcp_server
        self.active_debates = {}
        self.debate_history = {}
        
        # Debate configuration
        self.max_rounds = 3
        self.round_timeout = 300  # 5 minutes per round
        self.min_participants = 2
        self.max_participants = 5
        
    async def initiate_debate(
        self, 
        query: str, 
        agents: List[str], 
        portfolio_context: Dict,
        debate_params: Optional[Dict] = None
    ) -> str:
        """Start structured debate between specified agents"""
        
        # Generate unique debate ID
        debate_id = str(uuid.uuid4())
        
        # Analyze query requirements
        query_analysis = await self._analyze_query_requirements(query, portfolio_context)
        
        # Assign agent perspectives for balanced debate
        agent_assignments = await self._assign_agent_perspectives(agents, query_analysis)
        
        # Initialize debate state
        debate_state = {
            "debate_id": debate_id,
            "query": query,
            "portfolio_context": portfolio_context,
            "participants": agent_assignments,
            "current_stage": DebateStage.INITIALIZATION,
            "rounds": [],
            "consensus": None,
            "confidence_score": 0.0,
            "started_at": datetime.now(),
            "status": "active"
        }
        
        self.active_debates[debate_id] = debate_state
        
        # Start the debate process
        asyncio.create_task(self._execute_debate(debate_id))
        
        return debate_id
    
    async def _analyze_query_requirements(self, query: str, context: Dict) -> Dict:
        """Analyze query to determine debate focus and complexity"""
        
        # Extract key themes from query
        themes = self._extract_query_themes(query)
        
        # Assess complexity level
        complexity = self._assess_query_complexity(query, context)
        
        # Determine required expertise areas
        expertise_needed = self._identify_required_expertise(query, themes)
        
        return {
            "primary_focus": themes.get("primary", "general_analysis"),
            "complexity_level": complexity,
            "time_sensitivity": self._assess_time_sensitivity(query),
            "required_expertise": expertise_needed,
            "debate_style": self._determine_debate_style(themes, complexity),
            "expected_disagreement_areas": self._predict_disagreement_areas(themes)
        }
    
    def _extract_query_themes(self, query: str) -> Dict:
        """Extract key themes and focus areas from user query"""
        query_lower = query.lower()
        
        themes = {
            "risk": any(word in query_lower for word in ["risk", "safe", "conservative", "protect", "downside"]),
            "growth": any(word in query_lower for word in ["growth", "aggressive", "opportunity", "upside"]),
            "tax": any(word in query_lower for word in ["tax", "harvest", "deduction", "after-tax"]),
            "options": any(word in query_lower for word in ["option", "hedge", "volatility", "strategy"]),
            "timing": any(word in query_lower for word in ["when", "timing", "buy", "sell", "market"]),
            "allocation": any(word in query_lower for word in ["allocate", "balance", "diversify", "weight"]),
            "economic": any(word in query_lower for word in ["economy", "fed", "inflation", "recession"])
        }
        
        # Determine primary focus
        primary = max(themes.items(), key=lambda x: x[1])
        themes["primary"] = primary[0] if primary[1] else "general_analysis"
        
        return themes
    
    async def _assign_agent_perspectives(self, agents: List[str], analysis: Dict) -> Dict:
        """Assign debate perspectives to create productive disagreement"""
        
        agent_assignments = {}
        
        # Base perspective assignments
        perspective_map = {
            "quantitative_analyst": AgentPerspective.CONSERVATIVE,
            "market_intelligence": AgentPerspective.AGGRESSIVE,
            "tax_strategist": AgentPerspective.SPECIALIST,
            "options_analyst": AgentPerspective.SPECIALIST,
            "economic_data": AgentPerspective.BALANCED
        }
        
        for agent_id in agents:
            base_perspective = perspective_map.get(agent_id, AgentPerspective.BALANCED)
            
            agent_assignments[agent_id] = {
                "perspective": base_perspective,
                "role": self._determine_debate_role(agent_id, analysis),
                "focus_areas": self._assign_focus_areas(agent_id, analysis),
                "expected_stance": self._predict_agent_stance(agent_id, analysis)
            }
        
        return agent_assignments
    
    def _determine_debate_role(self, agent_id: str, analysis: Dict) -> str:
        """Determine specific role for agent in this debate"""
        
        role_mapping = {
            "quantitative_analyst": "risk_assessor",
            "market_intelligence": "opportunity_identifier", 
            "tax_strategist": "efficiency_optimizer",
            "options_analyst": "strategy_specialist",
            "economic_data": "macro_advisor"
        }
        
        return role_mapping.get(agent_id, "general_analyst")
    
    async def _execute_debate(self, debate_id: str):
        """Execute the complete debate workflow"""
        
        debate_state = self.active_debates[debate_id]
        
        try:
            # Stage 1: Opening Positions
            await self._conduct_opening_positions(debate_id)
            
            # Stage 2: Cross-Examination (multiple rounds)
            for round_num in range(1, self.max_rounds):
                await self._conduct_cross_examination_round(debate_id, round_num)
            
            # Stage 3: Consensus Building
            await self._conduct_consensus_building(debate_id)
            
            # Stage 4: Final Synthesis
            await self._conduct_final_synthesis(debate_id)
            
            # Mark debate as completed
            debate_state["status"] = "completed"
            debate_state["completed_at"] = datetime.now()
            
        except Exception as e:
            await self._handle_debate_error(debate_id, e)
    
    async def _conduct_opening_positions(self, debate_id: str):
        """Stage 1: Collect initial positions from all agents"""
        
        debate_state = self.active_debates[debate_id]
        debate_state["current_stage"] = DebateStage.OPENING_POSITIONS
        
        positions = []
        
        # Collect positions from each agent
        for agent_id, assignment in debate_state["participants"].items():
            
            position_prompt = self._generate_opening_position_prompt(
                debate_state["query"],
                debate_state["portfolio_context"],
                assignment
            )
            
            # Get agent's initial position
            agent_response = await self.mcp.execute_agent_analysis(
                agent_id, 
                position_prompt,
                debate_state["portfolio_context"]
            )
            
            position = DebatePosition(
                agent_id=agent_id,
                stance=agent_response.get("stance", "neutral"),
                key_arguments=agent_response.get("key_arguments", []),
                supporting_evidence=agent_response.get("evidence", []),
                confidence_score=agent_response.get("confidence", 0.5),
                risk_assessment=agent_response.get("risk_assessment", {}),
                counterarguments_addressed=[],
                timestamp=datetime.now()
            )
            
            positions.append(position)
        
        # Create opening round
        opening_round = DebateRound(
            round_number=0,
            stage=DebateStage.OPENING_POSITIONS,
            positions=positions,
            challenges=[],
            responses=[],
            round_summary=self._summarize_opening_positions(positions),
            duration=0.0
        )
        
        debate_state["rounds"].append(opening_round)
    
    def _generate_opening_position_prompt(self, query: str, context: Dict, assignment: Dict) -> str:
        """Generate specialized prompt for agent's opening position"""
        
        perspective = assignment["perspective"].value
        role = assignment["role"]
        focus_areas = assignment["focus_areas"]
        
        prompt = f"""
        DEBATE PARTICIPATION REQUEST
        ===========================
        
        Query: {query}
        
        Your Role: {role} with {perspective} perspective
        Focus Areas: {', '.join(focus_areas)}
        
        Instructions:
        1. Analyze the query from your specialized {perspective} perspective
        2. Formulate a clear, defensible position
        3. Provide specific evidence supporting your stance
        4. Identify potential risks or opportunities others might miss
        5. Anticipate counterarguments from opposing perspectives
        
        Required Response Format:
        {{
            "stance": "your_clear_position",
            "key_arguments": ["argument_1", "argument_2", "argument_3"],
            "evidence": [
                {{"type": "statistical", "data": "specific_data", "source": "source"}},
                {{"type": "analytical", "analysis": "your_analysis", "confidence": 0.8}}
            ],
            "risk_assessment": {{
                "primary_risks": ["risk_1", "risk_2"],
                "mitigation_strategies": ["strategy_1", "strategy_2"]
            }},
            "confidence": 0.85,
            "expected_counterarguments": ["counter_1", "counter_2"]
        }}
        
        Remember: Take a strong, evidence-based position that reflects your {perspective} perspective.
        This is a collaborative debate - disagreement leads to better decisions.
        """
        
        return prompt
    
    async def _conduct_cross_examination_round(self, debate_id: str, round_num: int):
        """Conduct cross-examination between agents"""
        
        debate_state = self.active_debates[debate_id]
        debate_state["current_stage"] = DebateStage.CROSS_EXAMINATION
        
        previous_round = debate_state["rounds"][-1]
        challenges = []
        responses = []
        
        # Generate challenges between opposing positions
        for i, position_a in enumerate(previous_round.positions):
            for j, position_b in enumerate(previous_round.positions):
                if i != j:  # Don't challenge yourself
                    
                    challenge = await self._generate_challenge(
                        position_a, position_b, debate_state["query"]
                    )
                    
                    if challenge:
                        challenges.append({
                            "challenger": position_a.agent_id,
                            "challenged": position_b.agent_id,
                            "challenge": challenge,
                            "round": round_num
                        })
        
        # Collect responses to challenges
        for challenge in challenges:
            response = await self._get_challenge_response(
                challenge, debate_state
            )
            responses.append(response)
        
        # Update agent positions based on challenges
        updated_positions = await self._update_positions_after_challenges(
            previous_round.positions, challenges, responses
        )
        
        # Create round record
        round_record = DebateRound(
            round_number=round_num,
            stage=DebateStage.CROSS_EXAMINATION,
            positions=updated_positions,
            challenges=challenges,
            responses=responses,
            round_summary=self._summarize_challenge_round(challenges, responses),
            duration=0.0
        )
        
        debate_state["rounds"].append(round_record)
    
    async def _generate_challenge(self, challenger_pos: DebatePosition, target_pos: DebatePosition, query: str) -> Optional[str]:
        """Generate intelligent challenge between opposing positions"""
        
        # Find points of disagreement
        disagreements = self._identify_disagreements(challenger_pos, target_pos)
        
        if not disagreements:
            return None
        
        # Select strongest disagreement to challenge
        primary_disagreement = disagreements[0]
        
        challenge = f"""
        Challenge to {target_pos.agent_id}'s position:
        
        You argue that {target_pos.stance}, but I disagree on the following point:
        {primary_disagreement}
        
        My evidence suggests: {challenger_pos.supporting_evidence[0] if challenger_pos.supporting_evidence else 'alternative analysis'}
        
        How do you reconcile this with your position? What evidence supports your stance over mine?
        """
        
        return challenge
    
    async def _conduct_consensus_building(self, debate_id: str):
        """Build consensus from all agent positions"""
        
        debate_state = self.active_debates[debate_id]
        debate_state["current_stage"] = DebateStage.CONSENSUS_BUILDING
        
        # Get final positions from last round
        final_positions = debate_state["rounds"][-1].positions
        
        # Build weighted consensus
        consensus = await self._build_weighted_consensus(final_positions, debate_state["query"])
        
        debate_state["consensus"] = consensus
        debate_state["confidence_score"] = consensus["overall_confidence"]
    
    async def _build_weighted_consensus(self, positions: List[DebatePosition], query: str) -> Dict:
        """Build intelligent consensus from agent positions"""
        
        # Weight factors
        total_confidence = sum(pos.confidence_score for pos in positions)
        evidence_quality_scores = [self._assess_evidence_quality(pos.supporting_evidence) for pos in positions]
        
        # Build consensus recommendation
        consensus_arguments = []
        minority_opinions = []
        
        # Group similar positions
        position_groups = self._group_similar_positions(positions)
        
        for group in position_groups:
            group_weight = sum(pos.confidence_score for pos in group) / total_confidence
            
            if group_weight > 0.4:  # Majority opinion
                consensus_arguments.extend(group[0].key_arguments)
            else:  # Minority opinion - preserve
                minority_opinions.append({
                    "agents": [pos.agent_id for pos in group],
                    "position": group[0].stance,
                    "arguments": group[0].key_arguments,
                    "weight": group_weight
                })
        
        # Calculate overall confidence
        overall_confidence = self._calculate_consensus_confidence(positions, position_groups)
        
        consensus = {
            "recommendation": self._synthesize_final_recommendation(consensus_arguments),
            "confidence_level": overall_confidence,
            "supporting_arguments": consensus_arguments,
            "minority_opinions": minority_opinions,
            "risk_factors": self._extract_consensus_risks(positions),
            "implementation_priority": self._determine_implementation_priority(overall_confidence),
            "evidence_strength": sum(evidence_quality_scores) / len(evidence_quality_scores)
        }
        
        return consensus
    
    def _group_similar_positions(self, positions: List[DebatePosition]) -> List[List[DebatePosition]]:
        """Group agents with similar positions"""
        
        groups = []
        unprocessed = positions.copy()
        
        while unprocessed:
            current_pos = unprocessed.pop(0)
            similar_group = [current_pos]
            
            # Find similar positions
            to_remove = []
            for other_pos in unprocessed:
                if self._positions_similar(current_pos, other_pos):
                    similar_group.append(other_pos)
                    to_remove.append(other_pos)
            
            # Remove grouped positions
            for pos in to_remove:
                unprocessed.remove(pos)
            
            groups.append(similar_group)
        
        return groups
    
    def _positions_similar(self, pos1: DebatePosition, pos2: DebatePosition) -> bool:
        """Determine if two positions are similar enough to group"""
        
        # Simple similarity check - can be enhanced with NLP
        stance_similarity = pos1.stance.lower() in pos2.stance.lower() or pos2.stance.lower() in pos1.stance.lower()
        
        # Check argument overlap
        arg_overlap = len(set(pos1.key_arguments) & set(pos2.key_arguments)) > 0
        
        return stance_similarity or arg_overlap
    
    def _calculate_consensus_confidence(self, positions: List[DebatePosition], groups: List[List[DebatePosition]]) -> float:
        """Calculate overall confidence in consensus"""
        
        # Factors affecting confidence
        agent_agreement = len(groups) <= 2  # Fewer groups = more agreement
        average_confidence = sum(pos.confidence_score for pos in positions) / len(positions)
        evidence_diversity = len(set(ev.get("type", "unknown") for pos in positions for ev in pos.supporting_evidence))
        
        # Weighted confidence calculation
        base_confidence = average_confidence
        
        if agent_agreement:
            base_confidence *= 1.2  # Boost for agreement
        
        if evidence_diversity >= 3:
            base_confidence *= 1.1  # Boost for diverse evidence
        
        return min(base_confidence, 1.0)  # Cap at 1.0
    
    def _synthesize_final_recommendation(self, arguments: List[str]) -> str:
        """Create final recommendation from consensus arguments"""
        
        if not arguments:
            return "No clear consensus reached. Consider seeking additional analysis."
        
        # Simple synthesis - can be enhanced with NLP
        recommendation = f"Based on multi-agent analysis: {arguments[0]}"
        
        if len(arguments) > 1:
            recommendation += f" Additionally, consider: {'; '.join(arguments[1:3])}"
        
        return recommendation
    
    async def get_debate_status(self, debate_id: str) -> Dict:
        """Get current status of a debate"""
        
        if debate_id not in self.active_debates:
            return {"error": "Debate not found"}
        
        debate_state = self.active_debates[debate_id]
        
        return {
            "debate_id": debate_id,
            "status": debate_state["status"],
            "current_stage": debate_state["current_stage"].value,
            "rounds_completed": len(debate_state["rounds"]),
            "participants": list(debate_state["participants"].keys()),
            "consensus_ready": debate_state.get("consensus") is not None,
            "confidence_score": debate_state.get("confidence_score", 0.0)
        }
    
    async def get_debate_results(self, debate_id: str) -> Dict:
        """Get complete debate results"""
        
        if debate_id not in self.active_debates:
            return {"error": "Debate not found"}
        
        debate_state = self.active_debates[debate_id]
        
        if debate_state["status"] != "completed":
            return {"error": "Debate not yet completed"}
        
        return {
            "debate_id": debate_id,
            "query": debate_state["query"],
            "participants": debate_state["participants"],
            "rounds": [self._serialize_round(round_data) for round_data in debate_state["rounds"]],
            "final_consensus": debate_state["consensus"],
            "confidence_score": debate_state["confidence_score"],
            "duration": (debate_state["completed_at"] - debate_state["started_at"]).total_seconds(),
            "debate_summary": self._generate_debate_summary(debate_state)
        }
    
    def _serialize_round(self, round_data: DebateRound) -> Dict:
        """Convert DebateRound to serializable format"""
        return {
            "round_number": round_data.round_number,
            "stage": round_data.stage.value,
            "positions": [self._serialize_position(pos) for pos in round_data.positions],
            "challenges": round_data.challenges,
            "responses": round_data.responses,
            "summary": round_data.round_summary
        }
    
    def _serialize_position(self, position: DebatePosition) -> Dict:
        """Convert DebatePosition to serializable format"""
        return {
            "agent_id": position.agent_id,
            "stance": position.stance,
            "key_arguments": position.key_arguments,
            "supporting_evidence": position.supporting_evidence,
            "confidence_score": position.confidence_score,
            "risk_assessment": position.risk_assessment
        }
    
    # Helper methods (simplified implementations)
    
    def _assess_query_complexity(self, query: str, context: Dict) -> str:
        """Assess complexity level of the query"""
        word_count = len(query.split())
        portfolio_size = len(context.get("holdings", []))
        
        if word_count > 20 or portfolio_size > 20:
            return "high"
        elif word_count > 10 or portfolio_size > 10:
            return "medium"
        else:
            return "low"
    
    def _assess_time_sensitivity(self, query: str) -> str:
        """Assess time sensitivity of the query"""
        urgent_words = ["urgent", "immediate", "now", "today", "asap"]
        if any(word in query.lower() for word in urgent_words):
            return "high"
        return "medium"
    
    def _identify_required_expertise(self, query: str, themes: Dict) -> List[str]:
        """Identify required expertise areas"""
        expertise = []
        if themes["risk"]: expertise.append("risk_analysis")
        if themes["tax"]: expertise.append("tax_optimization")
        if themes["options"]: expertise.append("options_analysis")
        if themes["economic"]: expertise.append("economic_analysis")
        return expertise or ["general_analysis"]
    
    def _determine_debate_style(self, themes: Dict, complexity: str) -> str:
        """Determine appropriate debate style"""
        if complexity == "high":
            return "structured_formal"
        elif any(themes.values()):
            return "focused_collaborative"
        else:
            return "open_discussion"
    
    def _predict_disagreement_areas(self, themes: Dict) -> List[str]:
        """Predict areas where agents might disagree"""
        disagreements = []
        if themes["risk"] and themes["growth"]:
            disagreements.append("risk_vs_return_tradeoff")
        if themes["tax"]:
            disagreements.append("tax_efficiency_vs_simplicity")
        return disagreements
    
    def _assign_focus_areas(self, agent_id: str, analysis: Dict) -> List[str]:
        """Assign focus areas for agent in this debate"""
        agent_specializations = {
            "quantitative_analyst": ["risk_assessment", "statistical_analysis"],
            "market_intelligence": ["market_timing", "opportunity_identification"],
            "tax_strategist": ["tax_efficiency", "after_tax_returns"],
            "options_analyst": ["volatility_analysis", "hedging_strategies"],
            "economic_data": ["macro_trends", "economic_indicators"]
        }
        return agent_specializations.get(agent_id, ["general_analysis"])
    
    def _predict_agent_stance(self, agent_id: str, analysis: Dict) -> str:
        """Predict likely stance for agent"""
        if "risk" in analysis.get("required_expertise", []):
            if agent_id == "quantitative_analyst":
                return "risk_focused"
            elif agent_id == "market_intelligence":
                return "opportunity_focused"
        return "balanced"
    
    def _summarize_opening_positions(self, positions: List[DebatePosition]) -> str:
        """Summarize opening positions"""
        return f"Opening positions established by {len(positions)} agents with varying perspectives on the investment question."
    
    def _summarize_challenge_round(self, challenges: List[Dict], responses: List[Dict]) -> str:
        """Summarize challenge round"""
        return f"Cross-examination completed with {len(challenges)} challenges and {len(responses)} responses."
    
    def _identify_disagreements(self, pos1: DebatePosition, pos2: DebatePosition) -> List[str]:
        """Identify key disagreements between positions"""
        # Simplified - can be enhanced with NLP
        if pos1.stance != pos2.stance:
            return [f"Fundamental disagreement on approach: {pos1.stance} vs {pos2.stance}"]
        return []
    
    def _assess_evidence_quality(self, evidence: List[Dict]) -> float:
        """Assess quality of evidence provided"""
        if not evidence:
            return 0.0
        
        quality_score = 0.0
        for item in evidence:
            if item.get("type") == "statistical":
                quality_score += 0.8
            elif item.get("type") == "analytical":
                quality_score += 0.6
            else:
                quality_score += 0.4
        
        return min(quality_score / len(evidence), 1.0)
    
    def _extract_consensus_risks(self, positions: List[DebatePosition]) -> List[str]:
        """Extract key risks from all positions"""
        all_risks = []
        for pos in positions:
            all_risks.extend(pos.risk_assessment.get("primary_risks", []))
        return list(set(all_risks))  # Remove duplicates
    
    def _determine_implementation_priority(self, confidence: float) -> str:
        """Determine implementation priority based on confidence"""
        if confidence > 0.8:
            return "high"
        elif confidence > 0.6:
            return "medium"
        else:
            return "low"
    
    def _generate_debate_summary(self, debate_state: Dict) -> str:
        """Generate comprehensive debate summary"""
        participants = len(debate_state["participants"])
        rounds = len(debate_state["rounds"])
        confidence = debate_state.get("confidence_score", 0.0)
        
        return f"Multi-agent debate completed with {participants} specialized agents over {rounds} rounds. Final consensus confidence: {confidence:.1%}"
    
    async def _get_challenge_response(self, challenge: Dict, debate_state: Dict) -> Dict:
        """Get agent response to challenge"""
        # Simplified implementation
        return {
            "responder": challenge["challenged"],
            "response": "Counter-response provided with additional evidence",
            "updated_position": "Position refined based on challenge"
        }
    
    async def _update_positions_after_challenges(self, positions: List[DebatePosition], challenges: List[Dict], responses: List[Dict]) -> List[DebatePosition]:
        """Update agent positions based on challenges and responses"""
        # For now, return original positions - can be enhanced to actually update
        return positions
    
    async def _conduct_final_synthesis(self, debate_id: str):
        """Conduct final synthesis of debate results"""
        debate_state = self.active_debates[debate_id]
        debate_state["current_stage"] = DebateStage.FINAL_SYNTHESIS
        # Implementation would synthesize all rounds into final recommendation
    
    async def _handle_debate_error(self, debate_id: str, error: Exception):
        """Handle errors during debate execution"""
        debate_state = self.active_debates[debate_id]
        debate_state["status"] = "error"
        debate_state["error"] = str(error)
        print(f"Debate {debate_id} failed: {error}")