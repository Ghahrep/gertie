# mcp/consensus_builder.py
"""
Intelligent Consensus Building System
====================================
Advanced algorithms for building consensus from multi-agent debates,
preserving minority opinions and calculating confidence scores.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from collections import defaultdict
import json

@dataclass
class ConsensusMetrics:
    """Metrics for evaluating consensus quality"""
    agreement_level: float  # 0-1, how much agents agree
    evidence_strength: float  # 0-1, quality of supporting evidence
    confidence_distribution: Dict[str, float]  # agent_id -> confidence
    minority_strength: float  # 0-1, strength of minority positions
    overall_confidence: float  # 0-1, final consensus confidence

@dataclass
class MinorityOpinion:
    """Represents a preserved minority opinion"""
    agents: List[str]
    position: str
    arguments: List[str]
    evidence: List[Dict]
    weight: float
    risk_if_ignored: float
    importance_score: float

class ConsensusBuilder:
    """Intelligent consensus building from agent debates"""
    
    def __init__(self):
        # Weighting factors for consensus building
        self.confidence_weight = 0.4
        self.evidence_weight = 0.3
        self.expertise_weight = 0.2
        self.consistency_weight = 0.1
        
        # Agent expertise scores (can be learned over time)
        self.agent_expertise = {
            "quantitative_analyst": {"risk_analysis": 0.9, "statistical_modeling": 0.95, "portfolio_optimization": 0.9},
            "market_intelligence": {"market_timing": 0.85, "trend_analysis": 0.8, "sentiment_analysis": 0.75},
            "tax_strategist": {"tax_optimization": 0.95, "regulatory_compliance": 0.9, "after_tax_analysis": 0.85},
            "options_analyst": {"volatility_analysis": 0.9, "options_pricing": 0.95, "strategy_optimization": 0.85},
            "economic_data": {"macro_analysis": 0.8, "fed_policy": 0.85, "global_correlation": 0.75}
        }
        
        # Thresholds for consensus quality
        self.min_consensus_threshold = 0.6
        self.strong_consensus_threshold = 0.8
        self.minority_preservation_threshold = 0.3
        
    def calculate_weighted_consensus(self, agent_positions: List[Dict], query_context: Dict) -> Dict:
        """Build consensus using agent confidence and evidence strength"""
        
        if not agent_positions:
            return self._create_empty_consensus()
        
        # Calculate individual agent weights
        agent_weights = self._calculate_agent_weights(agent_positions, query_context)
        
        # Group similar positions
        position_groups = self._group_similar_positions(agent_positions)
        
        # Calculate group strengths
        group_strengths = self._calculate_group_strengths(position_groups, agent_weights)
        
        # Identify majority and minority positions
        majority_group, minority_groups = self._identify_majority_minority(position_groups, group_strengths)
        
        # Build consensus from majority position
        consensus = self._build_consensus_from_majority(majority_group, agent_weights)
        
        # Preserve important minority opinions
        preserved_minorities = self._preserve_minority_opinions(minority_groups, agent_weights, consensus)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            majority_group, minority_groups, agent_weights, query_context
        )
        
        # Create final consensus structure
        final_consensus = {
            "recommendation": consensus["recommendation"],
            "confidence_level": overall_confidence,
            "supporting_arguments": consensus["arguments"],
            "evidence_summary": consensus["evidence"],
            "majority_agents": [pos["agent_id"] for pos in majority_group],
            "minority_opinions": preserved_minorities,
            "consensus_metrics": self._calculate_consensus_metrics(
                agent_positions, majority_group, minority_groups, overall_confidence
            ),
            "implementation_guidance": self._generate_implementation_guidance(
                consensus, overall_confidence, preserved_minorities
            ),
            "risk_assessment": self._synthesize_risk_assessment(agent_positions),
            "decision_factors": self._extract_decision_factors(agent_positions)
        }
        
        return final_consensus
    
    def _calculate_agent_weights(self, agent_positions: List[Dict], query_context: Dict) -> Dict[str, float]:
        """Calculate weighted importance of each agent for this specific query"""
        
        weights = {}
        query_topics = self._extract_query_topics(query_context.get("query", ""))
        
        for position in agent_positions:
            agent_id = position["agent_id"]
            
            # Base weight from confidence
            confidence_weight = position.get("confidence_score", 0.5) * self.confidence_weight
            
            # Evidence quality weight
            evidence_quality = self._assess_evidence_quality(position.get("supporting_evidence", []))
            evidence_weight = evidence_quality * self.evidence_weight
            
            # Expertise relevance weight
            expertise_relevance = self._calculate_expertise_relevance(agent_id, query_topics)
            expertise_weight = expertise_relevance * self.expertise_weight
            
            # Position consistency weight (how well arguments support stance)
            consistency = self._assess_position_consistency(position)
            consistency_weight = consistency * self.consistency_weight
            
            # Combined weight
            total_weight = confidence_weight + evidence_weight + expertise_weight + consistency_weight
            weights[agent_id] = min(total_weight, 1.0)  # Cap at 1.0
        
        # Normalize weights so they sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _extract_query_topics(self, query: str) -> List[str]:
        """Extract key topics from query for expertise matching"""
        
        query_lower = query.lower()
        topics = []
        
        topic_keywords = {
            "risk_analysis": ["risk", "volatility", "downside", "protection", "safe"],
            "market_timing": ["timing", "when", "buy", "sell", "market", "trend"],
            "tax_optimization": ["tax", "harvest", "deduction", "after-tax", "efficiency"],
            "options_analysis": ["option", "hedge", "volatility", "strategy", "protection"],
            "portfolio_optimization": ["allocate", "balance", "diversify", "optimize"],
            "macro_analysis": ["economy", "fed", "inflation", "recession", "gdp"],
            "statistical_modeling": ["probability", "correlation", "model", "forecast"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics or ["general_analysis"]
    
    def _calculate_expertise_relevance(self, agent_id: str, query_topics: List[str]) -> float:
        """Calculate how relevant an agent's expertise is to the query"""
        
        agent_skills = self.agent_expertise.get(agent_id, {})
        
        if not query_topics or not agent_skills:
            return 0.5  # Default moderate relevance
        
        # Calculate relevance score
        relevance_scores = []
        for topic in query_topics:
            if topic in agent_skills:
                relevance_scores.append(agent_skills[topic])
            else:
                # Check for related skills
                related_score = self._find_related_expertise(topic, agent_skills)
                relevance_scores.append(related_score)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
    
    def _find_related_expertise(self, topic: str, agent_skills: Dict[str, float]) -> float:
        """Find related expertise when exact match isn't available"""
        
        # Skill relationships
        related_skills = {
            "risk_analysis": ["statistical_modeling", "portfolio_optimization"],
            "market_timing": ["trend_analysis", "sentiment_analysis"],
            "tax_optimization": ["after_tax_analysis", "regulatory_compliance"],
            "options_analysis": ["volatility_analysis", "strategy_optimization"],
            "macro_analysis": ["fed_policy", "global_correlation"]
        }
        
        if topic in related_skills:
            related = related_skills[topic]
            scores = [agent_skills.get(skill, 0) for skill in related]
            return max(scores) * 0.7 if scores else 0.3  # Discount for indirect match
        
        return 0.3  # Default low relevance for unrelated skills
    
    def _assess_evidence_quality(self, evidence: List[Dict]) -> float:
        """Assess the quality and strength of supporting evidence"""
        
        if not evidence:
            return 0.2  # Low score for no evidence
        
        quality_scores = []
        
        for item in evidence:
            score = 0.0
            
            # Evidence type scoring
            evidence_type = item.get("type", "").lower()
            if evidence_type == "statistical":
                score += 0.4
            elif evidence_type == "analytical":
                score += 0.3
            elif evidence_type == "historical":
                score += 0.3
            elif evidence_type == "empirical":
                score += 0.35
            else:
                score += 0.2
            
            # Data quality indicators
            if item.get("source"):
                score += 0.2
            if item.get("confidence", 0) > 0.7:
                score += 0.15
            if item.get("data") and len(str(item["data"])) > 20:  # Substantial data
                score += 0.1
            if item.get("methodology"):
                score += 0.1
            
            quality_scores.append(min(score, 1.0))
        
        # Average quality with bonus for multiple evidence types
        avg_quality = sum(quality_scores) / len(quality_scores)
        diversity_bonus = len(set(item.get("type", "") for item in evidence)) * 0.05
        
        return min(avg_quality + diversity_bonus, 1.0)
    
    def _assess_position_consistency(self, position: Dict) -> float:
        """Assess how well arguments support the stated position"""
        
        stance = position.get("stance", "").lower()
        arguments = position.get("key_arguments", [])
        
        if not stance or not arguments:
            return 0.3
        
        # Simple consistency check - can be enhanced with NLP
        consistency_indicators = {
            "bullish": ["growth", "opportunity", "upside", "positive", "increase"],
            "bearish": ["risk", "decline", "downside", "negative", "decrease"],
            "conservative": ["safe", "protect", "preserve", "stable", "low-risk"],
            "aggressive": ["growth", "opportunity", "high-return", "maximize"]
        }
        
        consistency_score = 0.5  # Base score
        
        for indicator_type, keywords in consistency_indicators.items():
            if indicator_type in stance:
                # Check if arguments contain related keywords
                argument_text = " ".join(arguments).lower()
                keyword_matches = sum(1 for keyword in keywords if keyword in argument_text)
                consistency_score += (keyword_matches / len(keywords)) * 0.3
        
        return min(consistency_score, 1.0)
    
    def _group_similar_positions(self, agent_positions: List[Dict]) -> List[List[Dict]]:
        """Group agents with similar positions"""
        
        if not agent_positions:
            return []
        
        groups = []
        unprocessed = agent_positions.copy()
        
        while unprocessed:
            current_position = unprocessed.pop(0)
            similar_group = [current_position]
            
            # Find similar positions
            to_remove = []
            for other_position in unprocessed:
                if self._positions_similar(current_position, other_position):
                    similar_group.append(other_position)
                    to_remove.append(other_position)
            
            # Remove grouped positions
            for pos in to_remove:
                unprocessed.remove(pos)
            
            groups.append(similar_group)
        
        return groups
    
    def _positions_similar(self, pos1: Dict, pos2: Dict) -> bool:
        """Determine if two positions are similar enough to group"""
        
        # Compare stances
        stance1 = pos1.get("stance", "").lower()
        stance2 = pos2.get("stance", "").lower()
        
        # Direct stance similarity
        if stance1 and stance2:
            if stance1 == stance2:
                return True
            # Check for semantic similarity
            similar_words = self._get_similar_stance_words(stance1, stance2)
            if similar_words:
                return True
        
        # Compare key arguments
        args1 = set(arg.lower() for arg in pos1.get("key_arguments", []))
        args2 = set(arg.lower() for arg in pos2.get("key_arguments", []))
        
        if args1 and args2:
            overlap = len(args1 & args2) / max(len(args1), len(args2))
            if overlap > 0.4:  # 40% argument overlap
                return True
        
        # Compare risk assessments
        risk1 = pos1.get("risk_assessment", {})
        risk2 = pos2.get("risk_assessment", {})
        
        if risk1 and risk2:
            risks1 = set(risk1.get("primary_risks", []))
            risks2 = set(risk2.get("primary_risks", []))
            if risks1 and risks2:
                risk_overlap = len(risks1 & risks2) / max(len(risks1), len(risks2))
                if risk_overlap > 0.5:  # 50% risk overlap
                    return True
        
        return False
    
    def _get_similar_stance_words(self, stance1: str, stance2: str) -> bool:
        """Check for semantically similar stance words"""
        
        similar_groups = [
            ["bullish", "optimistic", "positive", "growth", "aggressive"],
            ["bearish", "pessimistic", "negative", "decline", "conservative"],
            ["neutral", "balanced", "moderate", "cautious"],
            ["buy", "purchase", "acquire", "increase"],
            ["sell", "reduce", "decrease", "exit"],
            ["hold", "maintain", "keep", "stay"]
        ]
        
        for group in similar_groups:
            if any(word in stance1 for word in group) and any(word in stance2 for word in group):
                return True
        
        return False
    
    def _calculate_group_strengths(self, position_groups: List[List[Dict]], agent_weights: Dict[str, float]) -> Dict[int, float]:
        """Calculate the strength of each position group"""
        
        group_strengths = {}
        
        for i, group in enumerate(position_groups):
            total_weight = 0.0
            total_confidence = 0.0
            evidence_quality = 0.0
            
            for position in group:
                agent_id = position["agent_id"]
                agent_weight = agent_weights.get(agent_id, 0.5)
                agent_confidence = position.get("confidence_score", 0.5)
                agent_evidence_quality = self._assess_evidence_quality(
                    position.get("supporting_evidence", [])
                )
                
                total_weight += agent_weight
                total_confidence += agent_confidence * agent_weight
                evidence_quality += agent_evidence_quality * agent_weight
            
            # Group strength combines weight, confidence, and evidence
            if total_weight > 0:
                avg_confidence = total_confidence / total_weight
                avg_evidence = evidence_quality / total_weight
                group_strength = (total_weight * 0.5) + (avg_confidence * 0.3) + (avg_evidence * 0.2)
            else:
                group_strength = 0.0
            
            group_strengths[i] = group_strength
        
        return group_strengths
    
    def _identify_majority_minority(self, position_groups: List[List[Dict]], group_strengths: Dict[int, float]) -> Tuple[List[Dict], List[List[Dict]]]:
        """Identify majority position and minority positions"""
        
        if not position_groups:
            return [], []
        
        # Find strongest group as majority
        strongest_group_idx = max(group_strengths.items(), key=lambda x: x[1])[0]
        majority_group = position_groups[strongest_group_idx]
        
        # All other groups are minorities
        minority_groups = [group for i, group in enumerate(position_groups) if i != strongest_group_idx]
        
        return majority_group, minority_groups
    
    def _build_consensus_from_majority(self, majority_group: List[Dict], agent_weights: Dict[str, float]) -> Dict:
        """Build consensus recommendation from majority position"""
        
        if not majority_group:
            return {"recommendation": "No clear consensus", "arguments": [], "evidence": []}
        
        # Aggregate arguments from majority agents
        all_arguments = []
        all_evidence = []
        total_weight = 0.0
        
        for position in majority_group:
            agent_weight = agent_weights.get(position["agent_id"], 0.5)
            total_weight += agent_weight
            
            # Weight arguments by agent importance
            for arg in position.get("key_arguments", []):
                all_arguments.append((arg, agent_weight))
            
            # Collect evidence
            all_evidence.extend(position.get("supporting_evidence", []))
        
        # Sort arguments by weight and remove duplicates
        weighted_args = {}
        for arg, weight in all_arguments:
            if arg in weighted_args:
                weighted_args[arg] += weight
            else:
                weighted_args[arg] = weight
        
        # Select top arguments
        top_arguments = sorted(weighted_args.items(), key=lambda x: x[1], reverse=True)[:5]
        consensus_arguments = [arg for arg, weight in top_arguments]
        
        # Create recommendation from majority stance and top arguments
        primary_stance = majority_group[0].get("stance", "balanced approach")
        recommendation = f"{primary_stance}. Key considerations: {', '.join(consensus_arguments[:3])}"
        
        return {
            "recommendation": recommendation,
            "arguments": consensus_arguments,
            "evidence": all_evidence[:10]  # Limit evidence for readability
        }
    
    def preserve_minority_opinions(self, minority_groups: List[List[Dict]], agent_weights: Dict[str, float], consensus: Dict) -> List[MinorityOpinion]:
        """Ensure valuable minority perspectives are preserved"""
        
        preserved_minorities = []
        
        for group in minority_groups:
            if not group:
                continue
            
            # Calculate minority strength
            group_weight = sum(agent_weights.get(pos["agent_id"], 0.5) for pos in group)
            
            # Only preserve minorities above threshold
            if group_weight < self.minority_preservation_threshold:
                continue
            
            # Assess risk of ignoring this minority opinion
            risk_if_ignored = self._assess_minority_risk(group, consensus)
            
            # Calculate importance score
            importance_score = self._calculate_minority_importance(group, group_weight, risk_if_ignored)
            
            if importance_score > 0.4:  # Threshold for preservation
                minority_opinion = MinorityOpinion(
                    agents=[pos["agent_id"] for pos in group],
                    position=group[0].get("stance", "alternative view"),
                    arguments=self._aggregate_minority_arguments(group),
                    evidence=self._aggregate_minority_evidence(group),
                    weight=group_weight,
                    risk_if_ignored=risk_if_ignored,
                    importance_score=importance_score
                )
                preserved_minorities.append(minority_opinion)
        
        return preserved_minorities
    
    def _assess_minority_risk(self, minority_group: List[Dict], consensus: Dict) -> float:
        """Assess risk of ignoring minority opinion"""
        
        # Check if minority identifies risks not in consensus
        minority_risks = set()
        for position in minority_group:
            minority_risks.update(position.get("risk_assessment", {}).get("primary_risks", []))
        
        consensus_risks = set(consensus.get("risk_assessment", {}).get("primary_risks", []))
        
        unique_minority_risks = minority_risks - consensus_risks
        risk_score = len(unique_minority_risks) / max(len(minority_risks), 1)
        
        # Higher risk if minority has high confidence in different direction
        confidence_divergence = 0.0
        for position in minority_group:
            confidence = position.get("confidence_score", 0.5)
            if confidence > 0.7:  # High confidence minority
                confidence_divergence += 0.2
        
        return min(risk_score + confidence_divergence, 1.0)
    
    def _calculate_minority_importance(self, group: List[Dict], weight: float, risk: float) -> float:
        """Calculate overall importance of preserving this minority opinion"""
        
        # Factors: group weight, risk if ignored, evidence quality, expertise relevance
        avg_evidence_quality = sum(
            self._assess_evidence_quality(pos.get("supporting_evidence", []))
            for pos in group
        ) / len(group)
        
        # Weight different factors
        importance = (
            weight * 0.4 +  # Group influence
            risk * 0.3 +    # Risk of ignoring
            avg_evidence_quality * 0.2 +  # Evidence quality
            0.1  # Base importance for diversity
        )
        
        return min(importance, 1.0)
    
    def _aggregate_minority_arguments(self, group: List[Dict]) -> List[str]:
        """Aggregate arguments from minority group"""
        all_args = []
        for position in group:
            all_args.extend(position.get("key_arguments", []))
        
        # Remove duplicates while preserving order
        unique_args = []
        seen = set()
        for arg in all_args:
            if arg.lower() not in seen:
                unique_args.append(arg)
                seen.add(arg.lower())
        
        return unique_args[:3]  # Limit to top 3 arguments
    
    def _aggregate_minority_evidence(self, group: List[Dict]) -> List[Dict]:
        """Aggregate evidence from minority group"""
        all_evidence = []
        for position in group:
            all_evidence.extend(position.get("supporting_evidence", []))
        
        # Sort by quality and return top evidence
        evidence_with_quality = [
            (ev, self._assess_single_evidence_quality(ev))
            for ev in all_evidence
        ]
        evidence_with_quality.sort(key=lambda x: x[1], reverse=True)
        
        return [ev for ev, quality in evidence_with_quality[:3]]
    
    def _assess_single_evidence_quality(self, evidence: Dict) -> float:
        """Assess quality of a single piece of evidence"""
        score = 0.0
        
        if evidence.get("type") == "statistical":
            score += 0.4
        elif evidence.get("type") == "analytical":
            score += 0.3
        
        if evidence.get("source"):
            score += 0.2
        if evidence.get("confidence", 0) > 0.7:
            score += 0.2
        if evidence.get("data"):
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_overall_confidence(self, majority_group: List[Dict], minority_groups: List[List[Dict]], agent_weights: Dict[str, float], query_context: Dict) -> float:
        """Calculate overall confidence in consensus"""
        
        # Base confidence from majority group
        majority_confidence = 0.0
        majority_weight = 0.0
        
        for position in majority_group:
            agent_weight = agent_weights.get(position["agent_id"], 0.5)
            agent_confidence = position.get("confidence_score", 0.5)
            majority_confidence += agent_confidence * agent_weight
            majority_weight += agent_weight
        
        if majority_weight > 0:
            base_confidence = majority_confidence / majority_weight
        else:
            base_confidence = 0.5
        
        # Adjust for minority strength
        total_minority_weight = sum(
            sum(agent_weights.get(pos["agent_id"], 0.5) for pos in group)
            for group in minority_groups
        )
        
        # Strong minorities reduce confidence
        if total_minority_weight > 0.3:
            minority_penalty = (total_minority_weight - 0.3) * 0.5
            base_confidence -= minority_penalty
        
        # Adjust for evidence quality
        all_positions = majority_group + [pos for group in minority_groups for pos in group]
        avg_evidence_quality = sum(
            self._assess_evidence_quality(pos.get("supporting_evidence", []))
            for pos in all_positions
        ) / len(all_positions) if all_positions else 0.5
        
        evidence_boost = (avg_evidence_quality - 0.5) * 0.2
        
        # Adjust for query complexity
        complexity = query_context.get("complexity", "medium")
        complexity_penalty = {"high": 0.1, "medium": 0.05, "low": 0.0}.get(complexity, 0.05)
        
        final_confidence = base_confidence + evidence_boost - complexity_penalty
        
        return max(min(final_confidence, 1.0), 0.0)  # Clamp to [0, 1]
    
    def _calculate_consensus_metrics(self, agent_positions: List[Dict], majority_group: List[Dict], minority_groups: List[List[Dict]], overall_confidence: float) -> ConsensusMetrics:
        """Calculate comprehensive metrics for consensus quality"""
        
        # Agreement level
        agreement_level = len(majority_group) / len(agent_positions) if agent_positions else 0.0
        
        # Evidence strength
        evidence_strength = sum(
            self._assess_evidence_quality(pos.get("supporting_evidence", []))
            for pos in agent_positions
        ) / len(agent_positions) if agent_positions else 0.0
        
        # Confidence distribution
        confidence_distribution = {
            pos["agent_id"]: pos.get("confidence_score", 0.5)
            for pos in agent_positions
        }
        
        # Minority strength
        minority_strength = sum(len(group) for group in minority_groups) / len(agent_positions) if agent_positions else 0.0
        
        return ConsensusMetrics(
            agreement_level=agreement_level,
            evidence_strength=evidence_strength,
            confidence_distribution=confidence_distribution,
            minority_strength=minority_strength,
            overall_confidence=overall_confidence
        )
    
    def _generate_implementation_guidance(self, consensus: Dict, confidence: float, minorities: List[MinorityOpinion]) -> Dict:
        """Generate guidance for implementing the consensus decision"""
        
        if confidence > self.strong_consensus_threshold:
            urgency = "high"
            approach = "proceed_with_confidence"
        elif confidence > self.min_consensus_threshold:
            urgency = "medium"
            approach = "proceed_with_caution"
        else:
            urgency = "low"
            approach = "seek_additional_analysis"
        
        # Check for high-risk minorities
        high_risk_minorities = [m for m in minorities if m.risk_if_ignored > 0.6]
        
        guidance = {
            "implementation_urgency": urgency,
            "recommended_approach": approach,
            "confidence_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
            "risk_mitigation_needed": len(high_risk_minorities) > 0,
            "monitoring_recommendations": self._generate_monitoring_recommendations(minorities),
            "decision_timeline": self._suggest_decision_timeline(confidence, urgency)
        }
        
        return guidance
    
    def _generate_monitoring_recommendations(self, minorities: List[MinorityOpinion]) -> List[str]:
        """Generate recommendations for monitoring minority risk factors"""
        
        recommendations = []
        
        for minority in minorities:
            if minority.risk_if_ignored > 0.5:
                recommendations.append(
                    f"Monitor concerns raised by {', '.join(minority.agents)}: {minority.arguments[0] if minority.arguments else 'alternative perspective'}"
                )
        
        if not recommendations:
            recommendations.append("No specific monitoring required - strong consensus achieved")
        
        return recommendations
    
    def _suggest_decision_timeline(self, confidence: float, urgency: str) -> str:
        """Suggest appropriate timeline for decision implementation"""
        
        if confidence > 0.8 and urgency == "high":
            return "immediate"
        elif confidence > 0.6:
            return "within_24_hours"
        elif confidence > 0.4:
            return "within_week"
        else:
            return "seek_additional_input"
    
    def _synthesize_risk_assessment(self, agent_positions: List[Dict]) -> Dict:
        """Synthesize risk assessment from all agent positions"""
        
        all_risks = []
        risk_mitigation = []
        
        for position in agent_positions:
            risk_data = position.get("risk_assessment", {})
            all_risks.extend(risk_data.get("primary_risks", []))
            risk_mitigation.extend(risk_data.get("mitigation_strategies", []))
        
        # Count risk frequency
        risk_counts = defaultdict(int)
        for risk in all_risks:
            risk_counts[risk] += 1
        
        # Sort by frequency (consensus on risks)
        consensus_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_risks": [risk for risk, count in consensus_risks[:5]],
            "risk_consensus_level": consensus_risks[0][1] / len(agent_positions) if consensus_risks else 0.0,
            "mitigation_strategies": list(set(risk_mitigation))[:5],
            "risk_diversity": len(set(all_risks))
        }
    
    def _extract_decision_factors(self, agent_positions: List[Dict]) -> Dict:
        """Extract key decision factors from agent analysis"""
        
        factors = {
            "market_conditions": [],
            "portfolio_specific": [],
            "timing_considerations": [],
            "risk_factors": [],
            "opportunity_factors": []
        }
        
        for position in agent_positions:
            # Extract from arguments - simplified categorization
            arguments = position.get("key_arguments", [])
            for arg in arguments:
                arg_lower = arg.lower()
                if any(word in arg_lower for word in ["market", "economy", "fed"]):
                    factors["market_conditions"].append(arg)
                elif any(word in arg_lower for word in ["portfolio", "allocation", "holding"]):
                    factors["portfolio_specific"].append(arg)
                elif any(word in arg_lower for word in ["timing", "when", "now"]):
                    factors["timing_considerations"].append(arg)
                elif any(word in arg_lower for word in ["risk", "loss", "downside"]):
                    factors["risk_factors"].append(arg)
                elif any(word in arg_lower for word in ["opportunity", "growth", "upside"]):
                    factors["opportunity_factors"].append(arg)
        
        # Remove duplicates and limit
        for category in factors:
            factors[category] = list(set(factors[category]))[:3]
        
        return factors
    
    def _create_empty_consensus(self) -> Dict:
        """Create empty consensus structure when no positions available"""
        return {
            "recommendation": "Insufficient agent input for consensus",
            "confidence_level": 0.0,
            "supporting_arguments": [],
            "evidence_summary": [],
            "majority_agents": [],
            "minority_opinions": [],
            "consensus_metrics": ConsensusMetrics(0, 0, {}, 0, 0),
            "implementation_guidance": {"recommended_approach": "seek_additional_analysis"},
            "risk_assessment": {},
            "decision_factors": {}
        }
    
    def generate_debate_confidence_score(self, debate_results: Dict) -> float:
        """Calculate overall confidence in debate conclusions"""
        
        # Factors affecting debate confidence
        factors = {
            "agent_agreement": 0.0,
            "evidence_strength": 0.0,
            "query_complexity_handling": 0.0,
            "debate_quality": 0.0
        }
        
        consensus = debate_results.get("consensus", {})
        rounds = debate_results.get("rounds", [])
        participants = debate_results.get("participants", {})
        
        # 1. Agent Agreement Factor
        if consensus:
            majority_size = len(consensus.get("majority_agents", []))
            total_agents = len(participants)
            if total_agents > 0:
                factors["agent_agreement"] = majority_size / total_agents
        
        # 2. Evidence Strength Factor
        evidence_summary = consensus.get("evidence_summary", [])
        if evidence_summary:
            avg_evidence_quality = sum(
                self._assess_single_evidence_quality(ev) for ev in evidence_summary
            ) / len(evidence_summary)
            factors["evidence_strength"] = avg_evidence_quality
        
        # 3. Query Complexity Handling
        query_complexity = debate_results.get("query_analysis", {}).get("complexity_level", "medium")
        complexity_scores = {"low": 0.9, "medium": 0.7, "high": 0.5}
        factors["query_complexity_handling"] = complexity_scores.get(query_complexity, 0.7)
        
        # 4. Debate Quality Factor
        if rounds:
            # Quality indicators: number of rounds, challenges, responses
            total_challenges = sum(len(r.get("challenges", [])) for r in rounds)
            total_responses = sum(len(r.get("responses", [])) for r in rounds)
            
            debate_engagement = min((total_challenges + total_responses) / (len(participants) * 2), 1.0)
            factors["debate_quality"] = debate_engagement
        
        # Weighted combination
        weights = {
            "agent_agreement": 0.4,
            "evidence_strength": 0.3,
            "query_complexity_handling": 0.2,
            "debate_quality": 0.1
        }
        
        final_confidence = sum(factors[factor] * weights[factor] for factor in factors)
        
        return min(max(final_confidence, 0.0), 1.0)  # Clamp to [0, 1]
    
    def evaluate_consensus_quality(self, consensus_result: Dict) -> Dict:
        """Evaluate the quality of a consensus result"""
        
        metrics = consensus_result.get("consensus_metrics")
        if not metrics:
            return {"quality": "unknown", "score": 0.0}
        
        # Quality dimensions
        dimensions = {
            "agreement_strength": metrics.agreement_level,
            "evidence_quality": metrics.evidence_strength,
            "confidence_consistency": self._calculate_confidence_consistency(metrics.confidence_distribution),
            "minority_handling": 1.0 - metrics.minority_strength if metrics.minority_strength < 0.5 else 0.5
        }
        
        # Overall quality score
        quality_score = sum(dimensions.values()) / len(dimensions)
        
        # Quality categories
        if quality_score > 0.8:
            quality_level = "excellent"
        elif quality_score > 0.6:
            quality_level = "good"
        elif quality_score > 0.4:
            quality_level = "acceptable"
        else:
            quality_level = "poor"
        
        return {
            "quality": quality_level,
            "score": quality_score,
            "dimensions": dimensions,
            "recommendations": self._generate_quality_recommendations(dimensions, quality_level)
        }
    
    def _calculate_confidence_consistency(self, confidence_dist: Dict[str, float]) -> float:
        """Calculate how consistent agent confidence levels are"""
        
        if not confidence_dist:
            return 0.0
        
        confidences = list(confidence_dist.values())
        if len(confidences) <= 1:
            return 1.0
        
        # Calculate coefficient of variation (std dev / mean)
        mean_confidence = sum(confidences) / len(confidences)
        if mean_confidence == 0:
            return 0.0
        
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        std_dev = variance ** 0.5
        cv = std_dev / mean_confidence
        
        # Convert to consistency score (lower CV = higher consistency)
        consistency = max(0.0, 1.0 - cv)
        return consistency
    
    def _generate_quality_recommendations(self, dimensions: Dict, quality_level: str) -> List[str]:
        """Generate recommendations for improving consensus quality"""
        
        recommendations = []
        
        if dimensions["agreement_strength"] < 0.6:
            recommendations.append("Consider additional debate rounds to increase agent agreement")
        
        if dimensions["evidence_quality"] < 0.7:
            recommendations.append("Encourage agents to provide higher quality evidence and sources")
        
        if dimensions["confidence_consistency"] < 0.6:
            recommendations.append("Investigate why agent confidence levels vary significantly")
        
        if dimensions["minority_handling"] < 0.7:
            recommendations.append("Strong minority opinions present - consider preserving alternative viewpoints")
        
        if quality_level == "poor":
            recommendations.append("Consider seeking additional expert input or reformulating the query")
        
        return recommendations
    
    def update_agent_expertise(self, agent_id: str, topic: str, performance_score: float):
        """Update agent expertise scores based on performance"""
        
        if agent_id not in self.agent_expertise:
            self.agent_expertise[agent_id] = {}
        
        current_score = self.agent_expertise[agent_id].get(topic, 0.5)
        
        # Exponential moving average for score updates
        alpha = 0.1  # Learning rate
        new_score = (1 - alpha) * current_score + alpha * performance_score
        
        self.agent_expertise[agent_id][topic] = min(max(new_score, 0.0), 1.0)
    
    def get_consensus_summary(self, consensus_result: Dict) -> str:
        """Generate a human-readable summary of the consensus"""
        
        recommendation = consensus_result.get("recommendation", "No recommendation available")
        confidence = consensus_result.get("confidence_level", 0.0)
        majority_agents = consensus_result.get("majority_agents", [])
        minority_opinions = consensus_result.get("minority_opinions", [])
        
        summary = f"**Consensus Recommendation:** {recommendation}\n\n"
        summary += f"**Confidence Level:** {confidence:.1%}\n\n"
        
        if majority_agents:
            summary += f"**Supporting Agents:** {', '.join(majority_agents)}\n\n"
        
        if minority_opinions:
            summary += f"**Alternative Perspectives ({len(minority_opinions)}):**\n"
            for i, minority in enumerate(minority_opinions):
                summary += f"  {i+1}. {minority.position} (supported by {', '.join(minority.agents)})\n"
            summary += "\n"
        
        implementation = consensus_result.get("implementation_guidance", {})
        if implementation:
            urgency = implementation.get("implementation_urgency", "medium")
            summary += f"**Implementation Urgency:** {urgency}\n"
        
        return summary