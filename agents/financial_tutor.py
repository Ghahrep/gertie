# in agents/financial_tutor.py
from typing import Dict, Any, Optional, List
import re
import asyncio

from agents.mcp_base_agent import MCPBaseAgent

class FinancialTutorAgent(MCPBaseAgent):
    """
    Acts as a knowledge translator, explaining complex financial concepts to the user.
    Migrated to MCP architecture with enhanced capabilities and async execution.
    """
    
    def __init__(self):
        # Define MCP capabilities
        capabilities = [
            "concept_explanation",
            "term_definition", 
            "educational_guidance",
            "learning_support",
            "knowledge_synthesis"
        ]
        
        super().__init__(
            agent_id="financial_tutor",
            agent_name="FinancialTutor", 
            capabilities=capabilities
        )
        
        # Enhanced knowledge base with categorized content
        self._knowledge_base = {
            # Risk Management
            "cvar": {
                "explanation": (
                    "**Conditional Value at Risk (CVaR)**, also known as Expected Shortfall, is a risk measure that answers the question: "
                    "'If things get really bad, what is my average expected loss?' For a 95% CVaR, it calculates the average loss on the worst 5% of days."
                ),
                "category": "risk_management",
                "complexity": "intermediate",
                "related_terms": ["var", "expected shortfall", "tail risk"]
            },
            "var": {
                "explanation": (
                    "**Value at Risk (VaR)** is a statistic that quantifies the extent of possible financial losses within a firm, portfolio, or position over a specific time frame. "
                    "For example, a 95% VaR of $10,000 means there is a 5% chance of losing at least $10,000 on any given day."
                ),
                "category": "risk_management",
                "complexity": "beginner",
                "related_terms": ["cvar", "risk measure", "portfolio risk"]
            },
            
            # Performance Metrics
            "sharpe ratio": {
                "explanation": (
                    "The **Sharpe Ratio** is a measure of risk-adjusted return. It tells you how much return you are getting for each unit of risk you take (where risk is measured by volatility). "
                    "A higher Sharpe Ratio is generally better, indicating a more efficient portfolio. Formula: (Return - Risk-free rate) / Standard deviation."
                ),
                "category": "performance_metrics",
                "complexity": "intermediate",
                "related_terms": ["sortino ratio", "risk-adjusted return", "volatility"]
            },
            "sortino ratio": {
                "explanation": (
                    "The **Sortino Ratio** is a variation of the Sharpe Ratio that only penalizes for 'bad' volatility (downside deviation). "
                    "It's useful because it doesn't punish a portfolio for having strong positive returns, giving a better picture of its performance relative to downside risk."
                ),
                "category": "performance_metrics", 
                "complexity": "intermediate",
                "related_terms": ["sharpe ratio", "downside deviation", "risk-adjusted return"]
            },
            
            # Portfolio Management
            "herc": {
                "explanation": (
                    "**Hierarchical Risk Parity (HERC)** is a modern portfolio optimization method. Unlike traditional models that rely on predicting returns, "
                    "HERC structures the portfolio by grouping similar assets together and then diversifying risk across those groups. It's often more robust and stable."
                ),
                "category": "portfolio_management",
                "complexity": "advanced",
                "related_terms": ["risk parity", "hierarchical clustering", "portfolio optimization"]
            },
            
            # Basic Concepts
            "diversification": {
                "explanation": (
                    "**Diversification** is the practice of spreading investments across various financial instruments, industries, and other categories to reduce risk. "
                    "The idea is that a portfolio of different kinds of investments will, on average, yield higher returns and pose a lower risk than any individual investment."
                ),
                "category": "basic_concepts",
                "complexity": "beginner", 
                "related_terms": ["risk reduction", "portfolio construction", "correlation"]
            },
            "volatility": {
                "explanation": (
                    "**Volatility** measures how much the price of an asset fluctuates over time. High volatility means large price swings, while low volatility means more stable prices. "
                    "It's often used as a proxy for risk - more volatile assets are generally considered riskier."
                ),
                "category": "basic_concepts",
                "complexity": "beginner",
                "related_terms": ["standard deviation", "risk", "price movement"]
            }
        }
        
        # Learning pathways for progressive education
        self._learning_paths = {
            "beginner": ["diversification", "volatility", "var"],
            "intermediate": ["sharpe ratio", "sortino ratio", "cvar"],  
            "advanced": ["herc", "portfolio optimization", "risk modeling"]
        }
    
    @property
    def name(self) -> str:
        return "FinancialTutor"
    
    @property
    def purpose(self) -> str:
        return "Explains financial concepts, terms, and strategies in an easy-to-understand way."
    
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict[str, Any]:
        """Execute MCP capability with routing to appropriate methods"""
        
        try:
            if capability == "concept_explanation":
                return await self._explain_concept(data, context)
            elif capability == "term_definition":
                return await self._define_term(data, context)
            elif capability == "educational_guidance":
                return await self._provide_guidance(data, context)
            elif capability == "learning_support":
                return await self._support_learning(data, context)
            elif capability == "knowledge_synthesis":
                return await self._synthesize_knowledge(data, context)
            else:
                return {
                    "success": False,
                    "error": f"Unknown capability: {capability}",
                    "error_type": "invalid_capability"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error executing {capability}: {str(e)}",
                "error_type": "execution_error"
            }
    
    async def _explain_concept(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Provide detailed concept explanation"""
        
        query = data.get("query", "").lower()
        
        # Find matching concept
        for concept, info in self._knowledge_base.items():
            if re.search(r'\b' + re.escape(concept) + r'\b', query):
                
                # Build comprehensive explanation
                explanation = info["explanation"]
                
                # Add related concepts
                related = info.get("related_terms", [])
                if related:
                    explanation += f"\n\n**Related concepts:** {', '.join(related)}"
                
                # Add complexity indicator
                complexity = info.get("complexity", "intermediate")
                explanation += f"\n\n**Complexity level:** {complexity.title()}"
                
                return {
                    "success": True,
                    "explanation": explanation,
                    "concept": concept,
                    "category": info.get("category", "general"),
                    "complexity": complexity,
                    "related_terms": related
                }
        
        # No specific concept found - provide general guidance
        return await self._provide_general_help(query)
    
    async def _define_term(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Provide concise term definition"""
        
        query = data.get("query", "").lower()
        
        # Extract key definition from explanation
        for concept, info in self._knowledge_base.items():
            if concept in query:
                # Extract first sentence as definition
                explanation = info["explanation"]
                definition = explanation.split('.')[0] + '.'
                
                return {
                    "success": True,
                    "term": concept,
                    "definition": definition,
                    "full_explanation_available": True
                }
        
        return {
            "success": False,
            "error": "Term not found in knowledge base",
            "available_terms": list(self._knowledge_base.keys())
        }
    
    async def _provide_guidance(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Provide educational guidance and learning paths"""
        
        user_level = data.get("level", "beginner").lower()
        topic_area = data.get("topic", "general").lower()
        
        # Get appropriate learning path
        if user_level in self._learning_paths:
            recommended_concepts = self._learning_paths[user_level]
        else:
            recommended_concepts = self._learning_paths["beginner"]
        
        # Filter by topic area if specified
        if topic_area != "general":
            filtered_concepts = []
            for concept in recommended_concepts:
                if concept in self._knowledge_base:
                    if self._knowledge_base[concept].get("category") == topic_area:
                        filtered_concepts.append(concept)
            if filtered_concepts:
                recommended_concepts = filtered_concepts
        
        guidance = f"**Learning Path for {user_level.title()} Level:**\n\n"
        
        for i, concept in enumerate(recommended_concepts, 1):
            if concept in self._knowledge_base:
                guidance += f"{i}. **{concept.title()}** - {self._knowledge_base[concept]['complexity']} level\n"
        
        guidance += f"\nAsk me about any of these concepts to get detailed explanations!"
        
        return {
            "success": True,
            "guidance": guidance,
            "recommended_concepts": recommended_concepts,
            "user_level": user_level
        }
    
    async def _support_learning(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Provide learning support and study suggestions"""
        
        concept = data.get("concept", "").lower()
        
        if concept in self._knowledge_base:
            info = self._knowledge_base[concept]
            
            # Generate study suggestions
            suggestions = []
            complexity = info.get("complexity", "intermediate")
            
            if complexity == "beginner":
                suggestions = [
                    "Start with the basic definition and key characteristics",
                    "Look for real-world examples in financial news", 
                    "Practice identifying this concept in market scenarios",
                    "Connect it to your existing investment knowledge"
                ]
            elif complexity == "intermediate":
                suggestions = [
                    "Understand the mathematical foundation and calculations",
                    "Study practical applications in portfolio management",
                    "Compare with related concepts and metrics",
                    "Analyze case studies showing practical usage"
                ]
            else:  # advanced
                suggestions = [
                    "Dive deep into the theoretical framework",
                    "Examine academic research and empirical studies",
                    "Implement practical calculations or simulations", 
                    "Consider limitations and alternative approaches"
                ]
            
            return {
                "success": True,
                "concept": concept,
                "study_suggestions": suggestions,
                "complexity": complexity,
                "related_terms": info.get("related_terms", [])
            }
        
        return {
            "success": False,
            "error": f"Concept '{concept}' not found in knowledge base"
        }
    
    async def _synthesize_knowledge(self, data: Dict, context: Dict) -> Dict[str, Any]:
        """Synthesize knowledge across multiple concepts"""
        
        concepts = data.get("concepts", [])
        if isinstance(concepts, str):
            concepts = [c.strip() for c in concepts.split(",")]
        
        if not concepts:
            # Extract concepts from query
            query = data.get("query", "").lower()
            concepts = [concept for concept in self._knowledge_base.keys() 
                       if concept in query]
        
        if not concepts:
            return {
                "success": False, 
                "error": "No concepts specified for synthesis"
            }
        
        # Find concepts in knowledge base
        found_concepts = {}
        for concept in concepts:
            concept_lower = concept.lower()
            if concept_lower in self._knowledge_base:
                found_concepts[concept_lower] = self._knowledge_base[concept_lower]
        
        if not found_concepts:
            return {
                "success": False,
                "error": "None of the specified concepts found in knowledge base"
            }
        
        # Generate synthesis
        synthesis = "**Knowledge Synthesis:**\n\n"
        
        # Group by category
        categories = {}
        for concept, info in found_concepts.items():
            category = info.get("category", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append((concept, info))
        
        # Present synthesis by category
        for category, concept_list in categories.items():
            synthesis += f"**{category.replace('_', ' ').title()}:**\n"
            for concept, info in concept_list:
                synthesis += f"- **{concept.title()}:** {info['explanation'].split('.')[0]}.\n"
            synthesis += "\n"
        
        # Add connections
        all_related = set()
        for concept, info in found_concepts.items():
            all_related.update(info.get("related_terms", []))
        
        if all_related:
            synthesis += f"**Connected Concepts:** {', '.join(sorted(all_related))}\n"
        
        return {
            "success": True,
            "synthesis": synthesis,
            "concepts_covered": list(found_concepts.keys()),
            "categories": list(categories.keys())
        }
    
    async def _provide_general_help(self, query: str) -> Dict[str, Any]:
        """Provide general help when no specific concept is found"""
        
        # Categorize available concepts
        categories = {}
        for concept, info in self._knowledge_base.items():
            category = info.get("category", "general")
            if category not in categories:
                categories[category] = []
            categories[category].append(concept)
        
        help_text = "I can explain various financial concepts organized by category:\n\n"
        
        for category, concepts in categories.items():
            help_text += f"**{category.replace('_', ' ').title()}:**\n"
            for concept in sorted(concepts):
                complexity = self._knowledge_base[concept].get("complexity", "intermediate")
                help_text += f"- {concept.title()} ({complexity})\n"
            help_text += "\n"
        
        help_text += "Ask me about any of these concepts, or request guidance for your learning level!"
        
        return {
            "success": True,
            "explanation": help_text,
            "available_categories": list(categories.keys()),
            "total_concepts": len(self._knowledge_base)
        }
    
    def _generate_summary(self, result: Dict, capability: str, execution_time: float = 0.0) -> str:
        """Generate custom summary for financial tutor responses"""
        
        if not result.get("success"):
            return f"Unable to provide explanation: {result.get('error', 'Unknown error')}"
        
        if capability == "concept_explanation":
            concept = result.get("concept", "financial concept")
            complexity = result.get("complexity", "intermediate")
            return f"**{concept.title()}** explained ({complexity} level) with related concepts and practical context"
            
        elif capability == "educational_guidance":
            level = result.get("user_level", "beginner")
            concept_count = len(result.get("recommended_concepts", []))
            return f"**Learning guidance** provided for {level} level with {concept_count} recommended concepts"
            
        elif capability == "knowledge_synthesis":
            concept_count = len(result.get("concepts_covered", []))
            category_count = len(result.get("categories", []))
            return f"**Knowledge synthesis** completed across {concept_count} concepts in {category_count} categories"
            
        elif capability == "learning_support":
            concept = result.get("concept", "concept")
            return f"**Study support** provided for {concept} with tailored learning suggestions"
            
        else:
            return f"**Financial education** completed successfully in {execution_time:.2f}s"
    
    async def _health_check_capability(self, capability: str) -> bool:
        """Health check for specific capabilities"""
        
        test_data = {
            "concept_explanation": {"query": "what is var"},
            "term_definition": {"query": "define sharpe ratio"},
            "educational_guidance": {"level": "beginner"},
            "learning_support": {"concept": "diversification"},
            "knowledge_synthesis": {"concepts": ["var", "cvar"]}
        }
        
        if capability in test_data:
            try:
                result = await self.execute_capability(capability, test_data[capability], {})
                return result.get("success", False)
            except:
                return False
        
        return False