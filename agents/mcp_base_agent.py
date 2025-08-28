# agents/mcp_base_agent.py
import asyncio
import aiohttp
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import time

# Try to import MCP schemas, but don't fail if not available
try:
    from mcp.schemas import (
        AgentRegistration, 
        AgentJobRequest, 
        AgentJobResponse, 
        JobStatus,
        AgentCapability
    )
    MCP_AVAILABLE = True
except ImportError:
    # MCP schemas not available - use simple alternatives
    MCP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("MCP schemas not available - using simplified mode")

logger = logging.getLogger(__name__)

class MCPBaseAgent(ABC):
    """
    Enhanced MCP-compatible base agent with orchestrator integration.
    
    Features:
    - Full MCP compatibility when schemas available
    - Orchestrator compatibility via run() method
    - Autonomous data fetching
    - Standardized error handling
    - Health monitoring
    - Production-ready async patterns
    """
    
    def __init__(
        self, 
        agent_id: str, 
        agent_name: str, 
        capabilities: List[str],
        mcp_url: str = "http://localhost:8001",
        max_concurrent_jobs: int = 5,
        enable_mcp_registration: bool = False  # NEW: Control MCP registration
    ):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.capabilities = capabilities
        self.mcp_url = mcp_url
        self.max_concurrent_jobs = max_concurrent_jobs
        self.enable_mcp_registration = enable_mcp_registration
        self.is_registered = False
        self.active_jobs: Dict[str, asyncio.Task] = {}
        
        # Enhanced logging
        self.logger = logging.getLogger(f"agent.{agent_id}")
        
        # HTTP session for MCP communication (only if needed)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Performance tracking
        self.execution_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "avg_response_time_ms": 0.0
        }
    
    # ==========================================
    # NEW: ORCHESTRATOR COMPATIBILITY METHODS
    # ==========================================
    
    async def run(self, user_query: str, context: Dict = None) -> Dict[str, Any]:
        """
        🔥 MAIN ENTRY POINT for orchestrator compatibility.
        This bridges orchestrator expectations with MCP architecture.
        """
        if context is None:
            context = {}
            
        self.logger.info(f"--- {self.agent_name} received query: '{user_query}' ---")
        start_time = time.time()
        
        try:
            # Update stats
            self.execution_stats["total_requests"] += 1
            
            # Determine capability from query
            capability = self._determine_capability_from_query(user_query)
            self.logger.info(f"Selected capability: {capability}")
            
            # Autonomous data enrichment (MCP feature)
            enriched_data = await self.autonomous_data_fetch(user_query, context)
            combined_data = {
                "query": user_query,
                **context,
                **enriched_data
            }
            
            # Execute capability using MCP pattern
            result = await self.execute_capability(capability, combined_data, context)
            
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_performance_stats(execution_time_ms, success=True)
            
            # Format for orchestrator
            response = self._format_orchestrator_response(result, user_query, execution_time_ms)
            self.logger.info(f"✅ Query completed successfully in {execution_time_ms}ms")
            
            return response
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_performance_stats(execution_time_ms, success=False)
            
            self.logger.error(f"❌ Query failed after {execution_time_ms}ms: {str(e)}")
            return self._format_error_response(e, user_query)
    
    def _determine_capability_from_query(self, query: str) -> str:
        """
        🎯 Intelligent capability selection based on query content.
        Agents can override this for custom logic.
        """
        query_lower = query.lower()
        
        # Direct capability name matches (most specific)
        for capability in self.capabilities:
            capability_words = capability.replace("_", " ").split()
            if all(word in query_lower for word in capability_words):
                return capability
        
        # Enhanced keyword-based matching
        capability_keywords = {
            # Risk & Analysis
            "risk_analysis": ["risk", "analysis", "assess", "evaluate", "danger", "exposure"],
            "portfolio_analysis": ["portfolio", "analyze", "review", "overall", "comprehensive"],
            "var_analysis": ["var", "value at risk", "downside", "worst case"],
            "stress_testing": ["stress", "test", "scenario", "crash", "crisis", "shock"],
            "correlation_analysis": ["correlation", "relationship", "diversification", "dependency"],
            
            # Investment & Strategy
            "security_screening": ["screen", "find", "recommend", "stocks", "securities", "buy"],
            "factor_analysis": ["factor", "quality", "value", "growth", "momentum"],
            "strategy_development": ["strategy", "approach", "plan", "systematic"],
            "backtesting": ["backtest", "test strategy", "historical", "validate"],
            
            # Portfolio Management  
            "optimization": ["optimize", "rebalance", "allocation", "weight", "efficient"],
            "rebalancing": ["rebalance", "reweight", "adjust", "balance"],
            "hedging": ["hedge", "protect", "volatility", "insurance", "cover"],
            
            # Market Analysis
            "regime_detection": ["regime", "forecast", "transition", "change", "shift"],
            "market_analysis": ["market", "conditions", "environment", "outlook"],
            "scenario_simulation": ["scenario", "simulation", "what if", "simulate"],
            
            # Specialized
            "tax_optimization": ["tax", "harvest", "optimization", "efficiency"],
            "behavioral_analysis": ["bias", "behavior", "psychology", "sentiment"],
            "economic_analysis": ["economic", "fed", "inflation", "gdp", "rates"]
        }
        
        # Score each capability based on keyword matches
        capability_scores = {}
        for capability, keywords in capability_keywords.items():
            if capability in self.capabilities:
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    capability_scores[capability] = score
        
        # Return highest scoring capability
        if capability_scores:
            best_capability = max(capability_scores.items(), key=lambda x: x[1])[0]
            return best_capability
        
        # Fallback: return first capability or general analysis
        return self.capabilities[0] if self.capabilities else "general_analysis"
    
    def _format_orchestrator_response(self, result: Dict, query: str, execution_time_ms: int) -> Dict[str, Any]:
        """📋 Format MCP result for orchestrator consumption"""
        
        # Extract confidence score
        confidence = result.get("confidence_score", result.get("confidence", 0.8))
        
        # Generate user-friendly summary
        summary = self._generate_summary(result, query)
        
        # Extract recommendations
        recommendations = self._extract_recommendations(result)
        
        return {
            "success": True,
            "summary": summary,
            "agent_used": self.agent_name,
            "confidence": float(min(max(confidence, 0.0), 1.0)),  # Clamp to [0,1]
            "data": result,
            "recommendations": recommendations,
            "execution_time_ms": execution_time_ms,
            "capabilities_used": [self._determine_capability_from_query(query)],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _format_error_response(self, error: Exception, query: str) -> Dict[str, Any]:
        """🚨 Standardized error response format with user-friendly messages"""
        
        error_str = str(error).lower()
        
        # Categorize errors for user-friendly messages
        if "portfolio" in error_str and ("missing" in error_str or "not found" in error_str or "required" in error_str):
            user_message = "This analysis requires portfolio data. Please upload your portfolio first."
            error_category = "missing_portfolio"
        elif "data" in error_str and ("fetch" in error_str or "retrieve" in error_str):
            user_message = "Unable to retrieve required market data. Please try again."
            error_category = "data_fetch_error"
        elif "timeout" in error_str:
            user_message = "Analysis timed out. Please try a simpler query."
            error_category = "timeout"
        elif "capability" in error_str and ("not supported" in error_str or "unknown" in error_str):
            available_caps = ", ".join(self.capabilities[:3])
            user_message = f"I cannot handle that type of request. I can help with: {available_caps}..."
            error_category = "unsupported_capability"
        elif "network" in error_str or "connection" in error_str:
            user_message = "Network issue encountered. Please try again."
            error_category = "network_error"
        else:
            user_message = "Analysis failed due to a technical issue. Please try again or rephrase your request."
            error_category = "general_error"
        
        return {
            "success": False,
            "error": user_message,
            "error_category": error_category,
            "agent_used": self.agent_name,
            "confidence": 0.0,
            "technical_error": str(error),  # For debugging
            "capabilities": self.capabilities,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _generate_summary(self, result: Dict, query: str) -> str:
        """
        📝 Generate user-friendly summary from MCP result.
        Agents can override this for custom formatting.
        """
        # Check if result already has a summary
        if "summary" in result and result["summary"]:
            return result["summary"]
        
        # Generate basic summary based on analysis type
        analysis_type = result.get("analysis_type", "analysis")
        confidence = result.get("confidence_score", result.get("confidence", 0.8))
        
        # Create formatted summary
        agent_type = self.agent_name.replace("Agent", "").replace("_", " ")
        
        summary = f"### ✅ {agent_type} Analysis Complete\n\n"
        
        # Add analysis-specific details
        if "risk" in analysis_type.lower():
            risk_level = result.get("risk_level", "moderate")
            summary += f"**Risk Level**: {risk_level}\n"
        elif "screening" in analysis_type.lower():
            recommendations_count = len(result.get("recommendations", []))
            summary += f"**Securities Found**: {recommendations_count}\n"
        
        summary += f"**Confidence**: {confidence:.0%}\n"
        summary += f"**Analysis Type**: {analysis_type.replace('_', ' ').title()}"
        
        return summary
    
    def _extract_recommendations(self, result: Dict) -> List[str]:
        """📌 Extract actionable recommendations from result"""
        recommendations = result.get("recommendations", [])
        
        # Handle different recommendation formats
        if isinstance(recommendations, list):
            return [str(rec) for rec in recommendations[:5]]  # Limit to 5
        elif isinstance(recommendations, dict):
            return [f"{key}: {value}" for key, value in list(recommendations.items())[:5]]
        elif isinstance(recommendations, str):
            return [recommendations]
        
        # Generate default recommendations if none provided
        analysis_type = result.get("analysis_type", "")
        if "risk" in analysis_type.lower():
            return ["Monitor portfolio risk metrics regularly", "Consider diversification if concentration is high"]
        elif "screening" in analysis_type.lower():
            return ["Review recommended securities for fit with your strategy"]
        
        return []
    
    def _update_performance_stats(self, execution_time_ms: int, success: bool):
        """📊 Update performance statistics"""
        if success:
            self.execution_stats["successful_requests"] += 1
        else:
            self.execution_stats["failed_requests"] += 1
        
        # Update average response time
        total_requests = self.execution_stats["total_requests"]
        current_avg = self.execution_stats["avg_response_time_ms"]
        self.execution_stats["avg_response_time_ms"] = (
            (current_avg * (total_requests - 1) + execution_time_ms) / total_requests
        )
    
    # ==========================================
    # ENHANCED HEALTH CHECK SYSTEM
    # ==========================================
    
    async def health_check(self) -> Dict[str, Any]:
        """🏥 Enhanced health check with capability validation"""
        
        health_status = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "capabilities": self.capabilities,
            "performance_stats": self.execution_stats.copy(),
            "mcp_compatible": True,
            "orchestrator_compatible": True
        }
        
        # Test capability execution
        capability_health = {}
        test_timeout = 3.0  # 3 second timeout per capability
        
        for capability in self.capabilities[:3]:  # Test first 3 capabilities
            try:
                test_result = await asyncio.wait_for(
                    self._test_capability(capability), 
                    timeout=test_timeout
                )
                capability_health[capability] = "healthy" if test_result else "degraded"
            except asyncio.TimeoutError:
                capability_health[capability] = "timeout"
            except Exception as e:
                capability_health[capability] = f"error: {str(e)[:50]}"
        
        health_status["capability_health"] = capability_health
        
        # Determine overall status
        failed_capabilities = sum(1 for status in capability_health.values() 
                                if status not in ["healthy", "degraded"])
        
        if failed_capabilities == 0:
            health_status["status"] = "healthy"
        elif failed_capabilities < len(capability_health) / 2:
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "unhealthy"
        
        # Add response time assessment
        avg_response = self.execution_stats["avg_response_time_ms"]
        if avg_response > 30000:  # 30+ seconds
            health_status["response_time_status"] = "slow"
        elif avg_response > 10000:  # 10+ seconds  
            health_status["response_time_status"] = "acceptable"
        else:
            health_status["response_time_status"] = "fast"
        
        return health_status
    
    async def _test_capability(self, capability: str) -> bool:
        """🧪 Test capability with minimal input - agents can override"""
        try:
            # Default: just check if execute_capability can be called
            test_data = {"query": "health check", "test": True}
            test_context = {"health_check": True}
            
            # This should not actually execute full logic during health check
            result = await self._health_check_capability(capability, test_data, test_context)
            return isinstance(result, dict)
        except Exception as e:
            self.logger.warning(f"Capability {capability} health check failed: {str(e)}")
            return False
    
    async def _health_check_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """🔬 Lightweight capability check - agents should override this"""
        # Default implementation - agents should override for actual testing
        await asyncio.sleep(0.1)  # Simulate minimal work
        return {
            "status": "health_check_passed",
            "capability": capability,
            "test_mode": True
        }
    
    # ==========================================
    # ENHANCED DATA FETCHING (FIXED TIMEOUT ISSUE)
    # ==========================================
    
    async def autonomous_data_fetch(self, query: str, context: Dict) -> Dict:
        """🤖 Enhanced autonomous data fetching with better error handling"""
        enriched_data = {}
        query_lower = query.lower()
        
        fetch_tasks = []
        
        # Portfolio data
        if (("portfolio" in query_lower or "holdings" in query_lower) and 
            not context.get("portfolio_data") and not context.get("holdings_with_values")):
            fetch_tasks.append(("portfolio", self._safe_fetch_portfolio_data(query, context)))
        
        # Market data
        if (("market" in query_lower or "price" in query_lower or "stock" in query_lower) and 
            not context.get("market_data")):
            fetch_tasks.append(("market", self._safe_fetch_market_data(query, context)))
        
        # Economic data
        if (("economic" in query_lower or "fed" in query_lower or "interest" in query_lower) and 
            not context.get("economic_data")):
            fetch_tasks.append(("economic", self._safe_fetch_economic_data(query, context)))
        
        # News/sentiment data
        if (("news" in query_lower or "sentiment" in query_lower) and 
            not context.get("news_data")):
            fetch_tasks.append(("news", self._safe_fetch_news_data(query, context)))
        
        # Execute fetch tasks concurrently with timeout - FIXED VERSION
        if fetch_tasks:
            try:
                # FIXED: Create tasks explicitly, then use asyncio.wait()
                created_tasks = [asyncio.create_task(task) for _, task in fetch_tasks]
                
                done, pending = await asyncio.wait(
                    created_tasks,
                    timeout=5.0,  # 5 second total timeout
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Cancel any pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Process completed tasks
                for i, task in enumerate(created_tasks):
                    if task in done:
                        try:
                            result = task.result()
                            if result and not isinstance(result, Exception):
                                data_type = fetch_tasks[i][0]
                                enriched_data[f"{data_type}_data"] = result
                        except Exception as e:
                            data_type = fetch_tasks[i][0]
                            self.logger.warning(f"Failed to fetch {data_type} data: {str(e)}")
                
                if pending:
                    self.logger.warning("Some data fetching operations timed out")
                        
            except Exception as e:
                self.logger.warning(f"Error in concurrent data fetching: {str(e)}")
        
        if enriched_data:
            self.logger.info(f"🔍 Autonomously enriched data: {list(enriched_data.keys())}")
        
        return enriched_data
    
    async def _safe_fetch_portfolio_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Safe portfolio data fetching with error handling"""
        try:
            return await self.fetch_portfolio_data(query, context)
        except Exception as e:
            self.logger.warning(f"Portfolio data fetch failed: {str(e)}")
            return None
    
    async def _safe_fetch_market_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Safe market data fetching with error handling"""
        try:
            return await self.fetch_market_data(query, context)
        except Exception as e:
            self.logger.warning(f"Market data fetch failed: {str(e)}")
            return None
    
    async def _safe_fetch_economic_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Safe economic data fetching with error handling"""
        try:
            return await self.fetch_economic_data(query, context)
        except Exception as e:
            self.logger.warning(f"Economic data fetch failed: {str(e)}")
            return None
    
    async def _safe_fetch_news_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Safe news data fetching with error handling"""
        try:
            return await self.fetch_news_data(query, context)
        except Exception as e:
            self.logger.warning(f"News data fetch failed: {str(e)}")
            return None
    
    # ==========================================
    # EXISTING MCP SYSTEM METHODS (ENHANCED)
    # ==========================================
    
    async def start(self):
        """Start the agent and optionally register with MCP"""
        if self.enable_mcp_registration and MCP_AVAILABLE:
            self.session = aiohttp.ClientSession()
            await self.register_with_mcp()
            await self.start_job_listener()
        else:
            self.logger.info(f"🔄 Agent {self.agent_id} started in standalone mode")
    
    async def stop(self):
        """Stop the agent and cleanup resources"""
        if self.session:
            await self.session.close()
        
        # Cancel all active jobs
        for job_id, task in self.active_jobs.items():
            task.cancel()
            self.logger.info(f"Cancelled job {job_id}")
        
        if self.is_registered:
            await self.unregister_from_mcp()
    
    async def register_with_mcp(self):
        """Register this agent with the Master Control Plane"""
        if not MCP_AVAILABLE:
            self.logger.warning("Cannot register with MCP - schemas not available")
            return
            
        registration = AgentRegistration(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            agent_type=self.__class__.__name__,
            capabilities=self.capabilities,
            endpoint_url=f"http://localhost:{self.get_port()}/agent",
            max_concurrent_jobs=self.max_concurrent_jobs,
            response_time_sla=self.get_response_time_sla(),
            metadata=self.get_agent_metadata()
        )
        
        try:
            async with self.session.post(
                f"{self.mcp_url}/register",
                json=registration.dict()
            ) as response:
                if response.status == 200:
                    self.is_registered = True
                    result = await response.json()
                    self.logger.info(f"✅ Agent {self.agent_id} registered successfully")
                else:
                    error_text = await response.text()
                    self.logger.error(f"❌ Failed to register agent {self.agent_id}: {error_text}")
                    
        except Exception as e:
            self.logger.error(f"❌ Error registering agent {self.agent_id}: {str(e)}")
    
    async def unregister_from_mcp(self):
        """Unregister this agent from the MCP"""
        if not self.is_registered or not MCP_AVAILABLE:
            return
            
        try:
            async with self.session.delete(
                f"{self.mcp_url}/agents/{self.agent_id}"
            ) as response:
                if response.status == 200:
                    self.is_registered = False
                    self.logger.info(f"✅ Agent {self.agent_id} unregistered successfully")
                    
        except Exception as e:
            self.logger.error(f"❌ Error unregistering agent {self.agent_id}: {str(e)}")
    
    async def start_job_listener(self):
        """Start listening for job requests from MCP"""
        self.logger.info(f"🎧 Agent {self.agent_id} started listening for jobs")
        
        while self.is_registered:
            try:
                await self.check_for_new_jobs()
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                self.logger.error(f"Error in job listener for {self.agent_id}: {str(e)}")
                await asyncio.sleep(5)  # Back off on error
    
    async def check_for_new_jobs(self):
        """Check for new jobs assigned to this agent"""
        # This is a simplified polling mechanism
        # In production, use webhooks, message queues, or WebSocket connections
        pass  # MCP will call handle_job_request directly
    
    async def handle_job_request(self, job_request) -> Any:
        """Handle a job request from the MCP"""
        if not MCP_AVAILABLE:
            raise RuntimeError("MCP job request received but MCP schemas not available")
            
        start_time = time.time()
        step_id = job_request.step_id
        
        try:
            self.logger.info(f"🔄 Processing job step {step_id} with capability {job_request.capability}")
            
            # Check if we can handle this capability
            if job_request.capability not in self.capabilities:
                return AgentJobResponse(
                    step_id=step_id,
                    status=JobStatus.FAILED,
                    error_message=f"Agent {self.agent_id} does not support capability {job_request.capability}"
                )
            
            # Fetch any missing data autonomously
            enriched_data = await self.autonomous_data_fetch(
                job_request.input_data.get("query", ""), 
                job_request.context
            )
            
            # Merge enriched data with input data
            combined_data = {**job_request.input_data, **enriched_data}
            
            # Execute the specific capability
            result = await self.execute_capability(
                job_request.capability, 
                combined_data, 
                job_request.context
            )
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            return AgentJobResponse(
                step_id=step_id,
                status=JobStatus.COMPLETED,
                result=result,
                confidence_score=self.calculate_confidence_score(result),
                execution_time_ms=execution_time_ms,
                metadata={
                    "agent_id": self.agent_id,
                    "capability_version": self.get_capability_version(job_request.capability),
                    "data_sources_used": self.get_data_sources_used(combined_data)
                }
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self.logger.error(f"❌ Error processing step {step_id}: {str(e)}")
            
            return AgentJobResponse(
                step_id=step_id,
                status=JobStatus.FAILED,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                metadata={"agent_id": self.agent_id}
            )
    
    # ==========================================
    # EXISTING ABSTRACT AND UTILITY METHODS
    # ==========================================
    
    @abstractmethod
    async def execute_capability(self, capability: str, data: Dict, context: Dict) -> Dict:
        """Execute a specific capability - must be implemented by subclasses"""
        pass
    
    # Keep all your existing methods (fetch_portfolio_data, etc.) - they're good as-is
    async def fetch_portfolio_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch portfolio data from the portfolio service"""
        portfolio_id = context.get("portfolio_id")
        if not portfolio_id:
            return None
        
        try:
            # Mock portfolio data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "portfolio_id": portfolio_id,
                "holdings": [
                    {"symbol": "AAPL", "shares": 100, "current_price": 150.25},
                    {"symbol": "GOOGL", "shares": 50, "current_price": 2800.50},
                    {"symbol": "TSLA", "shares": 25, "current_price": 900.75}
                ],
                "total_value": 305518.75,
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching portfolio data: {str(e)}")
            return None
    
    async def fetch_market_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch real-time market data"""
        try:
            # Mock market data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "market_indices": {
                    "SPY": {"price": 445.20, "change": 2.15, "change_pct": 0.48},
                    "QQQ": {"price": 375.80, "change": -1.25, "change_pct": -0.33},
                    "IWM": {"price": 195.45, "change": 0.85, "change_pct": 0.44}
                },
                "volatility": {
                    "VIX": 18.25
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return None
    
    async def fetch_economic_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch economic indicators and data"""
        try:
            # Mock economic data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "indicators": {
                    "fed_funds_rate": 5.25,
                    "inflation_rate": 3.2,
                    "unemployment_rate": 3.7,
                    "gdp_growth": 2.1
                },
                "treasury_yields": {
                    "1Y": 4.95,
                    "10Y": 4.25,
                    "30Y": 4.35
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error fetching economic data: {str(e)}")
            return None
    
    async def fetch_news_data(self, query: str, context: Dict) -> Optional[Dict]:
        """Fetch relevant news and sentiment data"""
        try:
            # Mock news data fetch - replace with actual API call
            await asyncio.sleep(0.1)  # Simulate API delay
            return {
                "headlines": [
                    {
                        "title": "Fed Signals Potential Rate Cut in Q2",
                        "sentiment": "positive",
                        "relevance": 0.85,
                        "timestamp": "2024-01-15T10:30:00Z"
                    },
                    {
                        "title": "Tech Earnings Beat Expectations",
                        "sentiment": "positive", 
                        "relevance": 0.92,
                        "timestamp": "2024-01-15T09:15:00Z"
                    }
                ],
                "overall_sentiment": 0.72,
                "market_sentiment": "optimistic"
            }
        except Exception as e:
            self.logger.error(f"Error fetching news data: {str(e)}")
            return None
    
    def calculate_confidence_score(self, result: Dict) -> float:
        """Calculate confidence score for the result"""
        # Base implementation - can be overridden by subclasses
        if not result:
            return 0.0
        
        # Simple confidence calculation based on data completeness
        data_completeness = len([v for v in result.values() if v is not None]) / len(result) if result else 0
        return min(0.95, 0.7 + (data_completeness * 0.25))
    
    def get_capability_version(self, capability: str) -> str:
        """Get version of the capability implementation"""
        return "1.0.0"  # Override in subclasses
    
    def get_data_sources_used(self, data: Dict) -> List[str]:
        """Get list of data sources used in the analysis"""
        sources = []
        if "portfolio_data" in data:
            sources.append("portfolio_service")
        if "market_data" in data:
            sources.append("market_data_api")
        if "economic_data" in data:
            sources.append("economic_indicators_api")
        if "news_data" in data:
            sources.append("news_sentiment_api")
        return sources
    
    def get_port(self) -> int:
        """Get the port this agent should listen on"""
        # Generate port based on agent_id hash for consistency
        return 8000 + (hash(self.agent_id) % 1000)
    
    def get_response_time_sla(self) -> int:
        """Get the response time SLA for this agent in seconds"""
        return 30  # Override in subclasses if needed
    
    def get_agent_metadata(self) -> Dict[str, Any]:
        """Get additional metadata about this agent"""
        return {
            "version": "1.0.0",
            "startup_time": datetime.utcnow().isoformat(),
            "supported_data_sources": self.get_supported_data_sources(),
            "mcp_compatible": MCP_AVAILABLE,
            "orchestrator_compatible": True
        }
    
    def get_supported_data_sources(self) -> List[str]:
        """Get list of data sources this agent can work with"""
        return ["portfolio_service", "market_data_api"]  # Override in subclasses