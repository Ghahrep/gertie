# smart_suggestions/ab_testing_service.py
"""
A/B Testing and Analytics Framework for Smart Suggestions
========================================================
Provides statistical testing, performance analytics, and continuous
improvement capabilities for the suggestion system.
"""

import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from collections import defaultdict, deque

from sqlalchemy.orm import Session
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind

from db import crud, models
from smart_suggestions.suggestion_engine import SmartSuggestion

class TestStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"

class TestType(Enum):
    SUGGESTION_VARIANT = "suggestion_variant"
    RANKING_ALGORITHM = "ranking_algorithm"
    ML_MODEL = "ml_model"
    UI_PRESENTATION = "ui_presentation"

@dataclass
class TestVariant:
    """A/B test variant definition"""
    variant_id: str
    name: str
    description: str
    traffic_percentage: float
    config: Dict[str, Any] = field(default_factory=dict)
    active: bool = True

@dataclass
class TestMetrics:
    """A/B test performance metrics"""
    impressions: int = 0
    clicks: int = 0
    executions: int = 0
    user_ratings: List[float] = field(default_factory=list)
    engagement_time: List[float] = field(default_factory=list)
    conversion_events: int = 0
    
    @property
    def click_through_rate(self) -> float:
        return self.clicks / max(self.impressions, 1)
    
    @property
    def execution_rate(self) -> float:
        return self.executions / max(self.clicks, 1)
    
    @property
    def avg_rating(self) -> float:
        return np.mean(self.user_ratings) if self.user_ratings else 0.0
    
    @property
    def avg_engagement_time(self) -> float:
        return np.mean(self.engagement_time) if self.engagement_time else 0.0

@dataclass
class ABTest:
    """A/B test configuration and state"""
    test_id: str
    name: str
    description: str
    test_type: TestType
    variants: List[TestVariant]
    status: TestStatus
    start_date: datetime
    end_date: Optional[datetime]
    minimum_sample_size: int
    confidence_level: float = 0.95
    created_by: str = "system"
    
    # Metrics tracking
    variant_metrics: Dict[str, TestMetrics] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize variant metrics"""
        for variant in self.variants:
            if variant.variant_id not in self.variant_metrics:
                self.variant_metrics[variant.variant_id] = TestMetrics()

@dataclass
class StatisticalResult:
    """Statistical significance test result"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Optional[Tuple[float, float]] = None
    effect_size: Optional[float] = None
    interpretation: str = ""

class ABTestingService:
    """
    A/B Testing service for smart suggestions system
    Provides statistical testing, performance tracking, and optimization
    """
    
    def __init__(self):
        self.active_tests: Dict[str, ABTest] = {}
        self.completed_tests: Dict[str, ABTest] = {}
        self.user_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)  # user_id -> test_id -> variant_id
        
        # Analytics tracking
        self.event_buffer: deque = deque(maxlen=10000)  # Buffer for analytics events
        self.daily_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Background processing
        self.background_tasks = set()
    
    async def start_service(self):
        """Start the A/B testing service"""
        # Start background analytics processing
        task = asyncio.create_task(self._process_analytics_events())
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        
        print("A/B Testing and Analytics service started")
        print("- Statistical testing: ACTIVE")
        print("- Performance tracking: ACTIVE")
        print("- Event processing: ACTIVE")
    
    def create_suggestion_variant_test(
        self,
        test_name: str,
        description: str,
        variants: List[Dict[str, Any]],
        duration_days: int = 14,
        minimum_sample_size: int = 100,
        created_by: str = "system"
    ) -> ABTest:
        """Create a new A/B test for suggestion variants"""
        
        test_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create test variants
        test_variants = []
        total_traffic = 100
        variant_traffic = total_traffic / len(variants)
        
        for i, variant_config in enumerate(variants):
            variant = TestVariant(
                variant_id=f"variant_{chr(65 + i)}",  # A, B, C, etc.
                name=variant_config.get("name", f"Variant {chr(65 + i)}"),
                description=variant_config.get("description", ""),
                traffic_percentage=variant_traffic,
                config=variant_config.get("config", {})
            )
            test_variants.append(variant)
        
        # Create test
        test = ABTest(
            test_id=test_id,
            name=test_name,
            description=description,
            test_type=TestType.SUGGESTION_VARIANT,
            variants=test_variants,
            status=TestStatus.ACTIVE,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=duration_days),
            minimum_sample_size=minimum_sample_size,
            created_by=created_by
        )
        
        self.active_tests[test_id] = test
        
        print(f"Created A/B test: {test_name} ({test_id})")
        print(f"- Variants: {len(test_variants)}")
        print(f"- Duration: {duration_days} days")
        print(f"- Sample size target: {minimum_sample_size}")
        
        return test
    
    def assign_user_to_variant(self, test_id: str, user_id: str) -> Optional[str]:
        """Assign user to test variant using consistent hashing"""
        
        if test_id not in self.active_tests:
            return None
        
        test = self.active_tests[test_id]
        
        # Check if user already assigned
        if test_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][test_id]
        
        # Use consistent hashing for assignment
        hash_input = f"{user_id}_{test_id}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        percentage = (hash_value % 100) + 1
        
        # Assign to variant based on traffic allocation
        cumulative_percentage = 0
        for variant in test.variants:
            if not variant.active:
                continue
                
            cumulative_percentage += variant.traffic_percentage
            if percentage <= cumulative_percentage:
                self.user_assignments[user_id][test_id] = variant.variant_id
                return variant.variant_id
        
        # Fallback to first variant
        first_variant = next((v for v in test.variants if v.active), test.variants[0])
        self.user_assignments[user_id][test_id] = first_variant.variant_id
        return first_variant.variant_id
    
    def get_suggestion_variant(
        self, 
        suggestion: SmartSuggestion, 
        test_id: str, 
        user_id: str
    ) -> SmartSuggestion:
        """Get suggestion variant based on A/B test assignment"""
        
        variant_id = self.assign_user_to_variant(test_id, user_id)
        if not variant_id or test_id not in self.active_tests:
            return suggestion
        
        test = self.active_tests[test_id]
        variant = next((v for v in test.variants if v.variant_id == variant_id), None)
        
        if not variant:
            return suggestion
        
        # Apply variant configuration
        modified_suggestion = self._apply_variant_config(suggestion, variant.config)
        
        # Track impression
        self._track_event("impression", test_id, variant_id, user_id, {
            "suggestion_id": suggestion.id,
            "suggestion_category": suggestion.category
        })
        
        return modified_suggestion
    
    def _apply_variant_config(
        self, 
        suggestion: SmartSuggestion, 
        config: Dict[str, Any]
    ) -> SmartSuggestion:
        """Apply variant configuration to suggestion"""
        
        # Create a copy to avoid modifying the original
        import copy
        modified_suggestion = copy.deepcopy(suggestion)
        
        # Apply title modifications
        if "title_prefix" in config:
            modified_suggestion.title = f"{config['title_prefix']} {modified_suggestion.title}"
        
        if "title_suffix" in config:
            modified_suggestion.title = f"{modified_suggestion.title} {config['title_suffix']}"
        
        # Apply description modifications
        if "description_template" in config:
            template = config["description_template"]
            modified_suggestion.description = template.format(
                original_description=modified_suggestion.description,
                category=modified_suggestion.category
            )
        
        # Apply confidence adjustments
        if "confidence_multiplier" in config:
            multiplier = config["confidence_multiplier"]
            modified_suggestion.confidence = min(1.0, modified_suggestion.confidence * multiplier)
        
        # Apply urgency modifications
        if "urgency_override" in config:
            from smart_suggestions.suggestion_engine import Urgency
            urgency_map = {
                "low": Urgency.LOW,
                "medium": Urgency.MEDIUM,
                "high": Urgency.HIGH
            }
            new_urgency = config["urgency_override"].lower()
            if new_urgency in urgency_map:
                modified_suggestion.urgency = urgency_map[new_urgency]
        
        # Apply visual modifications
        if "icon_override" in config:
            modified_suggestion.icon = config["icon_override"]
        
        if "color_override" in config:
            modified_suggestion.color = config["color_override"]
        
        # Add variant metadata
        if modified_suggestion.metadata is None:
            modified_suggestion.metadata = {}
        
        modified_suggestion.metadata["ab_test_variant"] = config.get("variant_name", "unknown")
        modified_suggestion.metadata["ab_test_config"] = config
        
        return modified_suggestion
    
    def track_suggestion_click(self, test_id: str, user_id: str, suggestion_id: str):
        """Track when user clicks on a suggestion"""
        
        variant_id = self.user_assignments.get(user_id, {}).get(test_id)
        if variant_id and test_id in self.active_tests:
            self._track_event("click", test_id, variant_id, user_id, {
                "suggestion_id": suggestion_id
            })
    
    def track_suggestion_execution(
        self, 
        test_id: str, 
        user_id: str, 
        suggestion_id: str,
        execution_metadata: Dict[str, Any] = None
    ):
        """Track when user executes a suggestion"""
        
        variant_id = self.user_assignments.get(user_id, {}).get(test_id)
        if variant_id and test_id in self.active_tests:
            self._track_event("execution", test_id, variant_id, user_id, {
                "suggestion_id": suggestion_id,
                "execution_metadata": execution_metadata or {}
            })
    
    def track_user_rating(
        self, 
        test_id: str, 
        user_id: str, 
        suggestion_id: str, 
        rating: float
    ):
        """Track user rating for a suggestion"""
        
        variant_id = self.user_assignments.get(user_id, {}).get(test_id)
        if variant_id and test_id in self.active_tests:
            self._track_event("rating", test_id, variant_id, user_id, {
                "suggestion_id": suggestion_id,
                "rating": rating
            })
    
    def _track_event(
        self, 
        event_type: str, 
        test_id: str, 
        variant_id: str, 
        user_id: str, 
        metadata: Dict[str, Any] = None
    ):
        """Track an analytics event"""
        
        event = {
            "event_type": event_type,
            "test_id": test_id,
            "variant_id": variant_id,
            "user_id": user_id,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        self.event_buffer.append(event)
        
        # Update real-time metrics
        if test_id in self.active_tests:
            test = self.active_tests[test_id]
            variant_metrics = test.variant_metrics[variant_id]
            
            if event_type == "impression":
                variant_metrics.impressions += 1
            elif event_type == "click":
                variant_metrics.clicks += 1
            elif event_type == "execution":
                variant_metrics.executions += 1
                variant_metrics.conversion_events += 1
            elif event_type == "rating":
                rating = metadata.get("rating", 0)
                variant_metrics.user_ratings.append(rating)
    
    async def _process_analytics_events(self):
        """Background task to process analytics events"""
        
        while True:
            try:
                await asyncio.sleep(10)  # Process every 10 seconds
                
                # Process buffered events
                events_to_process = []
                while self.event_buffer and len(events_to_process) < 100:
                    events_to_process.append(self.event_buffer.popleft())
                
                if events_to_process:
                    await self._batch_process_events(events_to_process)
                
                # Check for test completion
                await self._check_test_completion()
                
            except Exception as e:
                print(f"Error processing analytics events: {e}")
    
    async def _batch_process_events(self, events: List[Dict[str, Any]]):
        """Process a batch of analytics events"""
        
        # Group events by test and variant
        event_groups = defaultdict(lambda: defaultdict(list))
        
        for event in events:
            test_id = event["test_id"]
            variant_id = event["variant_id"]
            event_groups[test_id][variant_id].append(event)
        
        # Update metrics for each test/variant combination
        for test_id, variant_events in event_groups.items():
            if test_id not in self.active_tests:
                continue
                
            test = self.active_tests[test_id]
            
            for variant_id, events in variant_events.items():
                if variant_id not in test.variant_metrics:
                    continue
                
                metrics = test.variant_metrics[variant_id]
                
                for event in events:
                    event_type = event["event_type"]
                    metadata = event.get("metadata", {})
                    
                    if event_type == "engagement_time":
                        engagement_time = metadata.get("engagement_time", 0)
                        metrics.engagement_time.append(engagement_time)
    
    async def _check_test_completion(self):
        """Check if any tests should be completed"""
        
        current_time = datetime.now()
        completed_tests = []
        
        for test_id, test in self.active_tests.items():
            # Check if test duration has ended
            if test.end_date and current_time >= test.end_date:
                completed_tests.append(test_id)
                continue
            
            # Check if minimum sample size reached for all variants
            min_sample_reached = True
            for variant_id, metrics in test.variant_metrics.items():
                if metrics.impressions < test.minimum_sample_size:
                    min_sample_reached = False
                    break
            
            if min_sample_reached:
                # Run statistical significance test
                significance_result = self._run_significance_test(test)
                if significance_result and significance_result.is_significant:
                    completed_tests.append(test_id)
        
        # Move completed tests
        for test_id in completed_tests:
            test = self.active_tests[test_id]
            test.status = TestStatus.COMPLETED
            test.end_date = current_time
            self.completed_tests[test_id] = test
            del self.active_tests[test_id]
            
            print(f"Test completed: {test.name} ({test_id})")
    
    def _run_significance_test(self, test: ABTest) -> Optional[StatisticalResult]:
        """Run statistical significance test for A/B test"""
        
        if len(test.variants) != 2:
            # For now, only handle two-variant tests
            return None
        
        variant_a = test.variants[0]
        variant_b = test.variants[1]
        metrics_a = test.variant_metrics[variant_a.variant_id]
        metrics_b = test.variant_metrics[variant_b.variant_id]
        
        # Test click-through rates
        if metrics_a.impressions > 0 and metrics_b.impressions > 0:
            # Chi-square test for conversion rates
            observed = [
                [metrics_a.clicks, metrics_a.impressions - metrics_a.clicks],
                [metrics_b.clicks, metrics_b.impressions - metrics_b.clicks]
            ]
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(observed)
                is_significant = p_value < (1 - test.confidence_level)
                
                # Calculate effect size (relative improvement)
                ctr_a = metrics_a.click_through_rate
                ctr_b = metrics_b.click_through_rate
                effect_size = (ctr_b - ctr_a) / ctr_a if ctr_a > 0 else 0
                
                interpretation = self._interpret_ab_test_result(
                    ctr_a, ctr_b, p_value, is_significant, effect_size
                )
                
                return StatisticalResult(
                    test_name="click_through_rate",
                    statistic=chi2,
                    p_value=p_value,
                    is_significant=is_significant,
                    effect_size=effect_size,
                    interpretation=interpretation
                )
                
            except ValueError as e:
                print(f"Error in significance test: {e}")
                return None
        
        return None
    
    def _interpret_ab_test_result(
        self, 
        ctr_a: float, 
        ctr_b: float, 
        p_value: float, 
        is_significant: bool, 
        effect_size: float
    ) -> str:
        """Interpret A/B test results"""
        
        if not is_significant:
            return f"No statistically significant difference found (p={p_value:.3f}). Continue testing or conclude with no clear winner."
        
        winner = "Variant B" if ctr_b > ctr_a else "Variant A"
        improvement = abs(effect_size) * 100
        
        if improvement < 5:
            significance_level = "marginal"
        elif improvement < 15:
            significance_level = "moderate"
        else:
            significance_level = "strong"
        
        return f"{winner} shows {significance_level} improvement of {improvement:.1f}% (p={p_value:.3f}). Recommend implementing winning variant."
    
    def get_test_analytics(self, test_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a test"""
        
        test = self.active_tests.get(test_id) or self.completed_tests.get(test_id)
        if not test:
            return {"error": "Test not found"}
        
        # Compile variant performance
        variant_performance = {}
        for variant in test.variants:
            metrics = test.variant_metrics[variant.variant_id]
            
            variant_performance[variant.variant_id] = {
                "name": variant.name,
                "traffic_percentage": variant.traffic_percentage,
                "impressions": metrics.impressions,
                "clicks": metrics.clicks,
                "executions": metrics.executions,
                "click_through_rate": metrics.click_through_rate,
                "execution_rate": metrics.execution_rate,
                "avg_rating": metrics.avg_rating,
                "avg_engagement_time": metrics.avg_engagement_time,
                "conversion_events": metrics.conversion_events,
                "sample_size": metrics.impressions
            }
        
        # Run current significance test
        significance_result = self._run_significance_test(test)
        
        # Calculate test progress
        max_sample_size = max(
            metrics.impressions for metrics in test.variant_metrics.values()
        )
        progress_percentage = min(100, (max_sample_size / test.minimum_sample_size) * 100)
        
        return {
            "test_id": test_id,
            "name": test.name,
            "description": test.description,
            "status": test.status.value,
            "test_type": test.test_type.value,
            "start_date": test.start_date.isoformat(),
            "end_date": test.end_date.isoformat() if test.end_date else None,
            "progress_percentage": progress_percentage,
            "days_running": (datetime.now() - test.start_date).days,
            "minimum_sample_size": test.minimum_sample_size,
            "confidence_level": test.confidence_level,
            "variant_performance": variant_performance,
            "statistical_significance": {
                "test_name": significance_result.test_name if significance_result else None,
                "p_value": significance_result.p_value if significance_result else None,
                "is_significant": significance_result.is_significant if significance_result else False,
                "effect_size": significance_result.effect_size if significance_result else None,
                "interpretation": significance_result.interpretation if significance_result else "Insufficient data for significance test"
            },
            "recommendations": self._generate_test_recommendations(test, significance_result)
        }
    
    def _generate_test_recommendations(
        self, 
        test: ABTest, 
        significance_result: Optional[StatisticalResult]
    ) -> List[str]:
        """Generate actionable recommendations based on test results"""
        
        recommendations = []
        
        # Check sample size
        max_sample_size = max(
            metrics.impressions for metrics in test.variant_metrics.values()
        )
        
        if max_sample_size < test.minimum_sample_size:
            remaining_samples = test.minimum_sample_size - max_sample_size
            recommendations.append(
                f"Continue test to reach minimum sample size. Need {remaining_samples} more samples."
            )
        
        # Statistical significance recommendations
        if significance_result:
            if significance_result.is_significant:
                recommendations.append(
                    f"Statistical significance achieved. {significance_result.interpretation}"
                )
                
                # Identify winning variant
                best_variant = None
                best_performance = 0
                
                for variant in test.variants:
                    metrics = test.variant_metrics[variant.variant_id]
                    performance_score = (
                        metrics.click_through_rate * 0.4 +
                        metrics.execution_rate * 0.4 +
                        metrics.avg_rating / 5.0 * 0.2
                    )
                    
                    if performance_score > best_performance:
                        best_performance = performance_score
                        best_variant = variant
                
                if best_variant:
                    recommendations.append(
                        f"Implement {best_variant.name} as the winning variant for production."
                    )
            else:
                recommendations.append(
                    "No significant difference detected. Consider running longer or testing more extreme variants."
                )
        
        # Performance-based recommendations
        variant_ctrs = [
            test.variant_metrics[v.variant_id].click_through_rate 
            for v in test.variants
        ]
        
        if max(variant_ctrs) < 0.1:  # All variants have low CTR
            recommendations.append(
                "Low click-through rates across all variants. Consider revising suggestion content or targeting."
            )
        
        # Engagement recommendations
        variant_ratings = [
            test.variant_metrics[v.variant_id].avg_rating 
            for v in test.variants if test.variant_metrics[v.variant_id].user_ratings
        ]
        
        if variant_ratings and max(variant_ratings) < 3.5:
            recommendations.append(
                "Low user ratings. Review suggestion quality and relevance."
            )
        
        return recommendations
    
    def get_global_analytics_dashboard(self) -> Dict[str, Any]:
        """Get system-wide analytics dashboard"""
        
        # Aggregate metrics across all tests
        total_tests = len(self.active_tests) + len(self.completed_tests)
        active_tests = len(self.active_tests)
        
        # Calculate overall performance metrics
        total_impressions = 0
        total_clicks = 0
        total_executions = 0
        all_ratings = []
        
        for test in {**self.active_tests, **self.completed_tests}.values():
            for metrics in test.variant_metrics.values():
                total_impressions += metrics.impressions
                total_clicks += metrics.clicks
                total_executions += metrics.executions
                all_ratings.extend(metrics.user_ratings)
        
        overall_ctr = total_clicks / max(total_impressions, 1)
        overall_execution_rate = total_executions / max(total_clicks, 1)
        overall_rating = np.mean(all_ratings) if all_ratings else 0
        
        # Recent performance trends (last 7 days)
        recent_events = [
            event for event in self.event_buffer
            if (datetime.now() - event["timestamp"]).days <= 7
        ]
        
        daily_metrics = defaultdict(lambda: {"impressions": 0, "clicks": 0, "executions": 0})
        for event in recent_events:
            date_key = event["timestamp"].date().isoformat()
            daily_metrics[date_key][event["event_type"]] += 1
        
        # Top performing test insights
        test_performance = []
        for test_id, test in {**self.active_tests, **self.completed_tests}.items():
            if not test.variant_metrics:
                continue
                
            best_variant_performance = 0
            for variant in test.variants:
                metrics = test.variant_metrics[variant.variant_id]
                performance = metrics.click_through_rate * metrics.avg_rating
                best_variant_performance = max(best_variant_performance, performance)
            
            test_performance.append({
                "test_id": test_id,
                "name": test.name,
                "performance_score": best_variant_performance,
                "status": test.status.value
            })
        
        top_tests = sorted(test_performance, key=lambda x: x["performance_score"], reverse=True)[:5]
        
        return {
            "summary": {
                "total_tests": total_tests,
                "active_tests": active_tests,
                "completed_tests": len(self.completed_tests),
                "total_impressions": total_impressions,
                "overall_ctr": overall_ctr,
                "overall_execution_rate": overall_execution_rate,
                "overall_rating": overall_rating,
                "unique_users_tested": len(self.user_assignments)
            },
            "recent_activity": {
                "events_last_7_days": len(recent_events),
                "daily_breakdown": dict(daily_metrics)
            },
            "top_performing_tests": top_tests,
            "system_health": {
                "event_buffer_size": len(self.event_buffer),
                "background_tasks_running": len(self.background_tasks),
                "last_analytics_update": datetime.now().isoformat()
            },
            "recommendations": self._generate_system_recommendations(
                overall_ctr, overall_execution_rate, overall_rating, active_tests
            )
        }
    
    def _generate_system_recommendations(
        self, 
        overall_ctr: float, 
        execution_rate: float, 
        avg_rating: float, 
        active_tests: int
    ) -> List[str]:
        """Generate system-wide optimization recommendations"""
        
        recommendations = []
        
        if overall_ctr < 0.15:
            recommendations.append(
                "Low overall click-through rate. Consider testing more engaging suggestion formats or improving targeting."
            )
        
        if execution_rate < 0.3:
            recommendations.append(
                "Low execution rate suggests suggestions may not be actionable enough. Test simpler, more direct suggestions."
            )
        
        if avg_rating < 3.5:
            recommendations.append(
                "User satisfaction is below target. Review suggestion relevance and quality."
            )
        
        if active_tests < 2:
            recommendations.append(
                "Consider running more A/B tests to continuously optimize suggestion performance."
            )
        
        if active_tests > 5:
            recommendations.append(
                "Many concurrent tests running. Ensure adequate traffic splitting and avoid test interference."
            )
        
        return recommendations

# Global service instance
_ab_testing_service = None

def get_ab_testing_service() -> ABTestingService:
    """Get the global A/B testing service instance"""
    global _ab_testing_service
    if _ab_testing_service is None:
        _ab_testing_service = ABTestingService()
    return _ab_testing_service

# Integration functions for existing smart suggestions system

def create_suggestion_optimization_test(
    test_name: str,
    control_config: Dict[str, Any],
    variant_configs: List[Dict[str, Any]],
    duration_days: int = 14
) -> ABTest:
    """Create an A/B test to optimize suggestion performance"""
    
    service = get_ab_testing_service()
    
    # Combine control and variants
    all_variants = [{"name": "Control", "config": control_config}]
    all_variants.extend(variant_configs)
    
    return service.create_suggestion_variant_test(
        test_name=test_name,
        description=f"Optimization test comparing {len(all_variants)} suggestion variants",
        variants=all_variants,
        duration_days=duration_days
    )

def apply_ab_testing_to_suggestions(
    suggestions: List[SmartSuggestion],
    user_id: str,
    active_test_ids: List[str]
) -> List[SmartSuggestion]:
    """Apply A/B testing variants to suggestions"""
    
    service = get_ab_testing_service()
    enhanced_suggestions = []
    
    for suggestion in suggestions:
        modified_suggestion = suggestion
        
        # Apply A/B test variants
        for test_id in active_test_ids:
            modified_suggestion = service.get_suggestion_variant(
                modified_suggestion, test_id, user_id
            )
        
        enhanced_suggestions.append(modified_suggestion)
    
    return enhanced_suggestions