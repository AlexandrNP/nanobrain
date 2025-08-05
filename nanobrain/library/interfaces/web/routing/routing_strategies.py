#!/usr/bin/env python3
"""
Routing Strategies for NanoBrain Framework
Configurable algorithms for intelligent workflow selection and routing decisions.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import math

from nanobrain.core.component_base import FromConfigBase
from nanobrain.core.config.config_base import ConfigBase
from nanobrain.library.interfaces.web.models.universal_models import (
    RequestAnalysis, WorkflowMatch, RoutingStrategy
)

# Strategies logger
logger = logging.getLogger(__name__)


class RoutingStrategyConfig(ConfigBase):
    """Configuration for routing strategies"""
    
    def __init__(self, config_data: Dict[str, Any]):
        super().__init__(config_data)
        
        # Strategy selection configuration
        self.strategy_type: str = config_data.get('strategy_type', 'adaptive')
        self.fallback_strategy: str = config_data.get('fallback_strategy', 'best_match')
        
        # Scoring weights for different factors
        self.scoring_weights: Dict[str, float] = config_data.get('scoring_weights', {
            'domain_match': 0.4,
            'intent_match': 0.3,
            'confidence_score': 0.2,
            'historical_performance': 0.1
        })
        
        # Threshold configurations
        self.thresholds: Dict[str, float] = config_data.get('thresholds', {
            'minimum_score': 0.3,
            'high_confidence': 0.8,
            'multi_candidate': 0.7,
            'fallback_trigger': 0.2
        })
        
        # Advanced strategy configurations
        self.advanced_config: Dict[str, Any] = config_data.get('advanced_config', {
            'enable_machine_learning': False,
            'enable_contextual_learning': True,
            'enable_performance_tracking': True,
            'enable_adaptive_thresholds': True
        })


class BaseRoutingStrategy(ABC):
    """Abstract base class for routing strategies"""
    
    def __init__(self, config: RoutingStrategyConfig):
        self.config = config
        self.performance_history: Dict[str, List[float]] = {}
        self.strategy_name = self.__class__.__name__
    
    @abstractmethod
    async def select_workflow(self, candidates: List[WorkflowMatch], 
                            analysis: RequestAnalysis) -> Tuple[WorkflowMatch, float]:
        """
        Select best workflow from candidates.
        
        Args:
            candidates: List of candidate workflows
            analysis: Request analysis result
            
        Returns:
            Tuple of (selected_workflow, confidence_score)
        """
        pass
    
    @abstractmethod
    def calculate_selection_score(self, workflow: WorkflowMatch, 
                                analysis: RequestAnalysis) -> float:
        """
        Calculate selection score for a workflow.
        
        Args:
            workflow: Workflow to score
            analysis: Request analysis
            
        Returns:
            Selection score (0.0 to 1.0)
        """
        pass
    
    def update_performance_history(self, workflow_id: str, performance_score: float) -> None:
        """Update performance history for a workflow"""
        if workflow_id not in self.performance_history:
            self.performance_history[workflow_id] = []
        
        self.performance_history[workflow_id].append(performance_score)
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history[workflow_id]) > 100:
            self.performance_history[workflow_id] = self.performance_history[workflow_id][-100:]
    
    def get_historical_performance(self, workflow_id: str) -> float:
        """Get average historical performance for a workflow"""
        if workflow_id not in self.performance_history or not self.performance_history[workflow_id]:
            return 0.5  # Neutral score for new workflows
        
        return sum(self.performance_history[workflow_id]) / len(self.performance_history[workflow_id])


class BestMatchStrategy(BaseRoutingStrategy):
    """Simple best match routing strategy based on match scores"""
    
    async def select_workflow(self, candidates: List[WorkflowMatch], 
                            analysis: RequestAnalysis) -> Tuple[WorkflowMatch, float]:
        """Select workflow with highest match score"""
        try:
            if not candidates:
                raise ValueError("No candidate workflows provided")
            
            # Calculate selection scores for all candidates
            scored_candidates = []
            for workflow in candidates:
                score = self.calculate_selection_score(workflow, analysis)
                scored_candidates.append((workflow, score))
            
            # Sort by score and select best
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_workflow, best_score = scored_candidates[0]
            
            # Check if best score meets minimum threshold
            if best_score < self.config.thresholds['minimum_score']:
                logger.warning(f"⚠️ Best match score {best_score} below minimum threshold")
            
            logger.debug(f"✅ Best match selected: {best_workflow.workflow_id} (score: {best_score:.3f})")
            return best_workflow, best_score
            
        except Exception as e:
            logger.error(f"❌ Best match selection failed: {e}")
            raise
    
    def calculate_selection_score(self, workflow: WorkflowMatch, 
                                analysis: RequestAnalysis) -> float:
        """Calculate selection score based on match score and analysis confidence"""
        try:
            base_score = workflow.match_score
            
            # Adjust based on analysis confidence
            confidence_factor = analysis.intent_classification.confidence * analysis.domain_classification.confidence
            adjusted_score = base_score * (0.7 + 0.3 * confidence_factor)
            
            # Apply historical performance if available
            if self.config.advanced_config.get('enable_performance_tracking', True):
                historical_performance = self.get_historical_performance(workflow.workflow_id)
                adjusted_score = adjusted_score * 0.9 + historical_performance * 0.1
            
            return min(1.0, adjusted_score)
            
        except Exception as e:
            logger.error(f"❌ Score calculation failed: {e}")
            return workflow.match_score


class WeightedScoringStrategy(BaseRoutingStrategy):
    """Advanced routing strategy using weighted scoring of multiple factors"""
    
    async def select_workflow(self, candidates: List[WorkflowMatch], 
                            analysis: RequestAnalysis) -> Tuple[WorkflowMatch, float]:
        """Select workflow using weighted multi-factor scoring"""
        try:
            if not candidates:
                raise ValueError("No candidate workflows provided")
            
            # Calculate weighted scores
            scored_candidates = []
            for workflow in candidates:
                score = self.calculate_selection_score(workflow, analysis)
                scored_candidates.append((workflow, score))
            
            # Sort by weighted score
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_workflow, best_score = scored_candidates[0]
            
            logger.debug(f"✅ Weighted scoring selected: {best_workflow.workflow_id} (score: {best_score:.3f})")
            return best_workflow, best_score
            
        except Exception as e:
            logger.error(f"❌ Weighted scoring selection failed: {e}")
            raise
    
    def calculate_selection_score(self, workflow: WorkflowMatch, 
                                analysis: RequestAnalysis) -> float:
        """Calculate weighted selection score using multiple factors"""
        try:
            weights = self.config.scoring_weights
            
            # Factor 1: Domain match score
            domain_score = self.calculate_domain_match_score(workflow, analysis)
            
            # Factor 2: Intent match score  
            intent_score = self.calculate_intent_match_score(workflow, analysis)
            
            # Factor 3: Confidence score
            confidence_score = self.calculate_confidence_score(workflow, analysis)
            
            # Factor 4: Historical performance
            historical_score = self.get_historical_performance(workflow.workflow_id)
            
            # Calculate weighted sum
            weighted_score = (
                domain_score * weights.get('domain_match', 0.4) +
                intent_score * weights.get('intent_match', 0.3) +
                confidence_score * weights.get('confidence_score', 0.2) +
                historical_score * weights.get('historical_performance', 0.1)
            )
            
            return min(1.0, weighted_score)
            
        except Exception as e:
            logger.error(f"❌ Weighted score calculation failed: {e}")
            return workflow.match_score
    
    def calculate_domain_match_score(self, workflow: WorkflowMatch, 
                                   analysis: RequestAnalysis) -> float:
        """Calculate domain match score"""
        # This would analyze how well the workflow's domain capabilities 
        # match the request's domain classification
        return workflow.match_score  # Simplified for now
    
    def calculate_intent_match_score(self, workflow: WorkflowMatch, 
                                   analysis: RequestAnalysis) -> float:
        """Calculate intent match score"""
        # This would analyze how well the workflow handles the detected intent
        intent_type = analysis.intent_classification.intent_type.value
        
        # Simple mapping of intents to capability alignment
        if intent_type in ['analysis_request', 'comparison_request']:
            return workflow.capability_alignment.get('analysis', 0.5)
        elif intent_type == 'information_request':
            return workflow.capability_alignment.get('information', 0.6)
        else:
            return 0.4
    
    def calculate_confidence_score(self, workflow: WorkflowMatch, 
                                 analysis: RequestAnalysis) -> float:
        """Calculate confidence-based score"""
        intent_confidence = analysis.intent_classification.confidence
        domain_confidence = analysis.domain_classification.confidence
        
        # Combined confidence affects workflow selection reliability
        combined_confidence = (intent_confidence + domain_confidence) / 2
        return workflow.match_score * combined_confidence


class AdaptiveStrategy(BaseRoutingStrategy):
    """Adaptive routing strategy that learns and adjusts over time"""
    
    def __init__(self, config: RoutingStrategyConfig):
        super().__init__(config)
        self.adaptation_history: Dict[str, Dict[str, Any]] = {}
        self.learning_rate = 0.1
    
    async def select_workflow(self, candidates: List[WorkflowMatch], 
                            analysis: RequestAnalysis) -> Tuple[WorkflowMatch, float]:
        """Select workflow using adaptive learning"""
        try:
            if not candidates:
                raise ValueError("No candidate workflows provided")
            
            # Apply adaptive scoring
            scored_candidates = []
            for workflow in candidates:
                score = self.calculate_adaptive_score(workflow, analysis)
                scored_candidates.append((workflow, score))
            
            # Select best candidate
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_workflow, best_score = scored_candidates[0]
            
            # Update adaptation history
            await self.update_adaptation_context(best_workflow, analysis)
            
            logger.debug(f"✅ Adaptive strategy selected: {best_workflow.workflow_id} (score: {best_score:.3f})")
            return best_workflow, best_score
            
        except Exception as e:
            logger.error(f"❌ Adaptive selection failed: {e}")
            raise
    
    def calculate_selection_score(self, workflow: WorkflowMatch, 
                                analysis: RequestAnalysis) -> float:
        """Calculate base selection score"""
        return self.calculate_adaptive_score(workflow, analysis)
    
    def calculate_adaptive_score(self, workflow: WorkflowMatch, 
                               analysis: RequestAnalysis) -> float:
        """Calculate adaptive score that learns from past performance"""
        try:
            # Start with base match score
            base_score = workflow.match_score
            
            # Apply contextual learning if enabled
            if self.config.advanced_config.get('enable_contextual_learning', True):
                context_key = self.generate_context_key(analysis)
                context_adjustment = self.get_context_adjustment(workflow.workflow_id, context_key)
                base_score = base_score * (1.0 + context_adjustment)
            
            # Apply adaptive thresholds if enabled
            if self.config.advanced_config.get('enable_adaptive_thresholds', True):
                threshold_adjustment = self.get_adaptive_threshold_adjustment(workflow.workflow_id)
                base_score = base_score * (1.0 + threshold_adjustment)
            
            # Historical performance with higher weight in adaptive strategy
            historical_performance = self.get_historical_performance(workflow.workflow_id)
            adapted_score = base_score * 0.7 + historical_performance * 0.3
            
            return min(1.0, max(0.0, adapted_score))
            
        except Exception as e:
            logger.error(f"❌ Adaptive score calculation failed: {e}")
            return workflow.match_score
    
    def generate_context_key(self, analysis: RequestAnalysis) -> str:
        """Generate context key for learning"""
        intent = analysis.intent_classification.intent_type.value
        domain = analysis.domain_classification.domain_type.value
        complexity = "high" if analysis.complexity_score > 0.7 else "medium" if analysis.complexity_score > 0.4 else "low"
        
        return f"{intent}_{domain}_{complexity}"
    
    def get_context_adjustment(self, workflow_id: str, context_key: str) -> float:
        """Get context-based adjustment for workflow"""
        if workflow_id not in self.adaptation_history:
            return 0.0
        
        workflow_history = self.adaptation_history[workflow_id]
        context_performance = workflow_history.get('contexts', {}).get(context_key, [])
        
        if not context_performance:
            return 0.0
        
        # Calculate adjustment based on context performance
        avg_performance = sum(context_performance) / len(context_performance)
        return (avg_performance - 0.5) * 0.2  # Small adjustment factor
    
    def get_adaptive_threshold_adjustment(self, workflow_id: str) -> float:
        """Get adaptive threshold adjustment"""
        if workflow_id not in self.adaptation_history:
            return 0.0
        
        workflow_history = self.adaptation_history[workflow_id]
        recent_performance = workflow_history.get('recent_performance', [])
        
        if len(recent_performance) < 5:  # Need minimum data
            return 0.0
        
        # Calculate trend
        recent_avg = sum(recent_performance[-5:]) / 5
        overall_avg = sum(recent_performance) / len(recent_performance)
        
        trend = recent_avg - overall_avg
        return trend * 0.1  # Small trend-based adjustment
    
    async def update_adaptation_context(self, selected_workflow: WorkflowMatch, 
                                      analysis: RequestAnalysis) -> None:
        """Update adaptation context for learning"""
        try:
            workflow_id = selected_workflow.workflow_id
            context_key = self.generate_context_key(analysis)
            
            if workflow_id not in self.adaptation_history:
                self.adaptation_history[workflow_id] = {
                    'contexts': {},
                    'recent_performance': [],
                    'selection_count': 0
                }
            
            # Update selection count
            self.adaptation_history[workflow_id]['selection_count'] += 1
            
            # Initialize context if not exists
            if context_key not in self.adaptation_history[workflow_id]['contexts']:
                self.adaptation_history[workflow_id]['contexts'][context_key] = []
            
            logger.debug(f"✅ Updated adaptation context for {workflow_id}: {context_key}")
            
        except Exception as e:
            logger.error(f"❌ Adaptation context update failed: {e}")
    
    def update_performance_feedback(self, workflow_id: str, performance_score: float, 
                                  context_key: Optional[str] = None) -> None:
        """Update performance feedback for adaptive learning"""
        try:
            # Update base performance history
            self.update_performance_history(workflow_id, performance_score)
            
            # Update adaptation-specific history
            if workflow_id in self.adaptation_history:
                self.adaptation_history[workflow_id]['recent_performance'].append(performance_score)
                
                # Keep only recent performance (last 50 entries)
                if len(self.adaptation_history[workflow_id]['recent_performance']) > 50:
                    self.adaptation_history[workflow_id]['recent_performance'] = \
                        self.adaptation_history[workflow_id]['recent_performance'][-50:]
                
                # Update context-specific performance if context provided
                if context_key and context_key in self.adaptation_history[workflow_id]['contexts']:
                    self.adaptation_history[workflow_id]['contexts'][context_key].append(performance_score)
                    
                    # Keep only recent context performance
                    if len(self.adaptation_history[workflow_id]['contexts'][context_key]) > 20:
                        self.adaptation_history[workflow_id]['contexts'][context_key] = \
                            self.adaptation_history[workflow_id]['contexts'][context_key][-20:]
            
            logger.debug(f"✅ Updated performance feedback for {workflow_id}: {performance_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Performance feedback update failed: {e}")


class MultiCriteriaStrategy(BaseRoutingStrategy):
    """Multi-criteria decision making strategy for complex routing scenarios"""
    
    def __init__(self, config: RoutingStrategyConfig):
        super().__init__(config)
        self.criteria_weights = {
            'accuracy': 0.3,
            'performance': 0.2,
            'reliability': 0.2,
            'cost': 0.1,
            'complexity_handling': 0.2
        }
    
    async def select_workflow(self, candidates: List[WorkflowMatch], 
                            analysis: RequestAnalysis) -> Tuple[WorkflowMatch, float]:
        """Select workflow using multi-criteria decision analysis"""
        try:
            if not candidates:
                raise ValueError("No candidate workflows provided")
            
            # Calculate multi-criteria scores
            scored_candidates = []
            for workflow in candidates:
                score = self.calculate_multi_criteria_score(workflow, analysis)
                scored_candidates.append((workflow, score))
            
            # Select best candidate
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            best_workflow, best_score = scored_candidates[0]
            
            logger.debug(f"✅ Multi-criteria selected: {best_workflow.workflow_id} (score: {best_score:.3f})")
            return best_workflow, best_score
            
        except Exception as e:
            logger.error(f"❌ Multi-criteria selection failed: {e}")
            raise
    
    def calculate_selection_score(self, workflow: WorkflowMatch, 
                                analysis: RequestAnalysis) -> float:
        """Calculate base selection score"""
        return self.calculate_multi_criteria_score(workflow, analysis)
    
    def calculate_multi_criteria_score(self, workflow: WorkflowMatch, 
                                     analysis: RequestAnalysis) -> float:
        """Calculate score using multiple criteria"""
        try:
            criteria_scores = {}
            
            # Criterion 1: Accuracy (based on match score and historical performance)
            accuracy = workflow.match_score * 0.7 + self.get_historical_performance(workflow.workflow_id) * 0.3
            criteria_scores['accuracy'] = accuracy
            
            # Criterion 2: Performance (based on estimated processing time)
            if workflow.estimated_processing_time:
                # Invert processing time for scoring (faster = better)
                max_time = 300.0  # 5 minutes max expected
                performance = max(0.0, 1.0 - (workflow.estimated_processing_time / max_time))
            else:
                performance = 0.5  # Neutral score if no estimate
            criteria_scores['performance'] = performance
            
            # Criterion 3: Reliability (based on historical success rate)
            reliability = self.calculate_reliability_score(workflow.workflow_id)
            criteria_scores['reliability'] = reliability
            
            # Criterion 4: Cost (simplified - based on processing time and complexity)
            cost = self.calculate_cost_score(workflow, analysis)
            criteria_scores['cost'] = cost
            
            # Criterion 5: Complexity handling (how well it handles complex requests)
            complexity_handling = self.calculate_complexity_handling_score(workflow, analysis)
            criteria_scores['complexity_handling'] = complexity_handling
            
            # Calculate weighted sum
            total_score = sum(
                criteria_scores[criterion] * self.criteria_weights.get(criterion, 0.0)
                for criterion in criteria_scores
            )
            
            return min(1.0, total_score)
            
        except Exception as e:
            logger.error(f"❌ Multi-criteria score calculation failed: {e}")
            return workflow.match_score
    
    def calculate_reliability_score(self, workflow_id: str) -> float:
        """Calculate reliability score based on historical success rates"""
        # This would track success/failure rates in a real implementation
        # For now, use historical performance as a proxy
        return self.get_historical_performance(workflow_id)
    
    def calculate_cost_score(self, workflow: WorkflowMatch, analysis: RequestAnalysis) -> float:
        """Calculate cost score (lower cost = higher score)"""
        # Simplified cost calculation based on processing time and complexity
        time_factor = 1.0
        if workflow.estimated_processing_time:
            time_factor = max(0.1, 1.0 - (workflow.estimated_processing_time / 300.0))
        
        complexity_factor = 1.0 - analysis.complexity_score * 0.5
        
        return time_factor * complexity_factor
    
    def calculate_complexity_handling_score(self, workflow: WorkflowMatch, 
                                          analysis: RequestAnalysis) -> float:
        """Calculate how well workflow handles complex requests"""
        # Higher complexity requests need workflows that can handle complexity
        workflow_complexity_capability = workflow.capability_alignment.get('complexity', 0.5)
        request_complexity = analysis.complexity_score
        
        # Score is higher when workflow capability matches request complexity
        if request_complexity > 0.7:  # High complexity request
            return workflow_complexity_capability
        elif request_complexity > 0.4:  # Medium complexity
            return 0.5 + workflow_complexity_capability * 0.5
        else:  # Low complexity
            return 0.8  # Most workflows can handle simple requests 