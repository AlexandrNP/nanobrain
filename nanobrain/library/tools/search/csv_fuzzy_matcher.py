"""
CSV Fuzzy Matcher for NanoBrain Framework

This module provides advanced fuzzy matching capabilities for CSV data,
combining multiple algorithms and confidence scoring for optimal search results.
"""

import re
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Optional imports with fallbacks
try:
    from fuzzywuzzy import fuzz, process
    from fuzzywuzzy.utils import full_process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class MatchAlgorithm(Enum):
    """Available fuzzy matching algorithms"""
    RATIO = "ratio"
    PARTIAL_RATIO = "partial_ratio"
    TOKEN_SORT_RATIO = "token_sort_ratio"
    TOKEN_SET_RATIO = "token_set_ratio"
    WEIGHTED_RATIO = "weighted_ratio"
    PHONETIC = "phonetic"
    ELASTICSEARCH = "elasticsearch"


@dataclass
class FuzzyMatch:
    """Represents a fuzzy match result"""
    value: str
    score: float
    algorithm: str
    field_name: str
    document_id: Optional[str] = None
    highlights: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchSuggestion:
    """Represents a search suggestion"""
    original_query: str
    suggested_query: str
    confidence: float
    reason: str


class CSVFuzzyMatcher:
    """
    Advanced fuzzy matcher for CSV data with multiple algorithms
    and intelligent scoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize fuzzy matcher with configuration"""
        self.config = config or {}
        
        # Configuration options
        self.default_threshold = self.config.get('default_threshold', 0.7)
        self.max_results = self.config.get('max_results', 100)
        self.enable_phonetic = self.config.get('enable_phonetic', True)
        self.enable_suggestions = self.config.get('enable_suggestions', True)
        self.field_weights = self.config.get('field_weights', {
            'default': 1.0,
            'name_fields': 2.0,
            'description_fields': 1.5,
            'id_fields': 3.0
        })
        
        # Algorithm weights for hybrid scoring
        self.algorithm_weights = self.config.get('algorithm_weights', {
            MatchAlgorithm.RATIO.value: 0.3,
            MatchAlgorithm.PARTIAL_RATIO.value: 0.2,
            MatchAlgorithm.TOKEN_SORT_RATIO.value: 0.25,
            MatchAlgorithm.TOKEN_SET_RATIO.value: 0.25
        })
        
        # Check dependencies
        if not FUZZYWUZZY_AVAILABLE:
            logger.warning("fuzzywuzzy not available, using basic string matching")
        
        logger.info("✅ CSV Fuzzy Matcher initialized")
    
    def fuzzy_search(self, 
                    query: str, 
                    candidates: List[Dict[str, Any]], 
                    search_fields: List[str],
                    threshold: Optional[float] = None) -> List[FuzzyMatch]:
        """
        Perform fuzzy search across multiple fields with confidence scoring
        """
        try:
            threshold = threshold or self.default_threshold
            matches = []
            
            logger.info(f"🔍 Fuzzy searching '{query}' across {len(candidates)} candidates")
            
            for candidate in candidates:
                doc_id = candidate.get('_id') or candidate.get('id')
                
                # Search across specified fields
                for field_name in search_fields:
                    if field_name not in candidate:
                        continue
                    
                    field_value = str(candidate[field_name])
                    if not field_value.strip():
                        continue
                    
                    # Calculate fuzzy match score
                    match_result = self._calculate_fuzzy_score(
                        query, field_value, field_name
                    )
                    
                    if match_result['score'] >= threshold:
                        fuzzy_match = FuzzyMatch(
                            value=field_value,
                            score=match_result['score'],
                            algorithm=match_result['algorithm'],
                            field_name=field_name,
                            document_id=doc_id,
                            highlights=match_result.get('highlights'),
                            metadata={
                                'candidate': candidate,
                                'algorithm_scores': match_result.get('algorithm_scores', {})
                            }
                        )
                        matches.append(fuzzy_match)
            
            # Sort by score (descending) and limit results
            matches.sort(key=lambda x: x.score, reverse=True)
            matches = matches[:self.max_results]
            
            logger.info(f"✅ Found {len(matches)} fuzzy matches above threshold {threshold}")
            
            return matches
            
        except Exception as e:
            logger.error(f"❌ Fuzzy search failed: {e}")
            return []
    
    def _calculate_fuzzy_score(self, query: str, candidate: str, field_name: str) -> Dict[str, Any]:
        """Calculate comprehensive fuzzy match score using multiple algorithms"""
        
        if not FUZZYWUZZY_AVAILABLE:
            # Fallback to basic string matching
            return self._basic_string_match(query, candidate, field_name)
        
        # Normalize inputs
        query_norm = self._normalize_string(query)
        candidate_norm = self._normalize_string(candidate)
        
        # Calculate scores using different algorithms
        algorithm_scores = {}
        
        try:
            algorithm_scores[MatchAlgorithm.RATIO.value] = fuzz.ratio(query_norm, candidate_norm) / 100.0
            algorithm_scores[MatchAlgorithm.PARTIAL_RATIO.value] = fuzz.partial_ratio(query_norm, candidate_norm) / 100.0
            algorithm_scores[MatchAlgorithm.TOKEN_SORT_RATIO.value] = fuzz.token_sort_ratio(query_norm, candidate_norm) / 100.0
            algorithm_scores[MatchAlgorithm.TOKEN_SET_RATIO.value] = fuzz.token_set_ratio(query_norm, candidate_norm) / 100.0
        except Exception as e:
            logger.warning(f"Fuzzywuzzy calculation failed: {e}")
            return self._basic_string_match(query, candidate, field_name)
        
        # Calculate weighted score
        weighted_score = 0.0
        for algorithm, weight in self.algorithm_weights.items():
            if algorithm in algorithm_scores:
                weighted_score += algorithm_scores[algorithm] * weight
        
        # Apply field-specific weighting
        field_weight = self._get_field_weight(field_name)
        final_score = min(1.0, weighted_score * field_weight)
        
        # Determine best algorithm
        best_algorithm = max(algorithm_scores, key=algorithm_scores.get)
        
        # Generate highlights
        highlights = self._generate_highlights(query, candidate)
        
        return {
            'score': final_score,
            'algorithm': best_algorithm,
            'algorithm_scores': algorithm_scores,
            'highlights': highlights,
            'field_weight': field_weight
        }
    
    def _basic_string_match(self, query: str, candidate: str, field_name: str) -> Dict[str, Any]:
        """Basic string matching fallback when fuzzywuzzy is not available"""
        query_lower = query.lower().strip()
        candidate_lower = candidate.lower().strip()
        
        # Exact match
        if query_lower == candidate_lower:
            score = 1.0
        # Contains match
        elif query_lower in candidate_lower:
            score = len(query_lower) / len(candidate_lower)
        # Word overlap
        else:
            query_words = set(query_lower.split())
            candidate_words = set(candidate_lower.split())
            
            if query_words and candidate_words:
                overlap = len(query_words.intersection(candidate_words))
                score = overlap / len(query_words.union(candidate_words))
            else:
                score = 0.0
        
        # Apply field weighting
        field_weight = self._get_field_weight(field_name)
        final_score = min(1.0, score * field_weight)
        
        return {
            'score': final_score,
            'algorithm': 'basic_string_match',
            'algorithm_scores': {'basic': score},
            'highlights': [query] if score > 0 else [],
            'field_weight': field_weight
        }
    
    def _normalize_string(self, text: str) -> str:
        """Normalize string for better matching"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common punctuation (but keep some for context)
        text = re.sub(r'[^\w\s\-\.]', ' ', text)
        
        # Remove extra spaces again
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _get_field_weight(self, field_name: str) -> float:
        """Get weight for specific field based on field name patterns"""
        field_lower = field_name.lower()
        
        # Check for specific field patterns
        if any(pattern in field_lower for pattern in ['name', 'title', 'label']):
            return self.field_weights.get('name_fields', 2.0)
        elif any(pattern in field_lower for pattern in ['description', 'summary', 'content']):
            return self.field_weights.get('description_fields', 1.5)
        elif any(pattern in field_lower for pattern in ['id', 'code', 'sku', 'key']):
            return self.field_weights.get('id_fields', 3.0)
        else:
            return self.field_weights.get('default', 1.0)
    
    def _generate_highlights(self, query: str, text: str) -> List[str]:
        """Generate highlight snippets for matched text"""
        highlights = []
        
        try:
            query_words = query.lower().split()
            text_lower = text.lower()
            
            for word in query_words:
                if len(word) >= 2 and word in text_lower:
                    # Find the word in context
                    start_idx = text_lower.find(word)
                    if start_idx != -1:
                        # Extract context around the match
                        context_start = max(0, start_idx - 20)
                        context_end = min(len(text), start_idx + len(word) + 20)
                        context = text[context_start:context_end]
                        
                        if context not in highlights:
                            highlights.append(context)
            
            return highlights[:3]  # Limit to 3 highlights
            
        except Exception as e:
            logger.warning(f"Highlight generation failed: {e}")
            return []
    
    def generate_suggestions(self, query: str, field_values: List[str]) -> List[SearchSuggestion]:
        """Generate search suggestions for query correction"""
        if not self.enable_suggestions:
            return []

        try:
            suggestions = []

            if FUZZYWUZZY_AVAILABLE:
                # Find close matches for potential typos
                close_matches = process.extract(
                    query,
                    field_values,
                    limit=10,  # Get more candidates
                    scorer=fuzz.ratio
                )

                for match, score in close_matches:
                    confidence = score / 100.0
                    if 0.5 <= confidence < 0.95:  # Broader range for suggestions
                        suggestion = SearchSuggestion(
                            original_query=query,
                            suggested_query=match,
                            confidence=confidence,
                            reason="spelling_correction"
                        )
                        suggestions.append(suggestion)

                # Also try partial ratio for substring matches
                partial_matches = process.extract(
                    query,
                    field_values,
                    limit=5,
                    scorer=fuzz.partial_ratio
                )

                for match, score in partial_matches:
                    confidence = score / 100.0
                    if confidence >= 0.7:  # High partial match confidence
                        suggestion = SearchSuggestion(
                            original_query=query,
                            suggested_query=match,
                            confidence=confidence * 0.9,  # Slightly lower for partial
                            reason="partial_match"
                        )
                        suggestions.append(suggestion)

            else:
                # Fallback without fuzzywuzzy
                query_lower = query.lower()
                for value in field_values:
                    value_lower = value.lower()

                    # Simple substring matching
                    if query_lower in value_lower or value_lower in query_lower:
                        # Calculate simple similarity
                        similarity = len(query_lower) / max(len(value_lower), len(query_lower))
                        if 0.3 <= similarity < 0.9:
                            suggestion = SearchSuggestion(
                                original_query=query,
                                suggested_query=value,
                                confidence=similarity,
                                reason="substring_match"
                            )
                            suggestions.append(suggestion)

            # Generate word-based suggestions
            query_words = query.split()
            if len(query_words) > 1:
                # Try partial queries
                for i in range(len(query_words)):
                    partial_query = ' '.join(query_words[:i+1])
                    if partial_query != query:
                        suggestion = SearchSuggestion(
                            original_query=query,
                            suggested_query=partial_query,
                            confidence=0.8,
                            reason="partial_query"
                        )
                        suggestions.append(suggestion)
            
            # Remove duplicates and sort by confidence
            unique_suggestions = {}
            for suggestion in suggestions:
                key = suggestion.suggested_query
                if key not in unique_suggestions or suggestion.confidence > unique_suggestions[key].confidence:
                    unique_suggestions[key] = suggestion
            
            result = list(unique_suggestions.values())
            result.sort(key=lambda x: x.confidence, reverse=True)
            
            return result[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"❌ Suggestion generation failed: {e}")
            return []
    
    def exact_match(self, query: str, candidates: List[Dict[str, Any]], search_fields: List[str]) -> List[FuzzyMatch]:
        """Perform exact matching for precise queries"""
        try:
            matches = []
            query_lower = query.lower().strip()
            
            for candidate in candidates:
                doc_id = candidate.get('_id') or candidate.get('id')
                
                for field_name in search_fields:
                    if field_name not in candidate:
                        continue
                    
                    field_value = str(candidate[field_name])
                    field_value_lower = field_value.lower().strip()
                    
                    # Exact match
                    if query_lower == field_value_lower:
                        match = FuzzyMatch(
                            value=field_value,
                            score=1.0,
                            algorithm="exact_match",
                            field_name=field_name,
                            document_id=doc_id,
                            metadata={'candidate': candidate}
                        )
                        matches.append(match)
                    # Contains match (for partial exact matching)
                    elif query_lower in field_value_lower:
                        score = len(query_lower) / len(field_value_lower)
                        match = FuzzyMatch(
                            value=field_value,
                            score=score,
                            algorithm="contains_match",
                            field_name=field_name,
                            document_id=doc_id,
                            metadata={'candidate': candidate}
                        )
                        matches.append(match)
            
            # Sort by score
            matches.sort(key=lambda x: x.score, reverse=True)
            
            logger.info(f"✅ Found {len(matches)} exact matches")
            return matches
            
        except Exception as e:
            logger.error(f"❌ Exact match failed: {e}")
            return []
    
    def calculate_confidence_score(self, matches: List[FuzzyMatch]) -> float:
        """Calculate overall confidence score for a set of matches"""
        if not matches:
            return 0.0
        
        # Weight by match scores and field importance
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for match in matches:
            field_weight = self._get_field_weight(match.field_name)
            weighted_score = match.score * field_weight
            total_weighted_score += weighted_score
            total_weight += field_weight
        
        if total_weight > 0:
            confidence = total_weighted_score / total_weight
        else:
            confidence = sum(match.score for match in matches) / len(matches)
        
        return min(1.0, confidence)


# Factory function following NanoBrain patterns
def from_config(config: Dict[str, Any]) -> CSVFuzzyMatcher:
    """Create CSVFuzzyMatcher from configuration dictionary"""
    return CSVFuzzyMatcher(config)
