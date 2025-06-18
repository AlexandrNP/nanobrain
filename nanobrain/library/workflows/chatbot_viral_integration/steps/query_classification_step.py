"""
Query Classification Step

Classifies user queries as viral annotation requests or conversational queries
using keyword-based intent detection and parameter extraction.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from nanobrain.core.step import Step, StepConfig
from nanobrain.library.infrastructure.data.chat_session_data import (
    QueryClassificationData, MessageType
)
from typing import Dict, Any, List, Optional
import re
import time
from datetime import datetime


class QueryClassificationStep(Step):
    """
    Step for classifying user queries into viral annotation requests 
    or conversational queries about alphaviruses.
    
    Uses keyword-based classification with confidence scoring and
    parameter extraction.
    """
    
    def __init__(self, config: StepConfig):
        super().__init__(config)
        
        # Get nested config dict
        step_config = getattr(config, 'config', {})
        
        # Classification configuration with improved thresholds
        self.confidence_thresholds = step_config.get('confidence_thresholds', {
            'annotation': 0.45,  # Lowered from 0.6 for better detection
            'conversational': 0.35,  # Lowered from 0.5 for better detection
            'unknown': 0.25  # Lowered from 0.4 for better fallback
        })
        
        self.enable_parameter_extraction = step_config.get('enable_parameter_extraction', True)
        self.enable_reasoning = step_config.get('enable_reasoning', True)
        
        # Initialize keyword mappings
        self.annotation_keywords = self._initialize_annotation_keywords()
        self.conversational_keywords = self._initialize_conversational_keywords()
        self.parameter_patterns = self._initialize_parameter_patterns()
        
        self.nb_logger.info("🧠 Query Classification Step initialized")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify user query and extract parameters.
        
        Args:
            input_data: Contains 'user_query' and 'session_data'
            
        Returns:
            Classification result with intent, confidence, and routing decision
        """
        start_time = time.time()
        
        try:
            user_query = input_data.get('user_query', '')
            session_data = input_data.get('session_data')
            
            if not user_query:
                raise ValueError("No user_query provided")
            
            query_lower = user_query.lower()
            
            self.nb_logger.info(f"🔍 Classifying query: '{user_query[:100]}...'")
            
            # Calculate intent scores
            annotation_score = self._calculate_annotation_score(query_lower)
            conversational_score = self._calculate_conversational_score(query_lower)
            
            # Determine intent and confidence
            intent, confidence = self._determine_intent(annotation_score, conversational_score)
            
            # Extract parameters if applicable
            extracted_parameters = {}
            if self.enable_parameter_extraction:
                extracted_parameters = await self._extract_parameters(user_query, intent)
            
            # Generate reasoning
            reasoning = ""
            if self.enable_reasoning:
                reasoning = self._generate_reasoning(
                    user_query, intent, confidence, annotation_score, conversational_score
                )
            
            # Create classification data
            classification_data = QueryClassificationData(
                original_query=user_query,
                intent=intent,
                confidence=confidence,
                extracted_parameters=extracted_parameters,
                reasoning=reasoning,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            # Determine routing decision
            routing_decision = self._determine_routing(classification_data)
            
            # Update session metrics if available
            if session_data:
                session_data.metrics.total_messages += 1
                if intent == 'annotation':
                    session_data.metrics.annotation_requests += 1
                elif intent == 'conversational':
                    session_data.metrics.conversational_queries += 1
            
            self.nb_logger.info(f"✅ Classified as '{intent}' (confidence: {confidence:.3f})")
            
            return {
                'success': True,
                'classification_data': classification_data,
                'routing_decision': routing_decision
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Classification failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'classification_data': None,
                'routing_decision': {'next_step': 'conversational_response'}
            }
    
    def _calculate_annotation_score(self, query_lower: str) -> float:
        """Calculate annotation intent score based on keywords"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, keywords in self.annotation_keywords.items():
            category_score = 0.0
            category_weight = keywords.get('weight', 1.0)
            
            for keyword in keywords.get('terms', []):
                if keyword in query_lower:
                    category_score += keywords.get('match_score', 1.0)
            
            # Cap category score at 1.0 instead of normalizing by term count
            # This prevents penalizing queries for not matching ALL terms in a category
            normalized_score = min(category_score, 1.0)
            
            total_score += normalized_score * category_weight
            total_weight += category_weight
        
        # Normalize final score
        return min(total_score / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def _calculate_conversational_score(self, query_lower: str) -> float:
        """Calculate conversational intent score based on keywords"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for category, keywords in self.conversational_keywords.items():
            category_score = 0.0
            category_weight = keywords.get('weight', 1.0)
            
            for keyword in keywords.get('terms', []):
                if keyword in query_lower:
                    category_score += keywords.get('match_score', 1.0)
            
            # Cap category score at 1.0 instead of normalizing by term count
            # This prevents penalizing queries for not matching ALL terms in a category
            normalized_score = min(category_score, 1.0)
            
            total_score += normalized_score * category_weight
            total_weight += category_weight
        
        # Normalize final score
        return min(total_score / total_weight if total_weight > 0 else 0.0, 1.0)
    
    def _determine_intent(self, annotation_score: float, conversational_score: float) -> tuple[str, float]:
        """Determine intent and confidence from scores"""
        
        # Compare scores against thresholds
        annotation_threshold = self.confidence_thresholds['annotation']
        conversational_threshold = self.confidence_thresholds['conversational']
        
        if annotation_score >= annotation_threshold and annotation_score > conversational_score:
            return 'annotation', annotation_score
        elif conversational_score >= conversational_threshold:
            return 'conversational', conversational_score
        else:
            # Default to conversational for unclear queries
            return 'unknown', max(annotation_score, conversational_score)
    
    async def _extract_parameters(self, query: str, intent: str) -> Dict[str, Any]:
        """Extract relevant parameters from query"""
        
        parameters = {}
        
        if intent == 'annotation':
            # Extract biological sequences
            sequences = self._extract_sequences(query)
            if sequences:
                parameters['sequences'] = sequences
            
            # Extract genome IDs
            genome_ids = self._extract_genome_ids(query)
            if genome_ids:
                parameters['genome_ids'] = genome_ids
            
            # Extract protein names
            protein_names = self._extract_protein_names(query)
            if protein_names:
                parameters['protein_names'] = protein_names
            
            # Extract organism names
            organisms = self._extract_organisms(query)
            if organisms:
                parameters['organisms'] = organisms
            
            # Extract accession numbers
            accessions = self._extract_accession_numbers(query)
            if accessions:
                parameters['accession_numbers'] = accessions
            
            # Determine analysis scope
            parameters['analysis_scope'] = self._determine_analysis_scope(query)
        
        elif intent == 'conversational':
            # Extract topic hints for conversational routing
            topic_hints = self._extract_topic_hints(query)
            if topic_hints:
                parameters['topic_hints'] = topic_hints
        
        return parameters
    
    def _extract_sequences(self, query: str) -> List[str]:
        """Extract biological sequences from query"""
        
        sequences = []
        
        # Look for DNA/RNA/protein sequences
        sequence_patterns = [
            r'[ATCG]{10,}',  # DNA sequences
            r'[AUCG]{10,}',  # RNA sequences  
            r'[ACDEFGHIKLMNPQRSTVWY]{5,}'  # Protein sequences
        ]
        
        for pattern in sequence_patterns:
            matches = re.findall(pattern, query.upper())
            sequences.extend(matches)
        
        return list(set(sequences))  # Remove duplicates
    
    def _extract_genome_ids(self, query: str) -> List[str]:
        """Extract genome IDs from query"""
        
        # Look for common genome ID patterns
        patterns = [
            r'NC_\d+',  # NCBI RefSeq
            r'AC_\d+',  # NCBI GenBank
            r'GCF_\d+',  # NCBI Assembly
        ]
        
        genome_ids = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            genome_ids.extend(matches)
        
        return genome_ids
    
    def _extract_protein_names(self, query: str) -> List[str]:
        """Extract protein names from query"""
        
        # Common alphavirus protein names
        protein_patterns = [
            r'\bE1\b', r'\bE2\b', r'\bE3\b',  # Envelope proteins
            r'\bcapsid\b', r'\bcore\b',  # Capsid proteins
            r'\bnsP[1-4]\b',  # Non-structural proteins
            r'\b6K\b',  # 6K protein
            r'\benvelope protein\b',
            r'\bstructural protein\b',
            r'\bnon-structural protein\b'
        ]
        
        proteins = []
        for pattern in protein_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            proteins.extend(matches)
        
        return proteins
    
    def _extract_organisms(self, query: str) -> List[str]:
        """Extract organism names from query"""
        
        # Common alphavirus names
        organisms = [
            'chikungunya', 'eastern equine encephalitis', 'western equine encephalitis',
            'venezuelan equine encephalitis', 'sindbis', 'semliki forest',
            'ross river', 'mayaro', 'o\'nyong-nyong', 'barmah forest'
        ]
        
        found_organisms = []
        query_lower = query.lower()
        
        for organism in organisms:
            if organism in query_lower:
                found_organisms.append(organism)
        
        return found_organisms
    
    def _extract_accession_numbers(self, query: str) -> List[str]:
        """Extract accession numbers from query"""
        
        # Common accession number patterns
        patterns = [
            r'[A-Z]{1,2}\d{5,8}',  # GenBank/RefSeq
            r'[A-Z]{3}\d{5}',  # UniProt
        ]
        
        accessions = []
        for pattern in patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            accessions.extend(matches)
        
        return accessions
    
    def _determine_analysis_scope(self, query: str) -> str:
        """Determine the scope of analysis requested"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['multiple', 'batch', 'many', 'several']):
            return 'batch_analysis'
        elif any(word in query_lower for word in ['single', 'one', 'this']):
            return 'single_sequence'
        elif any(word in query_lower for word in ['genome', 'complete', 'full']):
            return 'genome_analysis'
        else:
            return 'single_sequence'  # Default
    
    def _extract_topic_hints(self, query: str) -> List[str]:
        """Extract topic hints for conversational queries"""
        
        topic_keywords = {
            'structure': ['structure', 'protein', 'envelope', 'capsid', 'organization'],
            'replication': ['replication', 'cycle', 'reproduction', 'multiplication'],
            'diseases': ['disease', 'symptom', 'illness', 'infection', 'pathology'],
            'transmission': ['transmission', 'vector', 'mosquito', 'spread'],
            'evolution': ['evolution', 'phylogeny', 'mutation', 'variant'],
            'classification': ['classification', 'taxonomy', 'family', 'genus']
        }
        
        query_lower = query.lower()
        topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def _determine_routing(self, classification_data: QueryClassificationData) -> Dict[str, Any]:
        """Determine next step routing based on classification"""
        
        intent = classification_data.intent
        confidence = classification_data.confidence
        
        routing = {
            'next_step': 'conversational_response',  # Default
            'confidence': confidence,
            'requires_backend': False,
            'topic_hints': classification_data.extracted_parameters.get('topic_hints', ['general']),
            'clarification_needed': False
        }
        
        if intent == 'annotation' and confidence >= self.confidence_thresholds['annotation']:
            routing['next_step'] = 'annotation_job'
            routing['requires_backend'] = True
            
        elif intent == 'conversational' and confidence >= self.confidence_thresholds['conversational']:
            routing['next_step'] = 'conversational_response'
            
        elif intent == 'unknown' or confidence < self.confidence_thresholds.get('unknown', 0.4):
            routing['next_step'] = 'conversational_response'
            routing['clarification_needed'] = True
        
        return routing
    
    def _generate_reasoning(self, query: str, intent: str, confidence: float, 
                          annotation_score: float, conversational_score: float) -> str:
        """Generate human-readable reasoning for classification decision"""
        
        reasoning_parts = []
        
        # Explain score calculation
        reasoning_parts.append(f"Query analyzed for intent classification:")
        reasoning_parts.append(f"- Annotation signals: {annotation_score:.3f}")
        reasoning_parts.append(f"- Conversational signals: {conversational_score:.3f}")
        
        # Explain decision
        if intent == 'annotation':
            reasoning_parts.append(f"Classified as annotation request due to high annotation score")
        elif intent == 'conversational':
            reasoning_parts.append(f"Classified as conversational query based on educational keywords")
        else:
            reasoning_parts.append(f"Intent unclear, defaulting to conversational with clarification")
        
        # Explain confidence
        if confidence >= 0.8:
            reasoning_parts.append(f"High confidence classification")
        elif confidence >= 0.6:
            reasoning_parts.append(f"Moderate confidence classification")
        else:
            reasoning_parts.append(f"Low confidence, may need clarification")
        
        return " ".join(reasoning_parts)
    
    def _initialize_annotation_keywords(self) -> Dict[str, Dict[str, Any]]:
        """Initialize annotation-specific keyword mappings"""
        
        return {
            'analysis_requests': {
                'terms': ['analyze', 'annotate', 'process', 'run', 'execute', 'predict', 'identify', 'compute', 'find'],
                'weight': 2.2,  # Increased weight
                'match_score': 1.0
            },
            'bioinformatics_terms': {
                'terms': ['pssm', 'blast', 'annotation', 'pipeline', 'workflow', 'algorithm', 'analysis', 'search'],
                'weight': 2.5,
                'match_score': 1.2
            },
            'sequence_terms': {
                'terms': ['sequence', 'protein', 'genome', 'dna', 'rna', 'fasta', 'viral', 'virus'],
                'weight': 2.0,  # Increased weight since viral sequences are key
                'match_score': 1.0
            },
            'functional_terms': {
                'terms': ['function', 'domain', 'motif', 'feature', 'structure', 'annotation', 'characterization'],
                'weight': 1.5,
                'match_score': 0.8
            },
            'data_indicators': {
                'terms': ['data', 'file', 'input', 'upload', 'my', 'this'],
                'weight': 1.2,
                'match_score': 0.7
            }
        }
    
    def _initialize_conversational_keywords(self) -> Dict[str, Dict[str, Any]]:
        """Initialize conversational keyword mappings"""
        
        return {
            'question_words': {
                'terms': ['what', 'how', 'why', 'when', 'where', 'which', 'tell', 'explain'],
                'weight': 1.5,
                'match_score': 1.0
            },
            'virus_biology': {
                'terms': ['virus', 'viral', 'alphavirus', 'replication', 'infection'],
                'weight': 2.0,
                'match_score': 1.2
            },
            'disease_terms': {
                'terms': ['disease', 'symptom', 'pathology', 'clinical', 'treatment'],
                'weight': 1.8,
                'match_score': 1.0
            },
            'educational_terms': {
                'terms': ['learn', 'understand', 'know', 'information', 'about'],
                'weight': 1.3,
                'match_score': 0.9
            },
            'specific_viruses': {
                'terms': ['chikungunya', 'eastern equine', 'western equine', 'venezuelan equine'],
                'weight': 2.2,
                'match_score': 1.3
            }
        }
    
    def _initialize_parameter_patterns(self) -> Dict[str, str]:
        """Initialize regex patterns for parameter extraction"""
        
        return {
            'dna_sequence': r'[ATCG]{10,}',
            'rna_sequence': r'[AUCG]{10,}', 
            'protein_sequence': r'[ACDEFGHIKLMNPQRSTVWY]{5,}',
            'genbank_id': r'[A-Z]{1,2}\d{5,8}',
            'refseq_id': r'[A-Z]{2}_\d+',
            'uniprot_id': r'[A-Z]\d[A-Z0-9]{3}\d'
        } 