#!/usr/bin/env python3
"""
Mock Services for Chatbot Viral Integration Testing

This module provides mock implementations of external services
for controlled testing as specified in the testing plan.

Author: NanoBrain Development Team
Date: January 2025
Version: 1.0.0
"""

import asyncio
import time
import json
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock
from .test_data import MockTestData


class MockBVBRCService:
    """
    Mock BV-BRC service implementation (from section 5.3.1)
    
    Provides controlled responses for testing without
    depending on external BV-BRC API availability.
    """
    
    def __init__(self, simulate_delays: bool = True):
        self.simulate_delays = simulate_delays
        self.request_count = 0
        self.timeout_enabled = False
        self.rate_limit_enabled = False
        self.error_rate = 0.0
        
    async def get_viral_data(self, organism: str) -> Dict[str, Any]:
        """Return controlled test data for viral organisms"""
        self.request_count += 1
        
        if self.simulate_delays:
            await asyncio.sleep(0.1)  # Simulate network delay
            
        if self.timeout_enabled:
            await asyncio.sleep(35)  # Simulate timeout
            raise TimeoutError("BV-BRC service timeout")
            
        if self.rate_limit_enabled and self.request_count > 10:
            raise Exception("Rate limit exceeded")
            
        if self.error_rate > 0 and (self.request_count % int(1/self.error_rate)) == 0:
            raise Exception("Simulated BV-BRC service error")
        
        # Return organism-specific mock data
        if "eeev" in organism.lower() or "eastern equine" in organism.lower():
            return {
                **MockTestData.MOCK_BVBRC_RESPONSE,
                "query_organism": organism,
                "query_timestamp": time.time()
            }
        elif "chikungunya" in organism.lower():
            return {
                "genomes": [{
                    "genome_id": "2234567.3",
                    "genome_name": "Chikungunya virus",
                    "organism_name": "Chikungunya virus",
                    "taxon_id": 37124,
                    "genome_status": "complete",
                    "genome_length": 11826
                }],
                "proteins": [{
                    "feature_id": "fig|2234567.3.peg.1",
                    "annotation": "non-structural polyprotein nsP1",
                    "product": "nsP1",
                    "aa_sequence": "MADEKKKHVLSALG..."
                }],
                "query_organism": organism,
                "query_timestamp": time.time()
            }
        elif "alphavirus" in organism.lower():
            return {
                "genomes": [{
                    "genome_id": "3234567.3",
                    "genome_name": "Alphavirus",
                    "organism_name": "Alphavirus",
                    "taxon_id": 11018,
                    "genome_status": "complete",
                    "genome_length": 11500
                }],
                "proteins": [{
                    "feature_id": "fig|3234567.3.peg.1",
                    "annotation": "structural polyprotein",
                    "product": "Capsid",
                    "aa_sequence": "MSILGKGPQR..."
                }],
                "query_organism": organism,
                "query_timestamp": time.time()
            }
        else:
            # Unknown organism
            return {
                "genomes": [],
                "proteins": [],
                "error": f"No data found for organism: {organism}",
                "query_organism": organism,
                "query_timestamp": time.time()
            }
    
    def simulate_timeout(self) -> None:
        """Enable timeout simulation"""
        self.timeout_enabled = True
        
    def simulate_rate_limit(self) -> None:
        """Enable rate limiting simulation"""
        self.rate_limit_enabled = True
        
    def set_error_rate(self, rate: float) -> None:
        """Set error rate for simulating service failures"""
        self.error_rate = rate
        
    def reset(self) -> None:
        """Reset service state"""
        self.request_count = 0
        self.timeout_enabled = False
        self.rate_limit_enabled = False
        self.error_rate = 0.0


class MockExternalTools:
    """
    Mock external bioinformatics tools (from section 5.3.2)
    
    Provides mock implementations of MUSCLE, MMseqs2,
    and other external tools for testing.
    """
    
    def __init__(self, simulate_processing: bool = True):
        self.simulate_processing = simulate_processing
        self.tool_call_count = 0
        
    async def muscle_alignment(self, sequences: List[str]) -> str:
        """Return mock multiple sequence alignment"""
        self.tool_call_count += 1
        
        if self.simulate_processing:
            await asyncio.sleep(0.2)  # Simulate processing time
            
        # Generate mock FASTA alignment
        mock_alignment = ">seq1\nMSIKGKPQRFGFLAKVREKR\n"
        mock_alignment += ">seq2\nMSI-GKPQRFGFLAKVRE-R\n"
        mock_alignment += ">seq3\nMSIKGKPQRFGFLAKVREKR\n"
        
        return mock_alignment
    
    async def mmseqs2_clustering(self, sequences: List[str]) -> Dict[str, Any]:
        """Return mock clustering results"""
        self.tool_call_count += 1
        
        if self.simulate_processing:
            await asyncio.sleep(0.3)  # Simulate processing time
            
        return {
            **MockTestData.MOCK_CLUSTERING_RESULT,
            "input_sequences": len(sequences),
            "processing_time": 0.3
        }
    
    async def hmmer_search(self, query_sequence: str, database: str) -> Dict[str, Any]:
        """Return mock HMMER search results"""
        self.tool_call_count += 1
        
        if self.simulate_processing:
            await asyncio.sleep(0.15)
            
        return {
            "hits": [
                {
                    "target": "PF00123.45",
                    "description": "Viral protein domain",
                    "evalue": 1.2e-45,
                    "score": 156.7,
                    "start": 45,
                    "end": 230
                }
            ],
            "num_hits": 1,
            "query_length": len(query_sequence)
        }
    
    async def blast_search(self, query: str, database: str) -> Dict[str, Any]:
        """Return mock BLAST search results"""
        self.tool_call_count += 1
        
        if self.simulate_processing:
            await asyncio.sleep(0.25)
            
        return {
            "hits": [
                {
                    "subject_id": "ref|YP_009164643.1|",
                    "description": "non-structural polyprotein nsP1",
                    "evalue": 0.0,
                    "identity": 98.5,
                    "coverage": 95.2
                }
            ],
            "num_hits": 1
        }
    
    def reset_counters(self) -> None:
        """Reset tool call counters"""
        self.tool_call_count = 0


class MockWorkflowComponents:
    """Mock components for workflow testing"""
    
    def __init__(self):
        self.step_execution_times = {}
        self.step_results = {}
        
    async def mock_query_classification_step(self, query: str) -> Dict[str, Any]:
        """Mock query classification step execution"""
        await asyncio.sleep(0.1)  # Simulate processing
        
        # Handle edge cases first
        if not query or not query.strip():
            return {
                "intent": "conversational",
                "confidence": 0.3,  # Low confidence for empty queries
                "routing_decision": {"next_step": "conversational_response"},
                "processing_time": 0.1
            }
        
        # Handle very long queries
        if len(query) > 500:
            confidence_penalty = min(0.2, len(query) / 5000)
            base_confidence = max(0.4, 0.85 - confidence_penalty)
        else:
            base_confidence = 0.85
        
        # Handle queries with only special characters
        if query.strip() and all(not c.isalnum() for c in query.strip()):
            return {
                "intent": "conversational",
                "confidence": 0.4,  # Low confidence for special characters only
                "routing_decision": {"next_step": "conversational_response"},
                "processing_time": 0.1
            }
        
        # Simple keyword-based classification for testing
        if any(keyword in query.lower() for keyword in ['create', 'analyze', 'generate', 'pssm', 'clustering']):
            intent = "annotation"
            confidence = base_confidence
            routing = "annotation_job"
        else:
            intent = "conversational"
            confidence = min(0.90, base_confidence + 0.05)  # Slightly higher for conversational
            routing = "conversational_response"
            
        return {
            "intent": intent,
            "confidence": confidence,
            "routing_decision": {"next_step": routing},
            "processing_time": 0.1
        }
    
    async def mock_annotation_job_step(self, query: str, organism: str = "EEEV") -> Dict[str, Any]:
        """Mock annotation job step execution"""
        job_id = f"mock_job_{int(time.time())}"
        
        # Simulate progressive execution
        progress_steps = [
            (10, "Data acquisition"),
            (30, "Sequence processing"),
            (60, "Clustering analysis"),
            (80, "Multiple alignment"),
            (100, "PSSM generation")
        ]
        
        for progress, status in progress_steps:
            await asyncio.sleep(0.2)  # Simulate step execution
            
        return {
            "job_id": job_id,
            "status": "completed",
            "organism": organism,
            "pssm_matrix": MockTestData.MOCK_PSSM_MATRIX,
            "execution_time": 1.0,
            "output_files": {
                "pssm_json": f"/tmp/{job_id}_pssm.json",
                "alignment": f"/tmp/{job_id}_alignment.fasta"
            }
        }
    
    async def mock_conversational_response_step(self, query: str) -> Dict[str, Any]:
        """Mock conversational response step execution"""
        await asyncio.sleep(0.2)  # Simulate response generation
        
        # Handle empty/invalid queries
        if not query or not query.strip():
            response = "I need more information to help you. Could you please provide a specific question about viruses or bioinformatics?"
            confidence = 0.3
        elif len(query) > 1000:
            response = "Your query is quite complex. Let me provide a focused response on the main topic of viral analysis and bioinformatics."
            confidence = 0.6
        elif query.strip() and all(not c.isalnum() for c in query.strip()):
            response = "I notice your query contains special characters. Could you please rephrase your question about viruses or bioinformatics?"
            confidence = 0.4
        else:
            # Generate contextual response
            response = self._generate_contextual_response(query)
            confidence = 0.88
        
        # Ensure response meets length requirements (100-2000 chars)
        if len(response) < 100:
            response = self._pad_response(response)
        elif len(response) > 2000:
            response = response[:1997] + "..."
        
        return {
            "response": response,
            "requires_markdown": True,
            "confidence": confidence,
            "processing_time": 0.2
        }
    
    def _generate_contextual_response(self, query: str) -> str:
        """Generate appropriate response based on query content"""
        if "eeev" in query.lower():
            response = """
            # Eastern Equine Encephalitis Virus (EEEV)
            
            **Eastern Equine Encephalitis Virus (EEEV)** is a mosquito-borne virus that causes 
            Eastern equine encephalitis in humans and horses. 
            
            ## Key Features:
            - **Family**: Togaviridae
            - **Genus**: Alphavirus
            - **Transmission**: Mosquito vector (primarily *Culiseta melanura*)
            - **Geographic range**: Eastern United States, Central America
            
            ## Clinical Significance:
            - High case fatality rate (30-90%)
            - Severe neurological symptoms
            - No specific treatment or vaccine available
            """
        elif "pssm" in query.lower():
            response = """
            # Position-Specific Scoring Matrix (PSSM)
            
            A **Position-Specific Scoring Matrix** is a bioinformatics tool used to 
            represent sequence motifs and evolutionary conservation patterns.
            
            ## Applications:
            - Protein domain identification
            - Sequence alignment
            - Functional annotation
            - Evolutionary analysis
            """
        elif any(term in query.lower() for term in ["spread", "transmission", "transmit"]):
            response = """
            # Viral Transmission Mechanisms
            
            Viruses spread through various **transmission** pathways:
            
            ## Primary Transmission Routes:
            - **Vector-borne transmission**: Mosquitoes, ticks, and other arthropod vectors
            - **Direct contact**: Person-to-person spread through respiratory droplets
            - **Fomite transmission**: Contaminated surfaces and objects
            
            ## Vector-Borne Transmission:
            For viruses like EEEV, **mosquito vectors** (primarily *Culiseta melanura*) 
            facilitate the spread between avian hosts and occasional transmission to mammals.
            
            ## Environmental Factors:
            - Seasonal patterns affect vector activity
            - Geographic distribution influences spread patterns
            - Climate change impacts transmission dynamics
            """
        elif any(term in query.lower() for term in ["protein", "structure", "alphavirus", "replicate"]):
            response = """
            # Viral Protein Structure and Function
            
            **Viral proteins** play essential roles in viral replication, pathogenesis, and host interaction.
            
            ## Key Structural Elements:
            - **Capsid proteins**: Form the protective protein shell
            - **Envelope proteins**: Facilitate host cell attachment and entry
            - **Non-structural proteins**: Support replication and assembly
            
            ## Alphavirus Structure:
            Alphaviruses like **EEEV** contain several **structural proteins**:
            - **E1 and E2 glycoproteins**: Mediate membrane fusion and receptor binding
            - **Capsid protein**: Forms nucleocapsid with genomic RNA
            
            ## Replication Mechanisms:
            - Translation of viral **polyproteins**
            - Processing by viral and host **proteases**
            - Assembly of replication complexes
            """
        else:
            response = """
            # Virology and Bioinformatics Information
            
            I can help you with information about **viruses** and **bioinformatics analysis**. 
            Please feel free to ask specific questions about:
            
            ## Available Topics:
            - **Viral classification** and taxonomy
            - **Protein structure** and function
            - **Transmission mechanisms** and epidemiology
            - **Computational analysis** methods
            - **PSSM matrices** and sequence analysis
            """
            
        return response
    
    def _pad_response(self, response: str) -> str:
        """Pad short responses to meet minimum length requirements"""
        if len(response) < 100:
            padding = """
            
            ## Additional Information
            
            For more detailed information about viral research and bioinformatics analysis, 
            please feel free to ask specific questions about viral proteins, transmission 
            mechanisms, or computational analysis methods.
            """
            response += padding
        return response


class MockSessionManager:
    """Mock session manager for testing session handling"""
    
    def __init__(self):
        self.sessions = {}
        
    async def get_or_create_session(self, session_id: str) -> Dict[str, Any]:
        """Get or create a mock session"""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "session_id": session_id,
                "created_at": time.time(),
                "message_count": 0,
                "last_activity": time.time()
            }
        return self.sessions[session_id]
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Update session data"""
        if session_id in self.sessions:
            self.sessions[session_id].update(data)
            self.sessions[session_id]["last_activity"] = time.time()
    
    def get_session_count(self) -> int:
        """Get number of active sessions"""
        return len(self.sessions)


# Mock service factory functions
def create_mock_bvbrc_service(**kwargs) -> MockBVBRCService:
    """Factory function for mock BV-BRC service"""
    return MockBVBRCService(**kwargs)

def create_mock_external_tools(**kwargs) -> MockExternalTools:
    """Factory function for mock external tools"""
    return MockExternalTools(**kwargs)

def create_mock_workflow_components() -> MockWorkflowComponents:
    """Factory function for mock workflow components"""
    return MockWorkflowComponents()

def create_mock_session_manager() -> MockSessionManager:
    """Factory function for mock session manager"""
    return MockSessionManager()


# Error simulation utilities
class MockServiceError(Exception):
    """Custom exception for mock service errors"""
    pass

class MockTimeoutError(Exception):
    """Custom exception for mock timeout errors"""
    pass

class MockRateLimitError(Exception):
    """Custom exception for mock rate limit errors"""
    pass 