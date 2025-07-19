"""
Chatbot Viral Integration Workflow Steps

Steps for the chatbot-viral annotation integration workflow following NanoBrain patterns.
ALL steps must be registered here for framework discovery.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.2.0
"""

from .query_classification_step import QueryClassificationStep
from .virus_name_resolution_step import VirusNameResolutionStep
from .annotation_job_step import AnnotationJobStep
from .conversational_response_step import ConversationalResponseStep
from .response_formatting_step import ResponseFormattingStep

# Cross-workflow imports for integrated functionality
from nanobrain.library.workflows.viral_protein_analysis.steps.bv_brc_data_acquisition_step import BVBRCDataAcquisitionStep
from nanobrain.library.workflows.viral_protein_analysis.steps.annotation_mapping_step import AnnotationMappingStep

__all__ = [
    'QueryClassificationStep',
    'VirusNameResolutionStep',
    'AnnotationJobStep', 
    'ConversationalResponseStep',
    'ResponseFormattingStep',
    # Cross-workflow integrations
    'BVBRCDataAcquisitionStep',
    'AnnotationMappingStep'
] 