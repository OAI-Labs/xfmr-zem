"""
Pre-built pipelines and factory for domain-specific pipelines
"""

from data_pipeline_framework.pipelines.legal_pipeline import LegalDataPipeline
from data_pipeline_framework.pipelines.text_pipeline import TextCleaningPipeline
from data_pipeline_framework.pipelines.domain_pipeline_factory import (
    DomainPipelineFactory,
    create_domain_pipeline,
)

__all__ = [
    "LegalDataPipeline",
    "TextCleaningPipeline",
    "DomainPipelineFactory",
    "create_domain_pipeline",
]
