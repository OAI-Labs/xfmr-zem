"""
Pre-built pipelines and factory for domain-specific pipelines
"""

from xfmr_zem.pipelines.legal_pipeline import LegalDataPipeline
from xfmr_zem.pipelines.text_pipeline import TextCleaningPipeline
from xfmr_zem.pipelines.domain_pipeline_factory import (
    DomainPipelineFactory,
    create_domain_pipeline,
)

__all__ = [
    "LegalDataPipeline",
    "TextCleaningPipeline",
    "DomainPipelineFactory",
    "create_domain_pipeline",
]
