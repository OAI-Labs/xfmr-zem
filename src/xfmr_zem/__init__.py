"""
Zem
===

A unified data pipeline framework combining:
- ZenML: Orchestration and visualization
- NeMo Curator: Data curation and processing
- DataJuicer: Data processing operators

Designed for multi-domain data processing (legal, medical, finance, and custom domains).

Config-Driven Architecture:
    Add a new domain by simply creating a YAML config file in configs/domains/
    
Example:
    from xfmr_zem import create_domain_pipeline
    
    # Create pipeline from YAML config - no class needed!
    pipeline = create_domain_pipeline("medical")
    result = pipeline.run(data)
"""

__version__ = "0.1.0"
__author__ = "Khai Hoang"

from xfmr_zem.client import PipelineClient

__all__ = [
    "PipelineClient",
]
