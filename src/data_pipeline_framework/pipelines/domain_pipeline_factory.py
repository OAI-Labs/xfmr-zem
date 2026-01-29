"""
Domain Pipeline Factory - Create pipelines from YAML configurations

This module provides a factory class that creates pipelines dynamically from
YAML configuration files. Adding a new domain requires only creating a new
YAML config file, no coding required.
"""

from typing import Any, Dict, List, Optional
from loguru import logger

from data_pipeline_framework.core import Pipeline
from data_pipeline_framework.processors import NemoProcessor, DataJuicerProcessor
from data_pipeline_framework.utils.config_loader import ConfigLoader, DomainConfig


class DomainPipelineFactory:
    """
    Factory class to create pipelines from YAML domain configurations.
    
    This enables a config-driven architecture where new domains can be added
    simply by creating a new YAML configuration file.
    
    Example:
        # Create pipeline for any domain defined in configs/domains/
        pipeline = DomainPipelineFactory.create("medical")
        pipeline = DomainPipelineFactory.create("finance")
        pipeline = DomainPipelineFactory.create("custom_domain")  # From custom YAML
        
        # Run the pipeline
        result = pipeline.run(input_data)
    """
    
    # Mapping from config step names to processor operations
    NEMO_OPERATION_MAP = {
        "pii_removal": "pii_removal",
        "deduplication": "fuzzy_dedup",
        "exact_dedup": "exact_dedup",
        "fuzzy_dedup": "fuzzy_dedup",
        "semantic_dedup": "semantic_dedup",
        "quality_filtering": "quality_filter",
        "quality_filter": "quality_filter",
        "language_id": "language_id",
        "unicode_fix": "unicode_fix",
        "text_cleaning": "text_cleaning",
    }
    
    DATAJUICER_CATEGORY_MAP = {
        "language_filter": "filter",
        "perplexity_filter": "filter",
        "text_length_filter": "filter",
        "word_num_filter": "filter",
        "special_char_filter": "filter",
        "medical_terminology_filter": "filter",
        "legal_citation_extractor": "mapper",
        "financial_entity_extractor": "mapper",
        "document_structure_filter": "filter",
        "numerical_accuracy_filter": "filter",
        "clean_html": "mapper",
        "clean_links": "mapper",
        "fix_unicode": "mapper",
        "remove_header_footer": "mapper",
    }
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize the factory.
        
        Args:
            config_dir: Optional path to configs directory. If not provided,
                       uses the package's default configs directory.
        """
        self.config_loader = ConfigLoader(config_dir)
    
    @classmethod
    def create(cls, domain: str, config_dir: Optional[str] = None) -> Pipeline:
        """
        Create a pipeline from domain configuration.
        
        Args:
            domain: Name of the domain (must have a corresponding YAML config)
            config_dir: Optional path to configs directory
            
        Returns:
            Configured Pipeline ready to run
            
        Raises:
            ValueError: If domain config not found
        """
        factory = cls(config_dir)
        return factory.build_pipeline(domain)
    
    @classmethod
    def list_domains(cls, config_dir: Optional[str] = None) -> List[str]:
        """List all available domain configurations."""
        factory = cls(config_dir)
        return factory.config_loader.list_available_domains()
    
    def build_pipeline(self, domain: str) -> Pipeline:
        """
        Build a pipeline from domain configuration.
        
        Args:
            domain: Name of the domain
            
        Returns:
            Configured Pipeline
        """
        config = self.config_loader.load_domain_config(domain)
        
        pipeline = Pipeline(
            name=f"{domain}-data-pipeline",
            description=config.description,
            domain=domain
        )
        
        # Store config in pipeline for reference
        pipeline.config = config
        
        # Add NeMo Curator steps
        self._add_nemo_steps(pipeline, config)
        
        # Add DataJuicer steps
        self._add_datajuicer_steps(pipeline, config)
        
        logger.info(
            f"Built pipeline '{pipeline.name}' with {len(pipeline.steps)} steps"
        )
        
        return pipeline
    
    def _add_nemo_steps(self, pipeline: Pipeline, config: DomainConfig) -> None:
        """Add NeMo Curator steps to the pipeline."""
        nemo_steps = config.steps.get('nemo_curator', [])
        
        for step_config in nemo_steps:
            if not step_config.get('enabled', True):
                logger.debug(f"Skipping disabled step: {step_config['name']}")
                continue
            
            step_name = step_config['name']
            operation = self.NEMO_OPERATION_MAP.get(step_name, step_name)
            params = step_config.get('params', {})
            
            try:
                processor = NemoProcessor(
                    name=step_name,
                    operation=operation,
                    **params
                )
                pipeline.add_operator(
                    name=f"nemo_{step_name}",
                    operator=processor,
                    enabled=True
                )
            except ValueError as e:
                logger.warning(f"Could not add NeMo step '{step_name}': {e}")
    
    def _add_datajuicer_steps(self, pipeline: Pipeline, config: DomainConfig) -> None:
        """Add DataJuicer steps to the pipeline."""
        dj_steps = config.steps.get('datajuicer', [])
        
        for step_config in dj_steps:
            if not step_config.get('enabled', True):
                logger.debug(f"Skipping disabled step: {step_config['name']}")
                continue
            
            step_name = step_config['name']
            category = self.DATAJUICER_CATEGORY_MAP.get(step_name, 'filter')
            params = step_config.get('params', {})
            
            try:
                processor = DataJuicerProcessor(
                    name=step_name,
                    category=category,
                    operator_name=step_name,
                    **params
                )
                pipeline.add_operator(
                    name=f"dj_{step_name}",
                    operator=processor,
                    enabled=True
                )
            except ValueError as e:
                logger.warning(f"Could not add DataJuicer step '{step_name}': {e}")


def create_domain_pipeline(domain: str, config_dir: Optional[str] = None) -> Pipeline:
    """
    Convenience function to create a domain pipeline.
    
    Args:
        domain: Name of the domain
        config_dir: Optional path to configs directory
        
    Returns:
        Configured Pipeline
        
    Example:
        pipeline = create_domain_pipeline("medical")
        result = pipeline.run(data)
    """
    return DomainPipelineFactory.create(domain, config_dir)
