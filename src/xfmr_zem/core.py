"""
Core module - Base classes for pipeline components
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class StepConfig:
    """Configuration for a pipeline step"""
    name: str
    operator: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class PipelineConfig:
    """Configuration for a complete pipeline"""
    name: str
    description: str = ""
    domain: str = "general"  # legal, medical, general, etc.
    steps: List[StepConfig] = field(default_factory=list)
    

class Operator(ABC):
    """Base class for data processing operators"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.params = kwargs
        
    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return output"""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"


class Step:
    """A single step in the pipeline"""
    
    def __init__(self, name: str, operator: Operator, enabled: bool = True):
        self.name = name
        self.operator = operator
        self.enabled = enabled
        
    def run(self, data: Any) -> Any:
        """Execute the step"""
        if not self.enabled:
            logger.info(f"Step {self.name} is disabled, skipping...")
            return data
            
        logger.info(f"Running step: {self.name}")
        result = self.operator.process(data)
        logger.info(f"Step {self.name} completed")
        return result


class Pipeline:
    """
    Main pipeline class that orchestrates data processing steps.
    Integrates with ZenML for visualization and tracking.
    """
    
    def __init__(self, name: str, description: str = "", domain: str = "general"):
        self.name = name
        self.description = description
        self.domain = domain
        self.steps: List[Step] = []
        
    def add_step(self, step: Step) -> "Pipeline":
        """Add a step to the pipeline"""
        self.steps.append(step)
        return self
    
    def add_operator(self, name: str, operator: Operator, enabled: bool = True) -> "Pipeline":
        """Convenience method to add an operator as a step"""
        step = Step(name=name, operator=operator, enabled=enabled)
        return self.add_step(step)
    
    def run(self, data: Any) -> Any:
        """Execute all steps in the pipeline"""
        logger.info(f"Starting pipeline: {self.name} (domain: {self.domain})")
        logger.info(f"Total steps: {len(self.steps)}")
        
        result = data
        for i, step in enumerate(self.steps, 1):
            logger.info(f"[{i}/{len(self.steps)}] {step.name}")
            result = step.run(result)
            
        logger.info(f"Pipeline {self.name} completed successfully")
        return result
    
    def to_zenml_pipeline(self):
        """Convert to ZenML pipeline for orchestration"""
        try:
            from zenml import pipeline, step
            
            @pipeline(name=self.name)
            def zenml_pipeline(input_data):
                result = input_data
                for s in self.steps:
                    if s.enabled:
                        result = s.operator.process(result)
                return result
            
            return zenml_pipeline
        except ImportError:
            logger.warning("ZenML not installed, returning None")
            return None
    
    def __repr__(self):
        return f"Pipeline(name={self.name}, steps={len(self.steps)})"
