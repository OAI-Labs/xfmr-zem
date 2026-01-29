"""
Configuration loader for domain-specific pipelines
"""
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class DomainConfig:
    """Domain configuration dataclass"""
    name: str
    description: str
    parameters: Dict[str, Any]
    steps: Dict[str, Any]
    quality_metrics: List[str]
    special_handling: Dict[str, Any]


class ConfigLoader:
    """Load and merge configurations for domain-specific pipelines"""
    
    def __init__(self, config_dir: Optional[str] = None):
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Auto-resolve to package configs directory
            self.config_dir = Path(__file__).parent.parent / "configs"
        
        self._base_config = None
    
    @property
    def base_config(self) -> Dict[str, Any]:
        """Lazy load base configuration"""
        if self._base_config is None:
            self._base_config = self._load_base_config()
        return self._base_config
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration"""
        base_path = self.config_dir / "base_config.yaml"
        if not base_path.exists():
            logger.warning(f"Base config not found at {base_path}, using defaults")
            return {
                "pipeline": {"name": "data_processing_pipeline", "version": "1.0.0"},
                "common": {"batch_size": 1000, "num_workers": 4},
                "logging": {"level": "INFO"},
            }
        
        with open(base_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_domain_config(self, domain: str) -> DomainConfig:
        """Load domain-specific configuration"""
        domain_path = self.config_dir / "domains" / f"{domain}.yaml"
        
        if not domain_path.exists():
            available = self.list_available_domains()
            raise ValueError(
                f"Domain config not found: '{domain}'. "
                f"Available domains: {available}"
            )
        
        with open(domain_path, 'r') as f:
            domain_config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration for domain: {domain}")
        
        return DomainConfig(
            name=domain_config['domain']['name'],
            description=domain_config['domain']['description'],
            parameters=domain_config.get('parameters', {}),
            steps=domain_config.get('steps', {}),
            quality_metrics=domain_config.get('quality_metrics', []),
            special_handling=domain_config.get('special_handling', {})
        )
    
    def get_merged_config(self, domain: str) -> Dict[str, Any]:
        """Get merged configuration (base + domain-specific)"""
        domain_config = self.load_domain_config(domain)
        
        merged = {
            **self.base_config,
            'domain': {
                'name': domain_config.name,
                'description': domain_config.description,
                'parameters': domain_config.parameters,
                'steps': domain_config.steps,
                'quality_metrics': domain_config.quality_metrics,
                'special_handling': domain_config.special_handling
            }
        }
        
        return merged
    
    def list_available_domains(self) -> List[str]:
        """List all available domain configurations"""
        domains_dir = self.config_dir / "domains"
        if not domains_dir.exists():
            return []
        return sorted([f.stem for f in domains_dir.glob("*.yaml")])
    
    def get_pipeline_params(self, domain: str) -> Dict[str, Any]:
        """Extract pipeline parameters for a domain"""
        config = self.get_merged_config(domain)
        return {
            'domain': domain,
            **config['domain']['parameters'],
            'common': config.get('common', {})
        }


def compare_domain_configs(domain1: str, domain2: str) -> Dict[str, Any]:
    """Compare configurations between two domains"""
    loader = ConfigLoader()
    config1 = loader.load_domain_config(domain1)
    config2 = loader.load_domain_config(domain2)
    
    comparison = {
        'domains': [domain1, domain2],
        'parameter_diff': {},
        'steps_diff': {
            f'only_in_{domain1}': [],
            f'only_in_{domain2}': [],
            'common': []
        }
    }
    
    # Compare parameters
    all_params = set(config1.parameters.keys()) | set(config2.parameters.keys())
    for param in all_params:
        val1 = config1.parameters.get(param)
        val2 = config2.parameters.get(param)
        if val1 != val2:
            comparison['parameter_diff'][param] = {
                domain1: val1,
                domain2: val2
            }
    
    # Compare steps
    steps1 = set(
        s['name'] for s in config1.steps.get('nemo_curator', []) + 
        config1.steps.get('datajuicer', [])
    )
    steps2 = set(
        s['name'] for s in config2.steps.get('nemo_curator', []) + 
        config2.steps.get('datajuicer', [])
    )
    
    comparison['steps_diff'][f'only_in_{domain1}'] = list(steps1 - steps2)
    comparison['steps_diff'][f'only_in_{domain2}'] = list(steps2 - steps1)
    comparison['steps_diff']['common'] = list(steps1 & steps2)
    
    return comparison
