"""
Processors module - Integration with NemoCurator and DataJuicer
"""

from data_pipeline_framework.processors.nemo_processor import NemoProcessor
from data_pipeline_framework.processors.datajuicer_processor import DataJuicerProcessor

__all__ = ["NemoProcessor", "DataJuicerProcessor"]
