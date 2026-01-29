"""
Legal Data Pipeline - Specialized pipeline for Vietnamese legal documents
"""

from typing import Optional
from loguru import logger

from xfmr_zem.core import Pipeline
from xfmr_zem.processors import NemoProcessor, DataJuicerProcessor


class LegalDataPipeline(Pipeline):
    """
    Pre-configured pipeline for processing Vietnamese legal documents.
    
    Steps:
    1. Unicode normalization
    2. Text cleaning (remove headers, footers, noise)
    3. Language filtering (Vietnamese)
    4. Quality filtering
    5. Deduplication (exact + fuzzy)
    6. Legal-specific processing
    """
    
    def __init__(
        self,
        name: str = "legal-data-pipeline",
        enable_dedup: bool = True,
        enable_quality_filter: bool = True,
        min_text_length: int = 100,
        language: str = "vi",
    ):
        super().__init__(
            name=name,
            description="Pipeline for Vietnamese legal document processing",
            domain="legal"
        )
        
        self.enable_dedup = enable_dedup
        self.enable_quality_filter = enable_quality_filter
        self.min_text_length = min_text_length
        self.language = language
        
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the pipeline with pre-configured steps"""
        
        # Step 1: Unicode normalization
        self.add_operator(
            name="unicode_normalization",
            operator=NemoProcessor(
                name="unicode_fix",
                operation="unicode_fix"
            )
        )
        
        # Step 2: Text cleaning
        self.add_operator(
            name="text_cleaning",
            operator=DataJuicerProcessor(
                name="clean_text",
                category="mapper",
                operator_name="clean_html"
            )
        )
        
        # Step 3: Remove headers/footers (legal docs often have page numbers, etc.)
        self.add_operator(
            name="remove_header_footer",
            operator=DataJuicerProcessor(
                name="remove_hf",
                category="mapper", 
                operator_name="remove_header_footer"
            )
        )
        
        # Step 4: Language filter
        self.add_operator(
            name="language_filter",
            operator=NemoProcessor(
                name="lang_id",
                operation="language_id",
                target_language=self.language
            )
        )
        
        # Step 5: Text length filter
        self.add_operator(
            name="length_filter",
            operator=DataJuicerProcessor(
                name="text_length",
                category="filter",
                operator_name="text_length_filter",
                min_length=self.min_text_length
            ),
            enabled=self.enable_quality_filter
        )
        
        # Step 6: Quality filter
        self.add_operator(
            name="quality_filter",
            operator=NemoProcessor(
                name="quality",
                operation="quality_filter"
            ),
            enabled=self.enable_quality_filter
        )
        
        # Step 7: Exact deduplication
        self.add_operator(
            name="exact_dedup",
            operator=NemoProcessor(
                name="exact_dedup",
                operation="exact_dedup"
            ),
            enabled=self.enable_dedup
        )
        
        # Step 8: Fuzzy deduplication
        self.add_operator(
            name="fuzzy_dedup",
            operator=NemoProcessor(
                name="fuzzy_dedup",
                operation="fuzzy_dedup",
                similarity_threshold=0.8
            ),
            enabled=self.enable_dedup
        )
        
        logger.info(f"Built {self.name} with {len(self.steps)} steps")
