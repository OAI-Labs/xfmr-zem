"""
General Text Cleaning Pipeline
"""

from xfmr_zem.core import Pipeline
from xfmr_zem.processors import NemoProcessor, DataJuicerProcessor


class TextCleaningPipeline(Pipeline):
    """
    General-purpose text cleaning pipeline.
    
    Suitable for: news articles, social media, general web content.
    """
    
    def __init__(
        self,
        name: str = "text-cleaning-pipeline",
        enable_dedup: bool = True,
        language: str = "vi",
    ):
        super().__init__(
            name=name,
            description="General text cleaning and processing pipeline",
            domain="general"
        )
        
        self._build_pipeline(enable_dedup, language)
    
    def _build_pipeline(self, enable_dedup: bool, language: str):
        """Build the pipeline"""
        
        # Unicode fix
        self.add_operator(
            name="unicode_fix",
            operator=NemoProcessor(name="unicode", operation="unicode_fix")
        )
        
        # Clean HTML
        self.add_operator(
            name="clean_html",
            operator=DataJuicerProcessor(
                name="html", category="mapper", operator_name="clean_html"
            )
        )
        
        # Clean links
        self.add_operator(
            name="clean_links",
            operator=DataJuicerProcessor(
                name="links", category="mapper", operator_name="clean_links"
            )
        )
        
        # Normalize whitespace
        self.add_operator(
            name="normalize_whitespace",
            operator=DataJuicerProcessor(
                name="whitespace", category="mapper", operator_name="whitespace_normalization"
            )
        )
        
        # Language filter
        self.add_operator(
            name="language_filter",
            operator=NemoProcessor(
                name="lang", operation="language_id", target_language=language
            )
        )
        
        # Deduplication
        self.add_operator(
            name="deduplication",
            operator=NemoProcessor(name="dedup", operation="exact_dedup"),
            enabled=enable_dedup
        )
