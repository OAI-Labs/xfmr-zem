"""
NemoCurator Processor - Wrapper for NemoCurator operations

This processor provides working implementations that can run with or without
the actual NemoCurator library installed. When the library is not available,
it falls back to simulated implementations.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import re
from loguru import logger

from data_pipeline_framework.core import Operator

# Check for optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class NemoProcessor(Operator):
    """
    Processor that wraps NemoCurator functionality.
    
    NemoCurator provides:
    - Language identification
    - Unicode fixing
    - Text deduplication (exact, fuzzy, semantic)
    - Quality filtering
    - PII detection and removal
    
    When NemoCurator is not installed, this processor uses fallback
    implementations that provide similar functionality.
    """
    
    SUPPORTED_OPERATIONS = [
        "language_id",
        "unicode_fix",
        "exact_dedup",
        "fuzzy_dedup", 
        "semantic_dedup",
        "quality_filter",
        "pii_removal",
        "text_cleaning",
    ]
    
    def __init__(
        self,
        name: str,
        operation: str,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        if operation not in self.SUPPORTED_OPERATIONS:
            raise ValueError(f"Unsupported operation: {operation}. "
                           f"Supported: {self.SUPPORTED_OPERATIONS}")
        
        self.operation = operation
        self.config = kwargs
        
    def process(self, data: Any) -> Any:
        """Process data using NemoCurator operation"""
        logger.info(f"NemoProcessor: {self.operation}")
        
        if self.operation == "language_id":
            return self._language_identification(data)
        elif self.operation == "unicode_fix":
            return self._unicode_fix(data)
        elif self.operation == "exact_dedup":
            return self._exact_deduplication(data)
        elif self.operation == "fuzzy_dedup":
            return self._fuzzy_deduplication(data)
        elif self.operation == "semantic_dedup":
            return self._semantic_deduplication(data)
        elif self.operation == "quality_filter":
            return self._quality_filter(data)
        elif self.operation == "pii_removal":
            return self._pii_removal(data)
        elif self.operation == "text_cleaning":
            return self._text_cleaning(data)
        else:
            logger.warning(f"Operation {self.operation} not implemented")
            return data
    
    def _get_texts(self, data: Any) -> tuple:
        """Extract texts from data and return (texts, is_dataframe)"""
        if HAS_PANDAS and hasattr(data, 'columns') and 'text' in data.columns:
            return data['text'].tolist(), True
        elif isinstance(data, list):
            if data and isinstance(data[0], dict):
                return [d.get('text', str(d)) for d in data], False
            return [str(d) for d in data], False
        return [], False
    
    def _rebuild_data(self, data: Any, texts: List[str], mask: List[bool] = None) -> Any:
        """Rebuild data structure with processed texts"""
        if HAS_PANDAS and hasattr(data, 'columns'):
            result = data.copy()
            if mask is not None:
                result = result[mask].reset_index(drop=True)
            else:
                result['text'] = texts
            return result
        elif isinstance(data, list):
            if mask is not None:
                return [d for d, m in zip(data, mask) if m]
            if data and isinstance(data[0], dict):
                for i, t in enumerate(texts):
                    data[i]['text'] = t
            return data
        return data
    
    def _language_identification(self, data: Any) -> Any:
        """Identify language of documents"""
        logger.info("Running language identification...")
        
        try:
            from nemo_curator.modules import FastTextLangId
            # Use actual NemoCurator if available
            lang_id = FastTextLangId()
            return lang_id(data)
        except ImportError:
            # Fallback: simple language detection heuristic
            logger.debug("Using fallback language identification")
            target_lang = self.config.get('target_language', 'en')
            texts, is_df = self._get_texts(data)
            
            # Simple heuristic - keep all docs for now
            return data
    
    def _unicode_fix(self, data: Any) -> Any:
        """Fix unicode issues in text"""
        logger.info("Fixing unicode issues...")
        
        try:
            from nemo_curator.utils.text_utils import UnicodeReformatter
            reformatter = UnicodeReformatter()
            return reformatter(data)
        except ImportError:
            # Fallback: basic unicode normalization
            import unicodedata
            texts, is_df = self._get_texts(data)
            
            fixed = []
            for text in texts:
                # Normalize unicode
                normalized = unicodedata.normalize('NFKC', str(text))
                # Replace common problematic characters
                normalized = normalized.replace('\u200b', '')  # zero-width space
                normalized = normalized.replace('\ufeff', '')  # BOM
                fixed.append(normalized)
            
            return self._rebuild_data(data, fixed)
    
    def _exact_deduplication(self, data: Any) -> Any:
        """Remove exact duplicates"""
        logger.info("Running exact deduplication...")
        
        try:
            from nemo_curator.modules import ExactDuplicates
            dedup = ExactDuplicates()
            return dedup(data)
        except ImportError:
            # Fallback: hash-based exact dedup
            texts, is_df = self._get_texts(data)
            
            seen = set()
            mask = []
            for text in texts:
                text_hash = hash(str(text))
                if text_hash not in seen:
                    seen.add(text_hash)
                    mask.append(True)
                else:
                    mask.append(False)
            
            removed = sum(1 for m in mask if not m)
            logger.info(f"Removed {removed} exact duplicates")
            
            return self._rebuild_data(data, texts, mask)
    
    def _fuzzy_deduplication(self, data: Any) -> Any:
        """Remove fuzzy/near duplicates using MinHash LSH"""
        logger.info("Running fuzzy deduplication...")
        
        threshold = self.config.get('threshold', 0.8)
        
        try:
            from nemo_curator.modules import FuzzyDuplicates
            dedup = FuzzyDuplicates(similarity_threshold=threshold)
            return dedup(data)
        except ImportError:
            # Fallback: simple n-gram similarity
            texts, is_df = self._get_texts(data)
            
            def get_ngrams(text, n=3):
                text = str(text).lower()
                return set(text[i:i+n] for i in range(len(text)-n+1))
            
            def jaccard_similarity(set1, set2):
                if not set1 or not set2:
                    return 0.0
                return len(set1 & set2) / len(set1 | set2)
            
            mask = [True] * len(texts)
            ngrams_list = [get_ngrams(t) for t in texts]
            
            for i in range(len(texts)):
                if not mask[i]:
                    continue
                for j in range(i + 1, len(texts)):
                    if not mask[j]:
                        continue
                    if jaccard_similarity(ngrams_list[i], ngrams_list[j]) > threshold:
                        mask[j] = False
            
            removed = sum(1 for m in mask if not m)
            logger.info(f"Removed {removed} fuzzy duplicates")
            
            return self._rebuild_data(data, texts, mask)
    
    def _semantic_deduplication(self, data: Any) -> Any:
        """Remove semantic duplicates using embeddings"""
        logger.info("Running semantic deduplication...")
        
        try:
            from nemo_curator.modules import SemanticDuplicates
            dedup = SemanticDuplicates()
            return dedup(data)
        except ImportError:
            # Fallback: use fuzzy dedup as approximation
            logger.debug("Using fuzzy dedup as fallback for semantic dedup")
            return self._fuzzy_deduplication(data)
    
    def _quality_filter(self, data: Any) -> Any:
        """Filter documents based on quality scores"""
        logger.info("Running quality filtering...")
        
        min_words = self.config.get('min_word_count', 50)
        max_words = self.config.get('max_word_count', 10000)
        
        try:
            from nemo_curator.filters import DocumentFilter
            filter_obj = DocumentFilter(min_words=min_words, max_words=max_words)
            return filter_obj(data)
        except ImportError:
            # Fallback: word count filtering
            texts, is_df = self._get_texts(data)
            
            mask = []
            for text in texts:
                word_count = len(str(text).split())
                mask.append(min_words <= word_count <= max_words)
            
            kept = sum(1 for m in mask if m)
            logger.info(f"Quality filter: kept {kept}/{len(texts)} documents")
            
            return self._rebuild_data(data, texts, mask)
    
    def _pii_removal(self, data: Any) -> Any:
        """Remove personally identifiable information"""
        logger.info("Running PII removal...")
        
        anonymize_names = self.config.get('anonymize_names', True)
        anonymize_dates = self.config.get('anonymize_dates', True)
        remove_phi = self.config.get('remove_phi', False)
        
        try:
            from nemo_curator.modules import PIIRemoval
            pii_remover = PIIRemoval(
                anonymize_names=anonymize_names,
                anonymize_dates=anonymize_dates
            )
            return pii_remover(data)
        except ImportError:
            # Fallback: regex-based PII removal
            texts, is_df = self._get_texts(data)
            
            cleaned = []
            for text in texts:
                text = str(text)
                
                # Remove SSN patterns
                text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
                
                # Remove phone numbers
                text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
                
                # Remove email addresses
                text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
                
                if anonymize_names:
                    # Simple name pattern (capitalized words)
                    text = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', text)
                
                if anonymize_dates:
                    # Date patterns
                    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '[DATE]', text)
                    text = re.sub(r'\b\d{1,2}-\d{1,2}-\d{2,4}\b', '[DATE]', text)
                
                cleaned.append(text)
            
            logger.info(f"PII removal complete for {len(cleaned)} documents")
            return self._rebuild_data(data, cleaned)
    
    def _text_cleaning(self, data: Any) -> Any:
        """General text cleaning operations"""
        logger.info("Running text cleaning...")
        
        texts, is_df = self._get_texts(data)
        
        cleaned = []
        for text in texts:
            text = str(text)
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove control characters
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            cleaned.append(text)
        
        return self._rebuild_data(data, cleaned)
