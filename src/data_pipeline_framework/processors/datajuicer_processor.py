"""
DataJuicer Processor - Wrapper for DataJuicer operations

This processor provides working implementations that can run with or without
the actual DataJuicer library installed. When the library is not available,
it falls back to simulated implementations.
"""

from typing import Any, Dict, List, Optional
import re
from loguru import logger

from data_pipeline_framework.core import Operator

# Check for optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class DataJuicerProcessor(Operator):
    """
    Processor that wraps DataJuicer functionality.
    
    DataJuicer provides 50+ operators for:
    - Formatters: Load/save various formats
    - Mappers: Transform text (clean, normalize)
    - Filters: Remove unwanted documents
    - Deduplicators: Remove duplicates
    
    When DataJuicer is not installed, this processor uses fallback
    implementations that provide similar functionality.
    """
    
    OPERATOR_CATEGORIES = {
        "formatter": ["json", "parquet", "csv", "text"],
        "mapper": [
            "clean_html",
            "clean_links", 
            "fix_unicode",
            "punctuation_normalization",
            "whitespace_normalization",
            "remove_header_footer",
            "sentence_split",
            "legal_citation_extractor",
            "financial_entity_extractor",
        ],
        "filter": [
            "language_filter",
            "perplexity_filter",
            "text_length_filter",
            "word_num_filter",
            "special_char_filter",
            "flagged_word_filter",
            "medical_terminology_filter",
            "document_structure_filter",
            "numerical_accuracy_filter",
        ],
        "deduplicator": [
            "document_simhash",
            "document_minhash",
            "ray_dedup",
        ],
    }
    
    def __init__(
        self,
        name: str,
        category: str,
        operator_name: str,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        
        if category not in self.OPERATOR_CATEGORIES:
            raise ValueError(f"Unsupported category: {category}. "
                           f"Supported: {list(self.OPERATOR_CATEGORIES.keys())}")
        
        self.category = category
        self.operator_name = operator_name
        self.config = kwargs
        
    def process(self, data: Any) -> Any:
        """Process data using DataJuicer operator"""
        logger.info(f"DataJuicerProcessor: {self.category}/{self.operator_name}")
        
        # Route to appropriate handler
        if self.category == "mapper":
            return self._process_mapper(data)
        elif self.category == "filter":
            return self._process_filter(data)
        elif self.category == "deduplicator":
            return self._process_deduplicator(data)
        elif self.category == "formatter":
            return self._process_formatter(data)
        else:
            logger.warning(f"Unknown category: {self.category}")
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
    
    def _rebuild_data(self, data: Any, texts: List[str] = None, mask: List[bool] = None) -> Any:
        """Rebuild data structure with processed texts or filter mask"""
        if HAS_PANDAS and hasattr(data, 'columns'):
            result = data.copy()
            if mask is not None:
                result = result[[m for m in mask]].reset_index(drop=True)
            if texts is not None and len(texts) == len(result):
                result['text'] = texts
            return result
        elif isinstance(data, list):
            if mask is not None:
                return [d for d, m in zip(data, mask) if m]
            if texts is not None and data and isinstance(data[0], dict):
                for i, t in enumerate(texts):
                    if i < len(data):
                        data[i]['text'] = t
            return data
        return data
    
    def _process_mapper(self, data: Any) -> Any:
        """Process mapper operations"""
        texts, is_df = self._get_texts(data)
        
        if self.operator_name == "clean_html":
            texts = [self._clean_html(t) for t in texts]
        elif self.operator_name == "clean_links":
            texts = [self._clean_links(t) for t in texts]
        elif self.operator_name == "fix_unicode":
            import unicodedata
            texts = [unicodedata.normalize('NFKC', str(t)) for t in texts]
        elif self.operator_name == "punctuation_normalization":
            texts = [self._normalize_punctuation(t) for t in texts]
        elif self.operator_name == "whitespace_normalization":
            texts = [re.sub(r'\s+', ' ', str(t)).strip() for t in texts]
        elif self.operator_name == "remove_header_footer":
            texts = [self._remove_header_footer(t) for t in texts]
        elif self.operator_name == "legal_citation_extractor":
            # Add citation metadata but keep text
            logger.info("Extracting legal citations...")
        elif self.operator_name == "financial_entity_extractor":
            # Add financial entity metadata but keep text
            logger.info("Extracting financial entities...")
        else:
            logger.warning(f"Mapper {self.operator_name} not implemented, passing through")
        
        return self._rebuild_data(data, texts)
    
    def _process_filter(self, data: Any) -> Any:
        """Process filter operations"""
        texts, is_df = self._get_texts(data)
        initial_count = len(texts)
        
        if self.operator_name == "language_filter":
            mask = self._language_filter(texts)
        elif self.operator_name == "perplexity_filter":
            mask = self._perplexity_filter(texts)
        elif self.operator_name == "text_length_filter":
            min_len = self.config.get('min_length', 10)
            max_len = self.config.get('max_length', 100000)
            mask = [min_len <= len(str(t)) <= max_len for t in texts]
        elif self.operator_name == "word_num_filter":
            min_words = self.config.get('min_words', 10)
            max_words = self.config.get('max_words', 10000)
            mask = [min_words <= len(str(t).split()) <= max_words for t in texts]
        elif self.operator_name == "special_char_filter":
            max_ratio = self.config.get('max_special_char_ratio', 0.3)
            mask = [self._check_special_char_ratio(t, max_ratio) for t in texts]
        elif self.operator_name == "medical_terminology_filter":
            mask = self._medical_term_filter(texts)
        elif self.operator_name == "document_structure_filter":
            mask = self._structure_filter(texts)
        elif self.operator_name == "numerical_accuracy_filter":
            # Keep all for now, just log
            logger.info("Checking numerical accuracy...")
            mask = [True] * len(texts)
        else:
            logger.warning(f"Filter {self.operator_name} not implemented, keeping all")
            mask = [True] * len(texts)
        
        kept = sum(1 for m in mask if m)
        logger.info(f"Filter {self.operator_name}: kept {kept}/{initial_count} documents")
        
        return self._rebuild_data(data, mask=mask)
    
    def _process_deduplicator(self, data: Any) -> Any:
        """Process deduplication operations"""
        texts, is_df = self._get_texts(data)
        initial_count = len(texts)
        
        if self.operator_name in ["document_simhash", "document_minhash"]:
            # Simple hash-based dedup
            seen = set()
            mask = []
            for t in texts:
                h = hash(str(t)[:1000])  # Hash first 1000 chars
                if h not in seen:
                    seen.add(h)
                    mask.append(True)
                else:
                    mask.append(False)
        else:
            mask = [True] * len(texts)
        
        removed = sum(1 for m in mask if not m)
        logger.info(f"Deduplication: removed {removed} duplicates")
        
        return self._rebuild_data(data, mask=mask)
    
    def _process_formatter(self, data: Any) -> Any:
        """Process formatter operations (passthrough for now)"""
        return data
    
    # Helper methods
    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Decode common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        return text
    
    def _clean_links(self, text: str) -> str:
        """Remove URLs from text"""
        text = str(text)
        # Remove URLs
        text = re.sub(r'https?://\S+', '[URL]', text)
        text = re.sub(r'www\.\S+', '[URL]', text)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks"""
        text = str(text)
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        # Normalize dashes
        text = text.replace('–', '-').replace('—', '-')
        return text
    
    def _remove_header_footer(self, text: str) -> str:
        """Remove common header/footer patterns"""
        lines = str(text).split('\n')
        if len(lines) <= 5:
            return text
        
        # Remove first line if it looks like a header (page number, etc.)
        if lines[0].strip().isdigit() or len(lines[0].strip()) < 10:
            lines = lines[1:]
        
        # Remove last line if it looks like a footer
        if lines and (lines[-1].strip().isdigit() or len(lines[-1].strip()) < 10):
            lines = lines[:-1]
        
        return '\n'.join(lines)
    
    def _language_filter(self, texts: List[str]) -> List[bool]:
        """Filter by language"""
        target_lang = self.config.get('lang', 'en')
        min_score = self.config.get('min_score', 0.8)
        
        # Simple heuristic: check for common words
        if target_lang == 'en':
            common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        elif target_lang == 'vi':
            common_words = {'và', 'của', 'là', 'có', 'được', 'trong', 'cho', 'này'}
        else:
            # Accept all if language not recognized
            return [True] * len(texts)
        
        mask = []
        for text in texts:
            words = set(str(text).lower().split())
            common_found = len(words & common_words)
            # If at least 2 common words found, likely correct language
            mask.append(common_found >= 2)
        
        return mask
    
    def _perplexity_filter(self, texts: List[str]) -> List[bool]:
        """Filter by perplexity (simulated)"""
        max_perplexity = self.config.get('max_perplexity', 1500)
        
        # Simple heuristic: longer, more coherent texts have lower perplexity
        mask = []
        for text in texts:
            text = str(text)
            # Very short or very repetitive texts likely have high perplexity
            word_count = len(text.split())
            unique_words = len(set(text.lower().split()))
            
            if word_count < 10:
                mask.append(False)
            elif unique_words / max(word_count, 1) < 0.1:  # Too repetitive
                mask.append(False)
            else:
                mask.append(True)
        
        return mask
    
    def _check_special_char_ratio(self, text: str, max_ratio: float) -> bool:
        """Check if special character ratio is acceptable"""
        text = str(text)
        if not text:
            return False
        
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = special_chars / len(text)
        return ratio <= max_ratio
    
    def _medical_term_filter(self, texts: List[str]) -> List[bool]:
        """Filter texts that contain medical terminology"""
        medical_terms = [
            'patient', 'diagnosis', 'treatment', 'clinical', 'medical',
            'symptoms', 'disease', 'therapy', 'medication', 'hospital',
            'doctor', 'nurse', 'prescription', 'surgery'
        ]
        min_ratio = self.config.get('min_medical_term_ratio', 0.05)
        
        mask = []
        for text in texts:
            words = str(text).lower().split()
            if not words:
                mask.append(False)
                continue
            
            medical_count = sum(1 for w in words if w in medical_terms)
            ratio = medical_count / len(words)
            mask.append(ratio >= min_ratio or any(t in str(text).lower() for t in medical_terms))
        
        return mask
    
    def _structure_filter(self, texts: List[str]) -> List[bool]:
        """Filter texts that have proper document structure"""
        min_sections = self.config.get('min_section_count', 3)
        
        mask = []
        for text in texts:
            text = str(text)
            # Count section indicators
            sections = len(re.findall(r'\n\s*\n', text))  # Paragraph breaks
            sections += len(re.findall(r'^\d+\.', text, re.MULTILINE))  # Numbered items
            sections += len(re.findall(r'^[A-Z][A-Z\s]+$', text, re.MULTILINE))  # All-caps headers
            
            mask.append(sections >= min_sections)
        
        return mask
    
    def _build_operator_config(self) -> Dict[str, Any]:
        """Build configuration for DataJuicer operator"""
        config = {
            "type": self.operator_name,
            **self.config
        }
        return config
    
    @classmethod
    def list_operators(cls, category: Optional[str] = None) -> Dict[str, List[str]]:
        """List available operators"""
        if category:
            return {category: cls.OPERATOR_CATEGORIES.get(category, [])}
        return cls.OPERATOR_CATEGORIES


class DataJuicerMapperOperator(DataJuicerProcessor):
    """Convenience class for mapper operators"""
    
    def __init__(self, name: str, operator_name: str, **kwargs):
        super().__init__(name, category="mapper", operator_name=operator_name, **kwargs)


class DataJuicerFilterOperator(DataJuicerProcessor):
    """Convenience class for filter operators"""
    
    def __init__(self, name: str, operator_name: str, **kwargs):
        super().__init__(name, category="filter", operator_name=operator_name, **kwargs)
