"""
NER-based Entity Extraction for Two-Stage Deduplication

Uses NlpHUST/ner-vietnamese-electra-base model to extract organizations
from document titles for entity-aware deduplication.
"""

import re
from typing import Dict, Set, Tuple
from loguru import logger

# =============================================================================
# NER MODEL (LAZY LOADING)
# =============================================================================

_ner_pipeline = None


def get_ner_pipeline():
    """
    Lazy load Vietnamese NER pipeline.
    Uses NlpHUST/ner-vietnamese-electra-base model.
    Auto-detects GPU if available.
    """
    global _ner_pipeline
    if _ner_pipeline is None:
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            import torch
            
            # Auto-detect GPU
            if torch.cuda.is_available():
                device = 0  # Use first GPU
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            else:
                device = -1  # CPU
                logger.info("No GPU detected, using CPU")
            
            logger.info("Loading Vietnamese NER model (NlpHUST/ner-vietnamese-electra-base)...")
            tokenizer = AutoTokenizer.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
            model = AutoModelForTokenClassification.from_pretrained("NlpHUST/ner-vietnamese-electra-base")
            _ner_pipeline = pipeline(
                "ner", 
                model=model, 
                tokenizer=tokenizer, 
                aggregation_strategy="simple",
                device=device
            )
            logger.info(f"Vietnamese NER model loaded successfully (device={'GPU' if device >= 0 else 'CPU'})")
        except Exception as e:
            logger.warning(f"Failed to load NER model: {e}. Falling back to regex-based extraction.")
            _ner_pipeline = None
    return _ner_pipeline


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_entities_ner(title: str) -> Set[str]:
    """
    Extract organization entities from title using NER.
    
    Args:
        title: Document title
    
    Returns:
        Set of organization names
    """
    ner = get_ner_pipeline()
    if ner is None:
        return set()
    
    try:
        results = ner(title)
        orgs = {r['word'] for r in results if r['entity_group'] == 'ORGANIZATION'}
        return orgs
    except Exception as e:
        logger.warning(f"NER extraction failed: {e}")
        return set()


def extract_entities(doc: Dict) -> Dict[str, any]:
    """
    Extract key entities from a legal document for entity-aware deduplication.
    
    Uses NER model (NlpHUST/ner-vietnamese-electra-base) to extract organizations
    from document titles. This is more general and accurate than regex-based extraction.
    
    Args:
        doc: Document dictionary with 'title', 'markdown_content', 'issuing_body', etc.
    
    Returns:
        Dictionary of extracted entities including 'organizations' set
    """
    title = doc.get('title', '')
    
    entities = {
        'document_number': doc.get('document_number', ''),
        'issuing_body': doc.get('issuing_body', ''),
        'document_type': doc.get('document_type', ''),
        'issue_date': doc.get('issue_date', ''),
        'title': title,
    }
    
    # Extract organizations using NER
    entities['organizations'] = extract_entities_ner(title)
    
    # Fallback: Extract province from title or issuing_body
    province_match = re.search(
        r'(?:tỉnh|thành\s+phố)\s+([A-Za-zđĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+(?:\s+[A-Za-zđĐáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ]+)?)',
        title + ' ' + entities['issuing_body'],
        re.IGNORECASE
    )
    if province_match:
        entities['province'] = province_match.group(1).strip()
    
    return entities


def entities_match(e1: Dict, e2: Dict) -> Tuple[bool, str]:
    """
    Check if two documents have matching key entities using NER-extracted organizations.
    
    Returns:
        (is_match, reason): True if likely true duplicates, False if template-based
    """
    # PRIMARY CHECK: Compare NER-extracted organizations
    orgs1 = e1.get('organizations', set())
    orgs2 = e2.get('organizations', set())
    
    if orgs1 and orgs2:
        # If both have organizations extracted, they must match exactly
        if orgs1 != orgs2:
            return False, 'different_organizations'
    
    # SECONDARY CHECK: Check issuing body - different provinces issue similar docs
    ib1 = e1.get('issuing_body', '')
    ib2 = e2.get('issuing_body', '')
    
    if ib1 and ib2 and ib1 != ib2:
        return False, 'different_issuing_body'
    
    # FALLBACK: If no organizations extracted, use exact title match
    if not orgs1 and not orgs2:
        title1 = e1.get('title', '').strip().lower()
        title2 = e2.get('title', '').strip().lower()
        if title1 and title2 and title1 != title2:
            return False, 'different_titles'
    
    return True, 'match'
