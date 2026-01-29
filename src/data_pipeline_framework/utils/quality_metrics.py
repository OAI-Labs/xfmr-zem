"""
Quality metrics calculation for domain-specific pipelines
"""
from typing import Any, Dict, List
import re
from loguru import logger

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def calculate_quality_metrics(
    documents: Any,
    metric_names: List[str],
    domain: str
) -> Dict[str, float]:
    """
    Calculate domain-specific quality metrics
    
    Args:
        documents: Processed documents (DataFrame or list of dicts)
        metric_names: List of metrics to calculate
        domain: Domain name
    
    Returns:
        Dictionary of metric names and values
    """
    logger.info(f"Calculating quality metrics for domain: {domain}")
    
    metrics = {}
    
    for metric in metric_names:
        try:
            if metric == "medical_term_coverage":
                metrics[metric] = calculate_medical_term_coverage(documents)
            elif metric == "clinical_coherence":
                metrics[metric] = calculate_clinical_coherence(documents)
            elif metric == "phi_removal_rate":
                metrics[metric] = calculate_phi_removal_rate(documents)
            elif metric == "document_structure_score":
                metrics[metric] = calculate_structure_score(documents)
            elif metric == "citation_accuracy":
                metrics[metric] = calculate_citation_accuracy(documents)
            elif metric == "legal_terminology_density":
                metrics[metric] = calculate_legal_term_density(documents)
            elif metric == "financial_term_coverage":
                metrics[metric] = calculate_financial_term_coverage(documents)
            elif metric == "numerical_consistency":
                metrics[metric] = calculate_numerical_consistency(documents)
            else:
                logger.warning(f"Unknown metric: {metric}")
                metrics[metric] = 0.0
        except Exception as e:
            logger.error(f"Error calculating {metric}: {e}")
            metrics[metric] = 0.0
    
    logger.info(f"Metrics calculated: {metrics}")
    return metrics


def _get_texts(documents: Any) -> List[str]:
    """Extract text content from documents (DataFrame or list)"""
    if HAS_PANDAS and hasattr(documents, 'iterrows'):
        # DataFrame
        if 'text' in documents.columns:
            return documents['text'].tolist()
        return []
    elif isinstance(documents, list):
        # List of dicts or strings
        texts = []
        for doc in documents:
            if isinstance(doc, dict):
                texts.append(doc.get('text', str(doc)))
            else:
                texts.append(str(doc))
        return texts
    return []


def calculate_medical_term_coverage(documents: Any) -> float:
    """Calculate coverage of medical terminology"""
    medical_terms = [
        'patient', 'diagnosis', 'treatment', 'clinical', 'medical', 
        'symptoms', 'disease', 'therapy', 'medication', 'hospital',
        'doctor', 'nurse', 'prescription', 'surgery', 'chronic'
    ]
    
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    docs_with_terms = sum(
        1 for text in texts 
        if any(term in str(text).lower() for term in medical_terms)
    )
    
    return docs_with_terms / len(texts)


def calculate_clinical_coherence(documents: Any) -> float:
    """Calculate clinical coherence score (simulated)"""
    # In real implementation, use domain-specific language model
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    # Simple heuristic: longer, structured documents score higher
    avg_length = sum(len(str(t).split()) for t in texts) / len(texts)
    score = min(0.95, 0.5 + (avg_length / 1000))
    return score


def calculate_phi_removal_rate(documents: Any) -> float:
    """Calculate PHI removal effectiveness"""
    phi_patterns = [
        r'\d{3}-\d{2}-\d{4}',  # SSN
        r'\d{1,2}/\d{1,2}/\d{4}',  # Dates
        r'\b\d{10}\b',  # Phone numbers
    ]
    
    texts = _get_texts(documents)
    if not texts:
        return 1.0
    
    docs_with_phi = 0
    for text in texts:
        text_str = str(text)
        if any(re.search(pattern, text_str) for pattern in phi_patterns):
            docs_with_phi += 1
    
    return 1.0 - (docs_with_phi / len(texts))


def calculate_structure_score(documents: Any) -> float:
    """Calculate document structure quality"""
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    # Check for structure indicators
    structure_score = 0.0
    for text in texts:
        text_str = str(text)
        score = 0.5
        if '\n\n' in text_str:  # Paragraphs
            score += 0.15
        if re.search(r'^\d+\.', text_str, re.MULTILINE):  # Numbered lists
            score += 0.15
        if len(text_str.split()) > 100:  # Sufficient length
            score += 0.2
        structure_score += min(1.0, score)
    
    return structure_score / len(texts)


def calculate_citation_accuracy(documents: Any) -> float:
    """Calculate legal citation accuracy"""
    # Check for proper citation formats
    citation_patterns = [
        r'\d+\s+U\.S\.\s+\d+',  # US Reports
        r'\d+\s+F\.\d+d\s+\d+',  # Federal Reporter
        r'\d+\s+[A-Z][a-z]+\.\s+\d+d?\s+\d+',  # State reporters
    ]
    
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    docs_with_citations = 0
    for text in texts:
        text_str = str(text)
        if any(re.search(pattern, text_str) for pattern in citation_patterns):
            docs_with_citations += 1
    
    return docs_with_citations / len(texts) if len(texts) > 0 else 0.0


def calculate_legal_term_density(documents: Any) -> float:
    """Calculate density of legal terminology"""
    legal_terms = [
        'plaintiff', 'defendant', 'court', 'statute', 'case',
        'jurisdiction', 'precedent', 'brief', 'motion', 'hearing',
        'verdict', 'judgment', 'appeal', 'counsel', 'testimony'
    ]
    
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    total_words = 0
    legal_word_count = 0
    
    for text in texts:
        words = str(text).lower().split()
        total_words += len(words)
        legal_word_count += sum(1 for word in words if word in legal_terms)
    
    return legal_word_count / total_words if total_words > 0 else 0.0


def calculate_financial_term_coverage(documents: Any) -> float:
    """Calculate coverage of financial terminology"""
    financial_terms = [
        'revenue', 'profit', 'earnings', 'assets', 'liabilities',
        'equity', 'balance', 'income', 'cash', 'investment',
        'dividend', 'stock', 'market', 'quarter', 'fiscal'
    ]
    
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    docs_with_terms = sum(
        1 for text in texts 
        if any(term in str(text).lower() for term in financial_terms)
    )
    
    return docs_with_terms / len(texts)


def calculate_numerical_consistency(documents: Any) -> float:
    """Calculate numerical data consistency"""
    texts = _get_texts(documents)
    if not texts:
        return 0.0
    
    # Check for properly formatted numbers
    number_pattern = r'\$?[\d,]+\.?\d*%?'
    
    docs_with_numbers = 0
    for text in texts:
        if re.search(number_pattern, str(text)):
            docs_with_numbers += 1
    
    return docs_with_numbers / len(texts) if docs_with_numbers > 0 else 0.5


def compare_metrics(
    current_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float] = None
) -> Dict[str, Any]:
    """
    Compare current metrics with baseline
    
    Args:
        current_metrics: Current run metrics
        baseline_metrics: Baseline metrics for comparison
    
    Returns:
        Comparison results
    """
    if baseline_metrics is None:
        return {
            'current': current_metrics,
            'baseline': None,
            'improvement': None
        }
    
    improvement = {}
    for metric, value in current_metrics.items():
        if metric in baseline_metrics:
            baseline_value = baseline_metrics[metric]
            improvement[metric] = {
                'current': value,
                'baseline': baseline_value,
                'delta': value - baseline_value,
                'percentage_change': (
                    ((value - baseline_value) / baseline_value * 100) 
                    if baseline_value != 0 else 0
                )
            }
    
    return {
        'current': current_metrics,
        'baseline': baseline_metrics,
        'improvement': improvement
    }
