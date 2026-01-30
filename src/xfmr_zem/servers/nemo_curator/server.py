import os
import sys
import re
from typing import Any, Dict, List, Optional
from xfmr_zem.server import ZemServer
from loguru import logger

# Remove default handler and point to stderr
logger.remove()
logger.add(sys.stderr, level="INFO")

# Initialize the server
server = ZemServer("nemo", parameter_file=os.path.join(os.path.dirname(__file__), "parameter.yaml"))

@server.tool()
def pii_removal(
    data: List[Dict[str, Any]], 
    anonymize_names: bool = True
) -> List[Dict[str, Any]]:
    """
    Simulated PII removal using NeMo Curator logic.
    
    Args:
        data: List of dictionaries with 'text' field.
        anonymize_names: Whether to anonymize names found in text.
    """
    logger.info(f"NeMo: Running pii_removal (anonymize_names={anonymize_names})")
    
    result = []
    # Simplified PII patterns
    patterns = {
        "[NAME]": r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",
        "[PHONE]": r"\b\d{3}-\d{3}-\d{4}\b",
        "[DATE]": r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    }
    
    for item in data:
        text = str(item.get("text", ""))
        processed_text = text
        
        if anonymize_names:
            for placeholder, pattern in patterns.items():
                processed_text = re.sub(pattern, placeholder, processed_text)
                
        new_item = item.copy()
        new_item["text"] = processed_text
        result.append(new_item)
        
    return result

@server.tool()
def text_cleaning(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    General text cleaning using NeMo Curator logic.
    
    Args:
        data: List of dictionaries with 'text' field.
    """
    logger.info("NeMo: Running text_cleaning")
    
    result = []
    for item in data:
        text = str(item.get("text", ""))
        
        # Simulated cleaning: trim, remove double spaces, fix basic punctuation
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([.,!?])\1+', r'\1', text)
        
        new_item = item.copy()
        new_item["text"] = text
        result.append(new_item)
        
    return result

if __name__ == "__main__":
    server.run()
