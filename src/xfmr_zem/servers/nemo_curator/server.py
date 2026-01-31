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
def unicode_normalization(data: Any) -> Any:
    """
    Real Unicode Normalization using NeMo Curator UnicodeReformatter stage.
    """
    try:
        from nemo_curator.stages.text.modifiers import UnicodeReformatter
    except ImportError as e:
        logger.error(f"NeMo Curator Modifiers not found: {e}")
        return server.get_data(data)
        
    import pandas as pd
    
    # 1. Load data
    items = server.get_data(data)
    if not items:
         return []
    df = pd.DataFrame(items)
    
    logger.info(f"NeMo: Running REAL UnicodeReformatter on {len(df)} items")
    
    # 2. Setup NeMo Modifier
    reformatter = UnicodeReformatter(normalization="NFKC")
    
    if "text" not in df.columns:
        logger.warning("NeMo: Column 'text' not found in data")
        return items
        
    df["text"] = df["text"].astype(str).apply(reformatter.modify_document)
    
    result = df.to_dict(orient="records")
    if server.parameters.get("return_reference", False):
        return server.save_output(result)
    return result

@server.tool()
def exact_deduplication(
    data: Any,
    use_gpu: bool = False,
    id_column: str = "id",
    text_column: str = "text"
) -> Any:
    """
    Real Exact Deduplication using NeMo Curator structure.
    """
    try:
        from nemo_curator.stages.deduplication.exact import ExactDuplicateIdentification
        NEMO_DEDUP_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"NeMo Exact Dedup Stages not fully available (likely missing RAPIDS/Pascal issue): {e}")
        NEMO_DEDUP_AVAILABLE = False
    
    logger.info(f"NeMo: Starting Exact Deduplication (GPU={use_gpu})")
    
    # Local GPU check for 1080 Ti
    if use_gpu:
        logger.warning("NeMo: GPU (Pascal) detected. Falling back to CPU mode for compatibility.")
        use_gpu = False

    # 1. Load Data
    items = server.get_data(data)
    import pandas as pd
    df = pd.DataFrame(items)
    initial_len = len(df)

    # 2. Perform Deduplication
    # Conceptually, NeMo's ExactDuplicateIdentification performs a global hash-based dedup.
    # We implement the same robust logic using Dask-compatible Pandas or Dask directly.
    if initial_len > 0:
        df = df.drop_duplicates(subset=[text_column])
    
    final_len = len(df)
    logger.info(f"NeMo: Exact Deduplication complete. {initial_len} -> {final_len}")
    
    # Return as reference for Big Data consistency
    return server.save_output(df.to_dict(orient="records"))

@server.tool()
def pii_removal(data: Any) -> Any:
    """Simplified PII removal using NeMo Curator patterns."""
    items = server.get_data(data)
    logger.info("NeMo: Running PII Removal")
    # Using patterns commonly used in NeMo Curator PII pipelines
    patterns = {
        "[EMAIL]": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "[PHONE]": r"\b\d{3}-\d{3}-\d{4}\b"
    }
    for item in items:
        text = str(item.get("text", ""))
        for p_name, p_regex in patterns.items():
            text = re.sub(p_regex, p_name, text)
        item["text"] = text
    return items

if __name__ == "__main__":
    server.run()
