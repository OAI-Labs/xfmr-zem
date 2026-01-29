# Data Pipeline Framework

Unified data pipeline framework combining **ZenML**, **NemoCurator**, and **DataJuicer** for multi-domain data processing.

## Features

- **ZenML Integration**: Orchestration, visualization, and experiment tracking
- **NemoCurator**: Industrial-grade data curation (deduplication, quality filtering, PII removal)
- **DataJuicer**: 50+ operators for text processing
- **Domain-specific Pipelines**: Pre-built pipelines for legal, medical, general text
- **Flexible Architecture**: Plug-and-play operators with MCP-style communication

## Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .

# With GPU support
uv sync --extra gpu
```

## Quick Start

### Using Pre-built Pipelines

```python
from data_pipeline_framework.pipelines import LegalDataPipeline

# Create pipeline for Vietnamese legal documents
pipeline = LegalDataPipeline(
    name="vn-legal-pipeline",
    enable_dedup=True,
    enable_quality_filter=True,
    min_text_length=100,
    language="vi"
)

# Run pipeline
result = pipeline.run(input_data)
```

### Custom Pipeline

```python
from data_pipeline_framework import Pipeline
from data_pipeline_framework.processors import NemoProcessor, DataJuicerProcessor

# Create custom pipeline
pipeline = Pipeline(name="my-pipeline", domain="legal")

# Add operators
pipeline.add_operator(
    name="unicode_fix",
    operator=NemoProcessor(name="unicode", operation="unicode_fix")
)

pipeline.add_operator(
    name="clean_html",
    operator=DataJuicerProcessor(
        name="html",
        category="mapper",
        operator_name="clean_html"
    )
)

pipeline.add_operator(
    name="dedup",
    operator=NemoProcessor(name="dedup", operation="fuzzy_dedup")
)

# Run
result = pipeline.run(data)
```

### CLI

```bash
# Show info
dpf info

# List operators
dpf operators

# Show templates
dpf templates --domain legal

# Run pipeline
dpf run config.yaml -i input/ -o output/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline Framework                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │    ZenML     │  │ NemoCurator  │  │  DataJuicer  │      │
│  │              │  │              │  │              │      │
│  │ Orchestration│  │ - Dedup      │  │ - 50+ ops    │      │
│  │ Visualization│  │ - Quality    │  │ - Mappers    │      │
│  │ Tracking     │  │ - PII        │  │ - Filters    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                 Pipeline Engine (Core)                 │  │
│  │  - Step orchestration                                  │  │
│  │  - Config management                                   │  │
│  │  - Plugin architecture                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Pre-built Domain Pipelines                │  │
│  │  - LegalDataPipeline                                   │  │
│  │  - TextCleaningPipeline                                │  │
│  │  - (Medical, Code, etc. - extensible)                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
data-pipeline-framework/
├── pyproject.toml
├── README.md
├── src/
│   └── data_pipeline_framework/
│       ├── __init__.py
│       ├── core.py              # Pipeline, Step, Operator base classes
│       ├── cli.py               # CLI commands
│       ├── processors/
│       │   ├── nemo_processor.py      # NemoCurator wrapper
│       │   └── datajuicer_processor.py # DataJuicer wrapper
│       ├── pipelines/
│       │   ├── legal_pipeline.py      # Legal docs pipeline
│       │   └── text_pipeline.py       # General text pipeline
│       ├── operators/           # Custom operators
│       ├── configs/             # Config templates
│       └── utils/               # Utilities
├── tests/
└── examples/
```

## Supported Operations

### NemoCurator
- `language_id` - Language identification
- `unicode_fix` - Unicode normalization
- `exact_dedup` - Exact deduplication
- `fuzzy_dedup` - Fuzzy/near deduplication (MinHash LSH)
- `semantic_dedup` - Semantic deduplication (embeddings)
- `quality_filter` - Quality-based filtering
- `pii_removal` - PII detection and removal
- `text_cleaning` - General text cleaning

### DataJuicer
- **Mappers**: clean_html, clean_links, fix_unicode, punctuation_normalization, whitespace_normalization, remove_header_footer, sentence_split
- **Filters**: language_filter, perplexity_filter, text_length_filter, word_num_filter, special_char_filter, flagged_word_filter
- **Deduplicators**: document_simhash, document_minhash, ray_dedup

## License

MIT
