# xfmr-zem

**Unified Data Pipeline Framework** combining **ZenML**, **NeMo Curator**, and **DataJuicer** for scalable, config-driven multi-domain data processing.

## Features

- **Config-Driven Architecture**: Add new domains (Medical, Legal, Finance, etc.) simply by adding a YAML configuration file. No new code required!
- **Unified Orchestration**: Powered by ZenML for robust pipeline management, tracking, and reproducibility.
- **Advanced Processing**: Integrates NeMo Curator and DataJuicer for state-of-the-art text curation, deduplication, and quality filtering.
- **Multi-Domain Ready**: Pre-configured for Medical, Legal, and Finance domains with specialized logic (PII removal, citation extraction, etc.).

## Installation

```bash
pip install xfmr-zem
```

## Quick Start

### 1. Run a Pipeline

```bash
# Run the medical pipeline
xz run medical

# Run the finance pipeline
xz run finance
```

### 2. Python API

```python
from xfmr_zem import create_domain_pipeline

# Create generic pipeline from config
pipeline = create_domain_pipeline("medical")
result = pipeline.run(my_data)
```

### 3. Add a New Domain

Create `src/xfmr_zem/configs/domains/my_new_domain.yaml`:

```yaml
domain:
  name: "my_new_domain"
  description: "My custom domain processing"

steps:
  nemo_curator:
    - name: "deduplication"
      enabled: true
```

Then run it:

```bash
xz run my_new_domain
```

## Development

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run tests
pytest
```
