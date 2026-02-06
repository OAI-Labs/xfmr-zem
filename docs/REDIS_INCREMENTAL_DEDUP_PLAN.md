# Redis Integration Plan for Incremental Deduplication

## Overview

This document outlines the plan to implement Redis-backed LSH for incremental deduplication of legal documents. This enables:
- **Persistent index**: No need to rebuild from scratch each time
- **Incremental updates**: Add new documents without re-indexing entire corpus
- **Fast queries**: Check if a new document is a duplicate in milliseconds

## Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Current Flow                              │
├─────────────────────────────────────────────────────────────┤
│  1. Load all documents from file                            │
│  2. Build MinHash LSH index in memory (~3 hours for 166K)   │
│  3. Find duplicates                                          │
│  4. Index is lost when process ends                          │
└─────────────────────────────────────────────────────────────┘
```

## Proposed Architecture with Redis

```
┌─────────────────────────────────────────────────────────────┐
│                    Redis-Backed Flow                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │  New Docs   │ --> │   MinHash   │ --> │    Redis    │   │
│  │  (batch)    │     │   + LSH     │     │   Storage   │   │
│  └─────────────┘     └─────────────┘     └─────────────┘   │
│                             │                    │          │
│                             v                    │          │
│                      ┌─────────────┐             │          │
│                      │   Entity    │ <-----------┘          │
│                      │   Filter    │                        │
│                      └─────────────┘                        │
│                             │                               │
│                             v                               │
│                      ┌─────────────┐                        │
│                      │   Results   │                        │
│                      └─────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Redis Backend Setup

**1.1 Dependencies**
```bash
uv add redis
```

**1.2 Redis Configuration**
```yaml
# parameter.yaml
redis:
  host: "localhost"
  port: 6379
  db: 0
  basename: "legal_dedup_lsh"
  # Optional
  password: null
  socket_timeout: 5
  connection_pool_size: 10
```

**1.3 MinHashLSH with Redis**
```python
# Already partially implemented in server.py
lsh = MinHashLSH(
    num_perm=128,
    threshold=0.9,
    redis_config={
        'host': 'localhost',
        'port': 6379,
        'basename': b'legal_dedup_lsh'
    }
)
```

### Phase 2: New MCP Tools

**2.1 `init_index` - Initialize or load existing index**
```python
@server.tool()
def init_index(
    redis_host: str = "localhost",
    redis_port: int = 6379,
    threshold: float = 0.9,
    num_perm: int = 128,
    clear_existing: bool = False
) -> Dict:
    """
    Initialize or connect to existing Redis-backed LSH index.
    
    Returns:
        Index status (num_documents, creation_time, etc.)
    """
```

**2.2 `add_documents` - Add new documents to index**
```python
@server.tool()
def add_documents(
    data: Any,
    text_column: str = "markdown_content",
    id_column: str = "document_number",
    check_duplicates: bool = True
) -> Dict:
    """
    Add new documents to the index and optionally check for duplicates.
    
    Returns:
        - num_added: Number of documents added
        - duplicates_found: List of duplicate pairs (if check_duplicates=True)
    """
```

**2.3 `query_document` - Check if a single document is duplicate**
```python
@server.tool()
def query_document(
    text: str,
    threshold: float = 0.9
) -> Dict:
    """
    Check if a document is a duplicate of any indexed document.
    
    Returns:
        - is_duplicate: bool
        - similar_docs: List of similar documents with similarity scores
    """
```

**2.4 `get_index_stats` - Get index statistics**
```python
@server.tool()
def get_index_stats() -> Dict:
    """
    Get statistics about the current index.
    
    Returns:
        - num_documents: Total documents indexed
        - index_size_mb: Approximate size in Redis
        - creation_time: When index was created
        - last_update: When last document was added
    """
```

### Phase 3: Workflow for Incremental Updates

**3.1 Initial Indexing (One-time, ~3 hours)**
```python
# Run once to build initial index
pipeline:
  - name: init_index
    dedup.init_index:
      input:
        redis_host: "localhost"
        clear_existing: true
        
  - name: index_all_documents
    dedup.add_documents:
      input:
        data:
          path: "legal_gold/tvpl_enriched.jsonl.gz"
        check_duplicates: false  # Skip for initial indexing
```

**3.2 Daily/Incremental Updates (Fast, seconds to minutes)**
```python
# Run daily for new documents
pipeline:
  - name: check_new_documents
    dedup.add_documents:
      input:
        data:
          path: "new_documents_today.jsonl"
        check_duplicates: true  # Find duplicates with existing corpus
```

**3.3 Single Document Check (Real-time, milliseconds)**
```python
# API endpoint for real-time duplicate check
result = dedup.query_document(
    text="Luật Doanh nghiệp số 59/2020/QH14...",
    threshold=0.9
)

if result['is_duplicate']:
    print(f"Duplicate found: {result['similar_docs'][0]}")
```

### Phase 4: Entity Metadata Storage

For Two-Stage Entity-Aware deduplication, we also need to store entity metadata:

```python
# Redis keys structure
legal_dedup_lsh:minhash:{doc_id}  -> MinHash signature
legal_dedup_lsh:entity:{doc_id}   -> Entity metadata (JSON)
legal_dedup_lsh:meta              -> Index metadata
```

**Entity storage schema:**
```json
{
    "document_number": "59/2020/QH14",
    "issuing_body": "QUỐC HỘI",
    "target_org": "Trường THPT Nguyễn Du",
    "province": "Hà Nội",
    "indexed_at": "2026-02-03T15:00:00Z"
}
```

## Performance Estimates

| Operation | Time (166K docs) |
|-----------|------------------|
| Initial indexing | ~3 hours |
| Add 1 document | ~70ms |
| Add 1000 documents | ~70 seconds |
| Query 1 document | ~5ms |
| Entity comparison | ~0.1ms |

## Redis Memory Requirements

| Component | Size per doc | Total (166K) |
|-----------|--------------|--------------|
| MinHash signature | ~1KB | ~160MB |
| Entity metadata | ~200B | ~32MB |
| LSH buckets | ~500B | ~80MB |
| **Total** | ~1.7KB | **~280MB** |

## Implementation Timeline

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Redis backend setup | 2 hours |
| 2 | New MCP tools | 4 hours |
| 3 | Workflow integration | 2 hours |
| 4 | Entity metadata storage | 2 hours |
| 5 | Testing & documentation | 2 hours |
| **Total** | | **12 hours** |

## Usage Example

```yaml
# Pipeline for daily incremental deduplication
name: daily_dedup_pipeline

servers:
  dedup: servers/deduplication

pipeline:
  # Connect to existing index
  - name: connect_index
    dedup.init_index:
      input:
        redis_host: "redis.internal"
        redis_port: 6379
        
  # Add new documents and check for duplicates
  - name: process_new_docs
    dedup.add_documents:
      input:
        data:
          path: "/data/new_legal_docs_2026-02-03.jsonl"
        text_column: "markdown_content"
        id_column: "document_number"
        check_duplicates: true
        
  # Get results
  - name: report
    dedup.get_index_stats:
      input: {}
```

## Next Steps

1. **Install Redis** (or use existing instance)
2. **Add `redis` to dependencies** in pyproject.toml
3. **Implement Phase 1** - Redis backend
4. **Test with small sample** (1K docs)
5. **Run initial indexing** for full dataset
6. **Set up daily cron job** for incremental updates

## References

- [datasketch Redis backend](https://ekzhu.com/datasketch/lsh.html#minhash-lsh-at-scale)
- [Redis persistence options](https://redis.io/docs/management/persistence/)
- [MinHash LSH theory](https://www.cs.utah.edu/~jeffp/papers/hash-lsh-SODA08.pdf)
