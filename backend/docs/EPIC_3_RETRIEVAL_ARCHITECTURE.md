# Epic 3: Retrieval & Knowledge Accuracy Improvements

**Project:** ThinkLife — Zoe AI Companion  
**User Stories:** 3.1 — Contextually Accurate Answers; 3.2 — Scalable Retrieval Architecture  
**Author:** Sai Mounika  
**Branch:** `SaiMounika`  
**Scope:** Zoe Backend knowledge ingestion, ChromaDB retrieval, evaluation, and scaling experiments 

**Document type:** Backend implementation and integration reference

## At a glance

| Area | Story | What changed | Main code |
| --- | --- | --- | --- |
| Knowledge ingestion | 3.1 | Source-specific collections, overlapping chunks, metadata, incremental indexing | `backend/agents/zoe/utils/knowledge_index_builder.py` |
| Retrieval quality | 3.1 | Dense search, BM25 sparse search, RRF hybrid ranking, application-level metadata filtering | `backend/brain/data_sources/vector_db.py` |
| Retrieval evaluation | 3.1 | Relevance cases and latency/Hit@k/MRR reporting | `backend/brain/data_sources/evaluation_queries.json`, `vector_db.py` |
| Scaling prototype | 3.2 | Primary and read-replica ChromaDB services, explicit collection synchronization | `backend/docker-compose.replication.yml`, `knowledge_index_builder.py` |
| Query performance | 3.2 | Bounded in-memory LRU result cache and repeat-query diagnostics | `vector_db.py`, `backend/test_lru_cache.py` |
| Delivery and configuration | 3.1 / 3.2 | Package list, sample environment configuration, three knowledge datasets | `backend/requirements.txt`, `backend/.env.example`, `backend/data/` |

## What this work changes

Epic 3 improves how Zoe prepares its knowledge and finds relevant passages when a user asks a question. The retrieval path now combines semantic similarity with keyword matching, separates the knowledge sources into three ChromaDB collections, and can read from two synchronized ChromaDB instances. The same work adds a bounded in-memory LRU cache and scripts for checking retrieval quality and timing.

The main design decision was to keep the indexing process separate from the query path. Indexing writes to the primary instance and explicitly copies the resulting collections to the read replicas. Queries use a replica and rank results from the indexed documents. **This is an application-managed replication prototype, not automatic database replication or production failover.**

## Story 3.1 — Knowledge preparation and retrieval

### Knowledge sources and chunking

`agents/zoe/utils/knowledge_index_builder.py` reads three source files:

| Input | What is indexed | Collection |
| --- | --- | --- |
| `data/context.txt` | Context text split into overlapping chunks | `zoe_context` |
| `data/knowledge_base.json` | Each question paired with chunks of its answer | `zoe_knowledge_shard` |
| `data/empathetic_dialogues_train.csv` | Structured emotion, situation, dialogue, and response examples | `zoe_empathy` |

Context and knowledge answers use **300-character chunks with 50-character overlap**. Empathy examples are assembled as individual documents rather than split using the same chunker. The builder defaults to the first **500** empathy examples and indexes in batches of **25**. Chunk size and overlap are currently constants in the builder; the similarly named values in `.env.example` do not override them.

Each document carries metadata including `source_type`, `source_key`, and `parent_id`; knowledge entries also include `category`, and empathy entries include `emotion`. IDs are generated deterministically from `source_key`, which makes it possible to compare records during incremental indexing.

The builder offers a full rebuild, a user-selected batch rebuild, and incremental indexing. Incremental mode compares stored text and metadata with the newly loaded records, upserts changes, and deletes records that no longer exist. All three modes verify the managed collections before synchronization.

### Embedding model

Both indexing and retrieval use `sentence-transformers/all-MiniLM-L6-v2` through `HuggingFaceEmbeddings`. The retrieval source rejects a configured model name that differs from the model used for indexing. This prevents accidental mixing of embeddings from different models. The code establishes a consistent model choice; it does **not** include a recorded side-by-side comparison proving this model outperforms alternatives.

### Hybrid ranking

`brain/data_sources/vector_db.py` supports three retrieval modes:

- `dense`: embed the query and retrieve nearest neighbors from eligible ChromaDB collections.
- `sparse`: rank the locally loaded document snapshot with BM25 keyword scores.
- `hybrid` (default): combine the dense and sparse result rankings using reciprocal rank fusion (RRF), with a rank offset of 60.

Dense search and BM25 each contribute candidates; hybrid mode merges their ranks and returns up to `k` documents (default `5`). Each returned item contains its content, metadata, source label, score, and document ID. The BM25 corpus is loaded into memory when the data source initializes.

Filtering is handled through the `filter` query argument or `context["filter"]`. Supported metadata conditions include equality, inequality, membership (`$in` / `$nin`), and `$and` / `$or`. A `source_type` condition can narrow the collections searched. Filtering is also checked against returned document metadata; it is not currently passed to ChromaDB as a server-side `where` condition.

### Retrieval flow (simplified)

```text
receive query, retrieval mode, k, and optional metadata filter
check the index version and the in-memory result cache
if a matching cached result exists: return a copy
choose a configured read replica
restrict candidate documents using metadata
run dense retrieval and/or BM25, depending on mode
if hybrid: fuse the two rankings using RRF
return the top k results and cache a copy when enabled
```

## Story 3.2 — Scaling and cache behavior

### ChromaDB layout

The Docker Compose file defines the following independent ChromaDB services:

| Service | Host port | Role in the current setup |
| --- | --- | --- |
| `chroma-primary` | `8020` | Index writes |
| `chroma-secondary` | `8021` | Additional instance for replication experiments |
| `chroma-read-1` | `8022` | Query replica |
| `chroma-read-2` | `8023` | Query replica |
| `chroma-snapshot` | `8024` | Snapshot/restore experiments |

All containers have their own bind-mounted data directories under `backend/replication_data/`. Merely starting these containers does not synchronize their data. The index builder copies the three managed collections from the primary to the two read replicas, then compares IDs, text, metadata, and embeddings to check consistency. The index-version file is published only after those checks pass.

At query time, `VectorDataSource` connects to available read replicas and selects one using round-robin routing. Initialization checks the document snapshots for consistency. If a replica cannot be reached at startup, the source logs the failure and can proceed with another available replica; there is no automatic continuous synchronization or complete failover controller.

The collections are separated by **source type**, not automatically distributed across machines as independently partitioned production shards. Separate `replication.py` and `sharding.py` research scripts are not required for the deployed retrieval path. The Docker configuration and index-builder synchronization are the implementation described here.

### LRU retrieval cache

`VectorDataSource` uses `cachetools.LRUCache` with a default capacity of **1,000 entries**, controlled by `ZOE_RETRIEVAL_CACHE_SIZE`. The cache is enabled by default and can be disabled with `ZOE_RETRIEVAL_CACHE_ENABLED=false`.

A cache key includes the normalized query text, result count, candidate count, retrieval mode, and serialized metadata filter. The implementation returns deep copies of cached results, tracks hit/miss counts, and exposes `cache_stats()` for inspection. Least-recently-used entries are evicted when the capacity is reached.

The cache is **in memory**, not persisted in ChromaDB or Redis. It is cleared when the data source closes or is reinitialized. The index builder writes a version marker after successful replica synchronization; when the query source detects a different marker, it clears the cache and requires a restart or an explicit `refresh_sparse_index()` before serving the new index. In the current `query()` implementation, that condition is logged and returns an empty result rather than automatically refreshing the sparse index.

**Integration caveat:** The current `brain/cortex/workflow_engine.py` creates, initializes, queries, and closes a `VectorDataSource` for each data-source execution. This means the cache is not retained across separate chat requests through that workflow. The standalone cache test reuses one source instance, which is a different lifecycle. A long-lived shared data source would be needed to realize cross-request cache hits in the main application.

## Files changed and files added

The inventory below covers the Epic 3 change set. Existing shared integration modules are listed separately.

### Existing files updated

| File | Change | Why it matters |
| --- | --- | --- |
| `backend/brain/data_sources/vector_db.py` | Added dense/BM25 hybrid retrieval, RRF, metadata filtering, replica routing, LRU cache, and retrieval diagnostics | Main query-side implementation for both stories |
| `backend/agents/zoe/utils/knowledge_index_builder.py` | Updated ingestion, chunking, metadata, incremental indexing, replica synchronization, and index-version publication | Produces the collections consumed by retrieval |
| `backend/requirements.txt` | Added `cachetools` for LRU caching and `rank-bm25` for BM25 sparse retrieval | Ensures the retrieval dependencies install in a clean environment |

### Files introduced in the Epic 3 change set

| File | Purpose | Used by |
| --- | --- | --- |
| `backend/brain/data_sources/evaluation_queries.json` | Retrieval relevance cases and expected source/parent metadata | Retrieval evaluation entry point |
| `backend/data/context.txt` | Context knowledge | Index builder |
| `backend/data/knowledge_base.json` | Knowledge Q&A entries | Index builder |
| `backend/data/empathetic_dialogues_train.csv` | Empathetic dialogue examples | Index builder |
| `backend/docker-compose.replication.yml` | Primary, secondary, two read replicas, snapshot test instance | Local ChromaDB scaling setup |
| `backend/test_lru_cache.py` | Repeated-query cache and timing diagnostics | Developer verification |
| `backend/.env.example` | Example embedding, dataset, cache, and index-version settings | Local setup and deployment configuration |

The distinction between updated and introduced files applies to this change set. The architecture document is maintained separately from the runtime implementation.

Research scripts such as `replication.py`, `sharding.py`, and `vertical_scaling.py` are not runtime dependencies of the documented retrieval path.

## Integration impact: other user stories and shared modules

The following existing modules are integration points. They are not included in the Epic 3 file-change inventory:

| Existing module | Dependency / review point |
| --- | --- |
| `backend/plugins/zoe_agent.py` | Creates Zoe's vector data-source specification; still supplies legacy `collection_name` and `db_path` fields. The new replica-based source uses host/port settings instead. |
| `backend/brain/cortex/workflow_engine.py` | Constructs the vector source, initializes it, sends the query, and closes it. This lifecycle prevents cross-request LRU reuse and adds initialization overhead. |
| `backend/brain/data_sources/data_source_registry.py` | Must continue to report `vector_db` as an available data-source type for initialization to succeed. |
| `backend/brain/data_sources/base_data_source.py` and `backend/brain/specs/` | Provide the data-source configuration, interface, and result contract consumed by the workflow. |
| `backend/agents/zoe/zoe_service.py` and `backend/agents/zoe/zoe_core.py` | Consume the response path through Zoe's agent and Cortex; end-to-end tests should confirm that retrieved context reaches generation as intended. |
| `backend/brain/cortex/workflow_engine.py` context formatter | Must handle the nested `{"source": "vector_db", "results": [...]}` shape and avoid treating missing/empty retrieval as successful grounding. |
| `backend/agents/zoe/data/index_version.txt` | Local version marker must be accessible at the same configured path to the indexing and query processes. |

Changes to these shared modules require regression checks for the retrieval lifecycle, result contract, and Zoe response generation.

## Runtime dependencies and configuration

### Python packages

From `backend/requirements.txt`, the relevant packages include `chromadb`, `langchain-huggingface`, `sentence-transformers`, `pandas`, `python-dotenv`, and `cachetools`.

`rank-bm25` provides `BM25Okapi` for sparse retrieval; `cachetools` provides the LRU result cache.

| Dependency | Purpose | Requirement status |
| --- | --- | --- |
| `chromadb` | Vector storage and HTTP client | Present |
| `langchain-huggingface` | Embedding adapter | Present |
| `sentence-transformers` | Local MiniLM embedding model | Present |
| `cachetools` | `LRUCache` implementation | Present |
| `rank-bm25` | BM25 keyword retrieval (`rank_bm25` import) | Present |
| `pandas` | Empathetic-dialogue CSV ingestion | Present |
| `python-dotenv` | Environment loading | Present |

### Settings

| Setting | Current value or behavior |
| --- | --- |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` (must match the indexed model) |
| `CONTEXT_TXT_PATH` | `data/context.txt` |
| `KNOWLEDGE_JSON_PATH` | `data/knowledge_base.json` |
| `EMPATHY_CSV_PATH` | `data/empathetic_dialogues_train.csv` |
| `ZOE_CHROMA_HOST` | Defaults to `localhost` in code |
| `ZOE_CHROMA_PRIMARY_PORT` | Defaults to `8020` in the index builder |
| `ZOE_CHROMA_READ1_PORT` / `ZOE_CHROMA_READ2_PORT` | Default to `8022` / `8023` |
| `ZOE_CHROMA_READ_PORTS` | Optional comma-separated read ports used by the query source |
| `ZOE_RETRIEVAL_CACHE_ENABLED` | `true` by default |
| `ZOE_RETRIEVAL_CACHE_SIZE` | `1000` by default |
| `ZOE_INDEX_VERSION_FILE` | `agents/zoe/data/index_version.txt` |
| `BATCH_INDEX_SIZE` | `25` by default |
| `MAX_EMPATHY_EXAMPLES` | `500` by default |

The `.env.example` contains dataset, embedding, cache, and version settings, but does not list the replica host/port variables. The code defaults are documented above; deployments with a different topology must set the corresponding environment variables. Secrets belong in runtime configuration, not the example file.

The `.env.example` also contains `CHUNK_SIZE=300` and `CHUNK_OVERLAP=50`; these values are currently hardcoded in the builder. `CHROMA_DB_DIR` and `CHROMA_COLLECTION_NAME` remain in the configuration for legacy compatibility, but the current HTTP replica path uses the named source collections and configured ports.

## Local verification

Run the following commands from the `backend` directory. A running Docker engine is required for the ChromaDB services.

```powershell
python -m pip install -r requirements.txt

docker compose -f docker-compose.replication.yml up -d

# Build the three collections and synchronize both read replicas.
python -m agents.zoe.utils.knowledge_index_builder
# Choose 1 for a full rebuild, or 3 for incremental indexing.

# Compare unfiltered/filtered hybrid retrieval and print Hit@k, MRR@5, latency.
python -m brain.data_sources.vector_db

# Include query throughput and replica-routing measurements.
python -m brain.data_sources.vector_db --benchmark --concurrency 5

# Exercise cache hits/misses using one initialized source instance.
python test_lru_cache.py
```

The index builder's `--benchmark` option prints indexing and synchronization timings:

```powershell
python -m agents.zoe.utils.knowledge_index_builder --benchmark
```

`test_lru_cache.py` repeats the questions “What is ThinkLife?” and “How does Zoe work?”, and prints cache status, response time, document count, hit/miss totals, and hit rate. It is a diagnostic script, **not an automated assertion-based test suite**. The evaluation entry point reads `brain/data_sources/evaluation_queries.json` and reports Hit@1, Hit@3, Hit@5, MRR@5, average latency, P95 latency, and filtered metadata compliance. Its benchmark mode additionally reports QPS, P95/P99, empty results, and replica routing.

The benchmark scripts report metrics at runtime. No baseline or comparative performance measurements are included in this document. Reproducible comparisons require dataset size, hardware, replica configuration, cache state, and concurrency to be recorded alongside results.

## Dependencies and conflicts

- **Depends on:** Zoe's existing agent and Cortex request flow, the data-source interface and registry, ChromaDB services, the embedding model, and the three knowledge datasets. The retrieval module expects the collections produced by the index builder; running the chat backend alone does not populate them.
- **Blocks:** No specific downstream user story is confirmed as blocked by Epic 3. End-to-end integration should nevertheless verify that Zoe receives retrieved context, handles empty results safely, and uses the intended replica/index version.
- **Shared files and integration ownership:** `backend/brain/cortex/workflow_engine.py`, `backend/plugins/zoe_agent.py`, `backend/agents/zoe/zoe_service.py`, `backend/brain/data_sources/data_source_registry.py`, and `backend/brain/specs/` are existing integration points. **They are outside the Epic 3 change set.** Changes to their lifecycle, request/response contracts, or configuration may affect Epic 3 and other user stories.
- **Shared configuration:** `backend/requirements.txt` and `backend/.env.example` can be edited by multiple contributors. Concurrent changes to these files must preserve package additions and environment settings required by other components.

## Change-sensitive contracts

| Contract / assumption | Producer | Consumer | What to recheck if changed |
| --- | --- | --- | --- |
| Collection names (`zoe_context`, `zoe_knowledge_shard`, `zoe_empathy`) | Knowledge index builder | `VectorDataSource` | Collection discovery, query results, and tests |
| Embedding model name | Index builder | Vector query source | Rebuild index and verify model compatibility |
| Document metadata (`source_type`, `source_key`, `parent_id`, etc.) | Index builder | Metadata filters and evaluation cases | Filtering and relevance scoring |
| Result structure (`source`, `results`, scores, IDs) | `VectorDataSource` | Cortex workflow and context formatter | Grounding and empty-result handling |
| Index-version file | Index builder | Vector query source | Cache invalidation and sparse-index refresh |
| ChromaDB host/ports and data directories | Docker Compose / environment | Builder and query source | Replica reachability and synchronization |
| Data-source instance lifecycle | Cortex workflow | In-memory LRU cache | Cross-request cache effectiveness |

## Delivery status

| Epic task | Evidence in reviewed source | Current assessment |
| --- | --- | --- |
| 3.1 — Embedding model selection | MiniLM configured consistently in builder and query source | Implemented model choice; comparative selection study not included |
| 3.1 — Hybrid search | Dense, BM25, and RRF modes in `vector_db.py` | Implemented; `rank-bm25` declared as a dependency |
| 3.1 — Metadata filtering | Local filter evaluation and source collection pruning | Implemented at application level |
| 3.1 — Chunking | 300/50 chunking, source metadata, incremental indexing | Implemented; configuration is hardcoded |
| 3.1 — Accuracy/latency evaluation | Evaluation dataset and CLI reporting | Instrumented; measured results not recorded in this document |
| 3.2 — Scaling patterns | Source-separated collections and replica architecture | Documented prototype design |
| 3.2 — Sharding/replication evaluation | Multi-instance ChromaDB configuration and index-builder replica synchronization | Replication prototype implemented; sharding study not production routing |
| 3.2 — Performance benchmarks | Indexing and retrieval benchmark options | Benchmark commands available; outcomes depend on recorded runs |
| 3.2 — Architecture update | This document and source implementation | Documented |

## Known limitations

| Area | Current limitation | Impact / follow-up |
| --- | --- | --- |
| Cache lifecycle | Cortex constructs and closes the vector source for each retrieval execution | LRU entries do not persist across ordinary separate chat requests |
| Index changes | Index-version mismatch invalidates results and currently returns an empty result until refresh/restart | Refresh should be coordinated with indexing and the sparse snapshot |
| Replica synchronization | Explicit copy/check after indexing, not continuous database replication | Read replicas can be stale between synchronizations |
| Sharding | Source-based collections and research scripts, not automated production sharding | No demonstrated distributed shard routing or rebalancing |
| Filtering | Metadata conditions evaluated in application code, not passed as ChromaDB `where` | May require more candidates and memory at larger scale |
| Configuration | Chunk values are constants; some replica host/port settings are absent from `.env.example` | Environment values do not control all builder settings |
| Benchmarks | Test harnesses are included without a published benchmark baseline | No performance improvement percentage or throughput SLA is claimed |
| Test coverage | Cache script prints diagnostics rather than asserting pass/fail | Needs automated regression tests and integration validation |


## Future Enhancements

The following enhancements can further improve the scalability, reliability, and retrieval performance of the existing Epic 3 implementation.

1. **Automated ChromaDB Replication and Failover**
   - Extend the current multi-instance ChromaDB setup with automated replica synchronization and health monitoring.
   - Introduce failover mechanisms to maintain retrieval availability when a primary or read replica becomes unavailable.

2. **Persistent and Cross-Request Caching**
   - Extend the existing LRU caching implementation to retain frequently accessed retrieval results across independent requests and backend restarts.
   - Evaluate Redis or another shared caching solution for deployments involving multiple backend instances.

3. **Retrieval Performance Optimization**
   - Expand benchmarking to evaluate retrieval latency, accuracy, and throughput under larger datasets and concurrent workloads.
   - Fine-tune hybrid search ranking, embedding configurations, and indexing strategies based on evaluation results.


**Validation scope:** The repository includes local retrieval and cache diagnostics. Production failover, cross-request cache reuse, and performance targets require separate end-to-end validation.
