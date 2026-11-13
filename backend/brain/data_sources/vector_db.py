
"""Zoe vector data source: sharded Chroma read replicas + BM25 + RRF."""
import argparse, asyncio, copy, json, logging, math, os, re, statistics, threading, time
from cachetools import LRUCache
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional
from ..specs import IDataSource, DataSourceType
from .base_data_source import DataSourceConfig
from .data_source_registry import get_data_source_registry
logger = logging.getLogger(__name__)
try:
    import chromadb
    from langchain_huggingface import HuggingFaceEmbeddings
    from rank_bm25 import BM25Okapi
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    chromadb = HuggingFaceEmbeddings = BM25Okapi = None
TOKEN_PATTERN = re.compile("\\w+(?:['’-]\\w+)*", re.UNICODE)
COLLECTION_NAMES = {'context': 'zoe_context', 'knowledge': 'zoe_knowledge_shard', 'empathy': 'zoe_empathy'}
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall((text or '').lower())
def metadata_matches(metadata: dict, where: Optional[dict]) -> bool:
    if not where:
        return True
    if '$and' in where:
        return all((metadata_matches(metadata, item) for item in where['$and']))
    if '$or' in where:
        return any((metadata_matches(metadata, item) for item in where['$or']))
    for key, expected in where.items():
        if key.startswith('$'):
            raise ValueError(f'Unsupported filter operator: {key}')
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if len(expected) != 1:
                raise ValueError('Expected one operator per field')
            operator, value = next(iter(expected.items()))
            if operator == '$eq' and actual != value:
                return False
            if operator == '$ne' and actual == value:
                return False
            if operator == '$in' and actual not in value:
                return False
            if operator == '$nin' and actual in value:
                return False
            if operator not in ('$eq', '$ne', '$in', '$nin'):
                raise ValueError(f'Unsupported filter: {operator}')
        elif actual != expected:
            return False
    return True
def _source_constraint(where):
    """Prune collections when source_type is unambiguous."""
    if not where:
        return list(COLLECTION_NAMES)
    if 'source_type' in where:
        value = where['source_type']
        if isinstance(value, str):
            return [value] if value in COLLECTION_NAMES else []
        if isinstance(value, dict):
            if '$eq' in value:
                v = value['$eq']
                return [v] if v in COLLECTION_NAMES else []
            if '$in' in value:
                return [v for v in COLLECTION_NAMES if v in value['$in']]
    if '$and' in where:
        permitted = set(COLLECTION_NAMES)
        for clause in where['$and']:
            permitted.intersection_update(_source_constraint(clause))
        return [v for v in COLLECTION_NAMES if v in permitted]
    return list(COLLECTION_NAMES)
def _all_records(collection):
    ids, texts, metadatas = ([], [], [])
    offset = 0
    while True:
        page = collection.get(limit=500, offset=offset, include=['documents', 'metadatas'])
        batch = page.get('ids') or []
        if not batch:
            break
        ids.extend(batch); texts.extend(page.get('documents') or []); metadatas.extend(page.get('metadatas') or [])
        offset += len(batch)
    return (ids, texts, metadatas)
class VectorDataSource(IDataSource):
    """Queries one read replica per request across eligible collections."""
    def __init__(self, config: DataSourceConfig):
        self.config = config; self.vectorstore = None
        self._initialized = False
        self._is_external = False; self._clients, self._client_ports = [], []
        self._rr_index = 0; self._route_counts = Counter()
        self._rr_lock = threading.Lock(); self._embeddings = None
        self._ids, self._texts, self._metadatas = [], [], []
        self._bm25 = None; self._index_count = 0
        self._version = None
        self._cache = LRUCache(maxsize=int(os.getenv('ZOE_RETRIEVAL_CACHE_SIZE', '1000')))
        self._cache_lock = threading.RLock(); self._cache_hits = self._cache_misses = 0; self._cache_version = None
        self._version_file = Path(os.getenv('ZOE_INDEX_VERSION_FILE', 'agents/zoe/data/index_version.txt'))
        self._cache_enabled = os.getenv('ZOE_RETRIEVAL_CACHE_ENABLED', 'true').lower() == 'true'
    def _validate_config(self) -> bool:
        return bool(get_data_source_registry().check_data_source_available('vector_db'))
    def _connect(self):
        settings = self.config.config or {}
        host = settings.get('chroma_host') or os.getenv('ZOE_CHROMA_HOST', 'localhost')
        port_setting = os.getenv('ZOE_CHROMA_READ_PORTS')
        if port_setting:
            ports = tuple((int(p.strip()) for p in port_setting.split(',') if p.strip()))
        else:
            ports = (int(settings.get('read1_port') or os.getenv('ZOE_CHROMA_READ1_PORT', '8022')), int(settings.get('read2_port') or os.getenv('ZOE_CHROMA_READ2_PORT', '8023')))
        if not ports or len(set(ports)) != len(ports):
            raise ValueError('Specify one or more unique replica ports')
        self._clients, self._client_ports = ([], []);
        for port in ports:
            try:
                client = chromadb.HttpClient(host=host, port=port)
                client.heartbeat()
                for name in COLLECTION_NAMES.values():
                    client.get_collection(name=name)
                self._clients.append(client); self._client_ports.append(port)
            except Exception:
                logger.exception('Replica %s unavailable at initialization', port)
        if not self._clients:
            raise RuntimeError('No read replicas available; run the index builder first')
    def _snapshot(self, client):
        records = {}
        for source, name in COLLECTION_NAMES.items():
            col = client.get_collection(name=name)
            ids, texts, metas = _all_records(col)
            for doc_id, text, metadata in zip(ids, texts, metas):
                if doc_id in records:
                    raise RuntimeError(f'Duplicate ID across collections: {doc_id}')
                records[doc_id] = (text, metadata)
        return records
    def _reload_sparse(self):
        records = self._snapshot(self._clients[0])
        self._ids = sorted(records); self._texts = [records[i][0] for i in self._ids]
        self._metadatas = [records[i][1] for i in self._ids]
        self._index_count = len(self._ids); tokens = [tokenize(text) or ['__empty__'] for text in self._texts]
        self._bm25 = BM25Okapi(tokens) if tokens else None; self._version = {i: records[i] for i in self._ids}
        return self._index_count
    def _current_version(self):
        return self._version_file.read_text(encoding='utf-8').strip() if self._version_file.exists() else None
    def cache_stats(self):
        with self._cache_lock:
            total = self._cache_hits + self._cache_misses
            return {'hits': self._cache_hits, 'misses': self._cache_misses,
                    'hit_rate': self._cache_hits / total if total else 0.0, 'entries': len(self._cache)}
    async def refresh_sparse_index(self):
        """Call after a completed indexing/synchronization cycle."""
        if not self._clients:
            raise RuntimeError('Replica clients not initialized')
        count = await asyncio.to_thread(self._reload_sparse)
        with self._cache_lock:
            self._cache.clear(); self._cache_version = self._current_version()
        return count
    async def initialize(self, init_config: Dict[str, Any]=None) -> bool:
        if not CHROMA_AVAILABLE or not self._validate_config():
            logger.error('Missing dependency or vector source not registered')
            return False
        try:
            init_config = init_config or {}
            if init_config.get('vectorstore') is not None:
                raise ValueError('External single vectorstore is not supported in replica mode')
            self._connect()
            model = (self.config.config or {}).get('embedding_model') or os.getenv('EMBEDDING_MODEL', EMBEDDING_MODEL)
            if model != EMBEDDING_MODEL:
                raise ValueError(f'Embedding model must match indexed model: {EMBEDDING_MODEL}')
            self._embeddings = HuggingFaceEmbeddings(model_name=model); await self.refresh_sparse_index()
            if self._index_count == 0:
                raise RuntimeError('No documents in replica collections')
            for client in self._clients[1:]:
                if self._snapshot(client) != self._version:
                    raise RuntimeError('Read replicas differ; re-run synchronization')
            self._initialized = True; logger.info('Vector source ready: %d docs across %d replicas', self._index_count, len(self._clients)); return True
        except Exception:
            logger.exception('Vector source initialization failed')
            return False
    def _choose_client(self):
        with self._rr_lock:
            index = self._rr_index % len(self._clients)
            self._rr_index += 1; self._route_counts[self._client_ports[index]] += 1
        return self._clients[index]
    def _hybrid_search(self, question: str, k: int, where: Optional[dict], candidates: int, mode: str='hybrid'):
        if mode not in ('dense', 'sparse', 'hybrid'):
            raise ValueError(f'Unsupported retrieval mode: {mode}')
        if self._bm25 is None:
            raise RuntimeError('BM25 index is empty')
        query_tokens = tokenize(question)
        if not query_tokens:
            return []
        client = self._choose_client()
        shards = _source_constraint(where)
        eligible = [i for i, meta in enumerate(self._metadatas) if meta.get('source_type') in shards and metadata_matches(meta, where)]
        lookup = {doc_id: i for i, doc_id in enumerate(self._ids)}
        dense_ids, sparse_ids = [], []
        if mode in ('dense', 'hybrid') and eligible:
            vector = self._embeddings.embed_query(question)
            dense_hits = []
            for source in shards:
                col = client.get_collection(name=COLLECTION_NAMES[source])
                if col.count() == 0:
                    continue
                result = col.query(query_embeddings=[vector], n_results=min(col.count(), candidates), include=['distances'])
                for doc_id, distance in zip(result['ids'][0], result['distances'][0]):
                    if doc_id in lookup and metadata_matches(self._metadatas[lookup[doc_id]], where):
                        dense_hits.append((float(distance), doc_id))
            dense_hits.sort(key=lambda x: (x[0], x[1]))
            dense_ids = list(dict.fromkeys((doc_id for _, doc_id in dense_hits)))[:candidates]
        scores = None
        if mode in ('sparse', 'hybrid'):
            scores = self._bm25.get_scores(query_tokens)
            ranked = sorted(eligible, key=lambda i: (-float(scores[i]), self._ids[i]))
            sparse_ids = [self._ids[i] for i in ranked if scores[i] > 0][:candidates]
        if mode == 'dense':
            selected = dense_ids[:k]
            ranking_scores = {doc_id: 1 / rank for rank, doc_id in enumerate(selected, 1)}
        elif mode == 'sparse':
            selected = sparse_ids[:k]
            ranking_scores = {doc_id: float(scores[lookup[doc_id]]) for doc_id in selected}
        else:
            ranking_scores = {}
            for ranking in (dense_ids, sparse_ids):
                for rank, doc_id in enumerate(ranking, 1):
                    ranking_scores[doc_id] = ranking_scores.get(doc_id, 0.0) + 1 / (60 + rank)
            selected = sorted(ranking_scores, key=lambda doc_id: (-ranking_scores[doc_id], doc_id))[:k]
        return [{'content': self._texts[lookup[doc_id]], 'metadata': self._metadatas[lookup[doc_id]], 'source': 'vector_db', 'score': ranking_scores[doc_id], 'id': doc_id} for doc_id in selected]
    async def query(self, query: str, context: Dict[str, Any]=None, **kwargs) -> List[Dict[str, Any]]:
        if not self._initialized:
            logger.error('Vectorstore not initialized')
            return []
        try:
            settings = self.config.config or {}
            k = int(kwargs.get('k', settings.get('k', 5)))
            if k <= 0:
                return []
            where = kwargs.get('filter')
            if where is None and context:
                where = context.get('filter')
            candidates = max(k, int(settings.get('hybrid_candidates', 20)))
            mode = kwargs.get('mode', 'hybrid')
            if not self._cache_enabled:
                return await asyncio.to_thread(self._hybrid_search, query, k, where, candidates, mode)
            key = (query.strip(), k, candidates, mode, json.dumps(where, sort_keys=True, default=str))
            with self._cache_lock:
                version = self._current_version()
                if version != self._cache_version:
                    self._cache.clear(); self._cache_version = version
                    # The in-memory BM25 snapshot must be refreshed after reindexing.
                    raise RuntimeError('Knowledge index changed; restart backend or refresh_sparse_index()')
                if key in self._cache:
                    self._cache_hits += 1; return copy.deepcopy(self._cache[key])
                self._cache_misses += 1
            results = await asyncio.to_thread(self._hybrid_search, query, k, where, candidates, mode)
            with self._cache_lock:
                if self._current_version() == version:
                    self._cache[key] = copy.deepcopy(results)
            return results
        except Exception:
            logger.exception('Vector query failed')
            return []
    async def health_check(self) -> Dict[str, Any]:
        if not self._initialized:
            return {'status': 'not_initialized', 'initialized': False}
        try:
            counts = {}
            for port, client in zip(self._client_ports, self._clients):
                counts[port] = sum((client.get_collection(name=name).count() for name in COLLECTION_NAMES.values()))
            healthy = all((v == self._index_count and v > 0 for v in counts.values()))
            return {'status': 'healthy' if healthy else 'stale_or_empty', 'document_count': self._index_count, 'bm25_document_count': self._index_count, 'replica_counts': counts, 'initialized': True, 'is_external': self._is_external}
        except Exception as error:
            return {'status': 'unhealthy', 'error': str(error)}
    async def close(self) -> None:
        self._initialized = False; self._clients = []; self._bm25 = None
        with self._cache_lock:
            self._cache.clear()
        self._ids, self._texts, self._metadatas = ([], [], [])
        logger.info('Vector source closed')
    @property
    def source_type(self) -> DataSourceType:
        return DataSourceType.VECTOR_DB
def create_vector_data_source(config: Optional[DataSourceConfig]=None) -> VectorDataSource:
    if config is None:
        config = DataSourceConfig(source_id='vector_db', source_type=DataSourceType.VECTOR_DB, priority=10)
    return VectorDataSource(config)
if __name__ == '__main__':
    async def main(benchmark=False, concurrency=5):
        path = Path(__file__).resolve().parent / 'evaluation_queries.json'
        with path.open(encoding='utf-8') as file:
            data = json.load(file)
        tests = data if isinstance(data, list) else data['queries']
        if not tests:
            raise ValueError('Evaluation dataset is empty')
        source = create_vector_data_source()
        if not await source.initialize():
            raise RuntimeError('Vector source initialization failed')
        async def evaluate(label, mode, filtered):
            hits = [0, 0, 0]; mrr, latencies, compliant = (0.0, [], 0)
            for number, test in enumerate(tests, 1):
                expected = test['expected_parent_keys']; expected = {expected} if isinstance(expected, str) else set(expected)
                category = test.get('expected_source_type')
                if filtered and (not category):
                    raise ValueError(f'Question {number}: missing expected_source_type')
                where = {'source_type': category} if filtered else None
                start = time.perf_counter(); results = await source.query(test['query'], k=5, filter=where, mode=mode)
                latencies.append((time.perf_counter() - start) * 1000)
                if filtered and results and all((item['metadata'].get('source_type') == category for item in results)):
                    compliant += 1
                rank = next((i for i, item in enumerate(results, 1) if item['metadata'].get('parent_id') in expected), None)
                if rank is not None:
                    hits[0] += rank <= 1
                    hits[1] += rank <= 3
                    hits[2] += rank <= 5
                    mrr += 1 / rank
            n = len(tests); p95 = sorted(latencies)[math.ceil(0.95 * n) - 1]
            print(f"\n{'=' * 48}\n{label}\n{'=' * 48}")
            print(f'Questions           : {n}')
            for kk, count in zip((1, 3, 5), hits):
                print(f'Hit@{kk}               : {count / n:.1%}')
            print(f'MRR@5               : {mrr / n:.3f}')
            print(f'Average latency     : {statistics.mean(latencies):.2f} ms')
            print(f'P95 latency         : {p95:.2f} ms')
            if filtered:
                print(f'Metadata compliance : {compliant}/{n} ({compliant / n:.1%})')
        try:
            await evaluate('1. UNFILTERED HYBRID (DENSE + BM25 + RRF)', 'hybrid', False)
            await evaluate('2. FILTERED HYBRID (DENSE + BM25 + RRF)', 'hybrid', True)
            print('\nHealth:', await source.health_check())
            if benchmark:
                questions = [item['query'] for item in tests]
                for active in (1, concurrency):
                    source._route_counts.clear(); durations, errors = ([], 0); sem = asyncio.Semaphore(active)
                    async def run_one(question):
                        nonlocal errors
                        async with sem:
                            begin = time.perf_counter(); results = await source.query(question, k=5, mode='hybrid')
                            durations.append((time.perf_counter() - begin) * 1000)
                            if not results:
                                errors += 1
                    begin = time.perf_counter(); await asyncio.gather(*(run_one(q) for q in questions))
                    elapsed = time.perf_counter() - begin; ordered = sorted(durations)
                    def pct(p):
                        return ordered[max(0, math.ceil(p * len(ordered)) - 1)]
                    print(f'\nBENCHMARK concurrency={active} | queries={len(questions)} | QPS={len(questions) / elapsed:.2f} | avg={statistics.mean(durations):.2f}ms | P95={pct(0.95):.2f}ms | P99={pct(0.99):.2f}ms | empty/failed={errors} | replica routing={dict(source._route_counts)}')
        finally:
            await source.close()
    parser = argparse.ArgumentParser(); parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--concurrency', type=int, default=5); args = parser.parse_args()
    if args.concurrency < 1:
        parser.error('--concurrency must be positive')
    asyncio.run(main(benchmark=args.benchmark, concurrency=args.concurrency))
