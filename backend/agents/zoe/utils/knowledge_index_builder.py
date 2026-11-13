#!/usr/bin/env python3
"""Zoe indexing: MiniLM 300/50, three logical collections, two read replicas.
Run from backend: python -m agents.zoe.utils.knowledge_index_builder
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
CONTEXT_PATH = os.getenv('CONTEXT_TXT_PATH', 'data/context.txt')
KNOWLEDGE_JSON_PATH = os.getenv('KNOWLEDGE_JSON_PATH', 'data/knowledge_base.json')
EMPATHY_CSV_PATH = os.getenv('EMPATHY_CSV_PATH', 'data/empathetic_dialogues_train.csv')
CHROMA_DB_DIR = os.getenv('CHROMA_DB_DIR', 'agents/zoe/chroma_db')
CHROMA_COLLECTION_NAME = os.getenv('CHROMA_COLLECTION_NAME', 'zoe_knowledge')  # legacy compatibility
MAX_EMPATHY_EXAMPLES = int(os.getenv('MAX_EMPATHY_EXAMPLES', '500'))
BATCH_SIZE = int(os.getenv('BATCH_INDEX_SIZE', '25'))
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
COLLECTION_NAMES = {'context': 'zoe_context', 'knowledge': 'zoe_knowledge_shard', 'empathy': 'zoe_empathy'}
PRIMARY_PORT = int(os.getenv('ZOE_CHROMA_PRIMARY_PORT', '8020'))
REPLICA_PORTS = (int(os.getenv('ZOE_CHROMA_READ1_PORT', '8022')),
                 int(os.getenv('ZOE_CHROMA_READ2_PORT', '8023')))
CHROMA_HOST = os.getenv('ZOE_CHROMA_HOST', 'localhost')
INDEX_VERSION_FILE = Path(os.getenv('ZOE_INDEX_VERSION_FILE', 'agents/zoe/data/index_version.txt'))
_embeddings = None
def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings
def safe_text(value):
    return '' if pd.isna(value) else str(value).strip()
def generate_document_id(source_key):
    return hashlib.sha256(source_key.encode('utf-8')).hexdigest()
def print_header(title):
    print(f"\n{'=' * 65}\n{title}\n{'=' * 65}")
def split_text(text, chunk_size=CHUNK_SIZE):
    if not text or not text.strip():
        return []
    step = chunk_size - CHUNK_OVERLAP
    if step <= 0:
        raise ValueError('Chunk size must exceed overlap')
    return [text[i:i + chunk_size] for i in range(0, len(text), step)
            if text[i:i + chunk_size].strip()]
def load_context_txt(path=CONTEXT_PATH, chunk_size=CHUNK_SIZE):
    text = Path(path).read_text(encoding='utf-8')
    return [Document(page_content=chunk, metadata={
        'source': Path(path).name, 'source_type': 'context',
        'parent_id': 'context', 'source_key': f'context:{i}'})
        for i, chunk in enumerate(split_text(text, chunk_size))]
def load_knowledge_json(path=KNOWLEDGE_JSON_PATH, chunk_size=CHUNK_SIZE):
    with open(path, encoding='utf-8') as stream:
        records = json.load(stream)
    docs = []
    for i, record in enumerate(records):
        question = safe_text(record.get('question', ''))
        answer = safe_text(record.get('answer', ''))
        category = safe_text(record.get('category', 'general')) or 'general'
        if not question and not answer:
            continue
        for j, chunk in enumerate(split_text(answer, chunk_size)):
            docs.append(Document(page_content=f'{question}\n{chunk}', metadata={
                'source': 'knowledge_base', 'source_type': 'knowledge',
                'category': category, 'parent_id': f'knowledge:{i}',
                'source_key': f'knowledge:{i}:{j}'}))
    return docs
def load_empathy_docs(path=EMPATHY_CSV_PATH, max_examples=MAX_EMPATHY_EXAMPLES):
    frame = pd.read_csv(path)
    docs = []
    for i, (_, row) in enumerate(frame.head(max_examples).iterrows()):
        emotion = safe_text(row.get('emotion', ''))
        content = (f"Emotion: {emotion}\n"
                   f"Situation: {safe_text(row.get('situation', ''))}\n"
                   f"Dialogue: {safe_text(row.get('empathetic_dialogues', ''))}\n"
                   f"Agent Response: {safe_text(row.get('labels', ''))}")
        docs.append(Document(page_content=content, metadata={
            'source': 'empathetic_dialogues', 'source_type': 'empathy',
            'emotion': emotion, 'parent_id': f'empathy:{i}',
            'source_key': f'empathy:{i}'}))
    return docs
def load_all_documents():
    context, knowledge, empathy = load_context_txt(), load_knowledge_json(), load_empathy_docs()
    print(f'Context: {len(context)} | Knowledge: {len(knowledge)} | Empathy: {len(empathy)}')
    return context + knowledge + empathy
def prepare_documents(documents):
    ids = [generate_document_id(d.metadata['source_key']) for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate source_key values detected')
    return ids, [d.page_content for d in documents], [d.metadata for d in documents]
def client_at(port):
    client = chromadb.HttpClient(host=CHROMA_HOST, port=port)
    client.heartbeat()
    return client
def get_collection(source_type='knowledge'):
    """Legacy two-value return, now points to the primary source-specific shard."""
    client = client_at(PRIMARY_PORT)
    return client, client.get_or_create_collection(name=COLLECTION_NAMES[source_type])
def embed_documents(texts):
    return get_embeddings().embed_documents(texts)
def remove_existing_database():
    """Delete only the three managed primary shards, never unrelated collections."""
    client = client_at(PRIMARY_PORT)
    names = {x if isinstance(x, str) else x.name for x in client.list_collections()}
    for name in COLLECTION_NAMES.values():
        if name in names:
            client.delete_collection(name=name)
def _write_batches(collection, documents, batch_size, *, upsert=False):
    if batch_size <= 0:
        raise ValueError('Batch size must be positive')
    total = 0
    for offset in range(0, len(documents), batch_size):
        group = documents[offset:offset + batch_size]
        ids, texts, metadata = prepare_documents(group)
        vectors = embed_documents(texts)
        operation = collection.upsert if upsert else collection.add
        operation(ids=ids, documents=texts, metadatas=metadata, embeddings=vectors)
        total += len(group)
        print(f'{collection.name}: indexed {total}/{len(documents)}', end='\r')
    if documents:
        print()
    return total
def _group(documents):
    grouped = {source: [] for source in COLLECTION_NAMES}
    for doc in documents:
        source = doc.metadata['source_type']
        if source not in grouped:
            raise ValueError(f'Unknown source_type: {source}')
        grouped[source].append(doc)
    return grouped
def _fetch_all(collection, *, include):
    result = {'ids': [], **{key: [] for key in include}}
    offset = 0
    while True:
        page = collection.get(limit=200, offset=offset, include=list(include))
        ids = page['ids']
        if not ids:
            break
        result['ids'].extend(ids)
        for key in include:
            result[key].extend(list(page[key]))
        offset += len(ids)
    return result
def _signature(data):
    """Compare IDs, documents, metadata and embedding values, not just counts."""
    entries = []
    for i, doc_id in enumerate(data['ids']):
        entries.append((doc_id, data['documents'][i], data['metadatas'][i],
                        [float(v) for v in data['embeddings'][i]]))
    payload = json.dumps(sorted(entries, key=lambda x: x[0]), sort_keys=True,
                         ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
def publish_index_version():
    """Atomically signal retrieval processes to invalidate their LRU caches.

    Call only after every read replica has passed synchronization checks.
    The cache itself is owned by VectorDataSource, not this indexing process.
    """
    INDEX_VERSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = INDEX_VERSION_FILE.with_name(INDEX_VERSION_FILE.name + '.tmp')
    new_version = uuid.uuid4().hex
    temporary.write_text(new_version, encoding='utf-8')
    os.replace(temporary, INDEX_VERSION_FILE)
    print(f'Published index version {new_version}; retrieval LRU caches will invalidate.')
    return new_version


def sync_replicas(benchmark=False):
    """Full-copy synchronization; replicas are NOT continuously replicated."""
    print_header('SYNCHRONIZING READ REPLICAS')
    primary = client_at(PRIMARY_PORT)
    sources = {}
    for source, name in COLLECTION_NAMES.items():
        col = primary.get_collection(name=name)
        data = _fetch_all(col, include=('documents', 'metadatas', 'embeddings'))
        sources[source] = (data, _signature(data))
    for port in REPLICA_PORTS:
        replica_start = time.perf_counter()
        replica = client_at(port)
        for source, name in COLLECTION_NAMES.items():
            data, expected_hash = sources[source]
            # Recreate only our managed collection on the replica.
            names = {x if isinstance(x, str) else x.name for x in replica.list_collections()}
            if name in names:
                replica.delete_collection(name=name)
            target = replica.get_or_create_collection(name=name)
            for offset in range(0, len(data['ids']), BATCH_SIZE):
                end = offset + BATCH_SIZE
                target.add(ids=data['ids'][offset:end],
                           documents=data['documents'][offset:end],
                           metadatas=data['metadatas'][offset:end],
                           embeddings=data['embeddings'][offset:end])
            actual = _fetch_all(target, include=('documents', 'metadatas', 'embeddings'))
            if len(actual['ids']) != len(data['ids']) or _signature(actual) != expected_hash:
                raise RuntimeError(f'Replica validation FAILED on port {port}: {name}')
            print(f'Replica {port} | {name}: {len(actual["ids"])} | content check PASS')
        if benchmark:
            print(f'Replica {port} sync + validation: {time.perf_counter() - replica_start:.2f}s')
    print('Both read replicas synchronized and validated.')
    # Only publish after both replicas have passed content validation.
    publish_index_version()
def verify_index(collection=None):
    if collection is not None:
        collections = [collection]
    else:
        client = client_at(PRIMARY_PORT)
        collections = [client.get_collection(name=n) for n in COLLECTION_NAMES.values()]
    total = 0
    for col in collections:
        count = col.count()
        if count == 0:
            raise RuntimeError(f'Empty index: {col.name}')
        page = col.get(limit=min(10, count), include=['metadatas'])
        for metadata in page['metadatas']:
            missing = {'source_key', 'source_type', 'parent_id'} - set(metadata or {})
            if missing:
                raise RuntimeError(f'Missing metadata: {sorted(missing)}')
        print(f'Index verification passed: {col.name}: {count}')
        total += count
    return total
def validate_configuration():
    if BATCH_SIZE <= 0 or MAX_EMPATHY_EXAMPLES < 0:
        raise ValueError('Invalid batch size or empathy limit')
    if len(set(COLLECTION_NAMES.values())) != 3:
        raise ValueError('Collection names must be unique')
    if len({PRIMARY_PORT, *REPLICA_PORTS}) != 3:
        raise ValueError('Primary and replica ports must differ')
    return True
def _rebuild(batch_size, benchmark=False):
    validate_configuration()
    pipeline_start = time.perf_counter()
    docs = load_all_documents()
    load_seconds = time.perf_counter() - pipeline_start
    if not docs or any(not group for group in _group(docs).values()):
        raise ValueError('Each source must have documents; refusing rebuild')
    prepare_documents(docs)
    get_embeddings()
    primary = client_at(PRIMARY_PORT)
    start = time.perf_counter()
    for source, group in _group(docs).items():
        shard_start = time.perf_counter()
        name = COLLECTION_NAMES[source]
        existing = {x if isinstance(x, str) else x.name for x in primary.list_collections()}
        if name in existing:
            primary.delete_collection(name=name)
        col = primary.get_or_create_collection(name=name)
        _write_batches(col, group, batch_size)
        if benchmark:
            elapsed = time.perf_counter() - shard_start
            print(f"{name}: {elapsed:.2f}s | {len(group)/max(elapsed, 1e-9):.1f} docs/s (embedding + writes)")
    index_seconds = time.perf_counter() - start
    verify_index()
    sync_start = time.perf_counter()
    sync_replicas(benchmark=benchmark)
    sync_seconds = time.perf_counter() - sync_start
    if benchmark:
        print(f"BENCHMARK | load={load_seconds:.2f}s | primary={index_seconds:.2f}s "
              f"| sync+validation={sync_seconds:.2f}s | total={time.perf_counter()-pipeline_start:.2f}s")
    print(f'Indexed {len(docs)} chunks in {time.perf_counter() - start:.2f}s (includes synchronization)')
    return len(docs)
def full_rebuild(benchmark=False):
    print_header('FULL REBUILD: THREE COLLECTIONS')
    return _rebuild(BATCH_SIZE, benchmark=benchmark)
def get_batch_size():
    raw = input(f'Enter batch size [{BATCH_SIZE}]: ').strip()
    try:
        return int(raw) if raw and int(raw) > 0 else BATCH_SIZE
    except ValueError:
        return BATCH_SIZE
def batch_indexing(benchmark=False):
    print_header('BATCH INDEXING: THREE COLLECTIONS')
    return _rebuild(get_batch_size(), benchmark=benchmark)
def _changed_documents(documents, existing):
    old = {doc_id: (text, metadata) for doc_id, text, metadata in zip(
        existing['ids'], existing['documents'], existing['metadatas'])}
    current = {generate_document_id(doc.metadata['source_key']): doc for doc in documents}
    deleted = sorted(set(old) - set(current))
    changed = [doc for doc_id, doc in current.items()
               if doc_id not in old or old[doc_id] != (doc.page_content, doc.metadata)]
    return deleted, changed
def incremental_indexing(benchmark=False):
    print_header('INCREMENTAL INDEXING: THREE COLLECTIONS')
    start = time.perf_counter()
    validate_configuration()
    docs = load_all_documents()
    if not docs or any(not group for group in _group(docs).values()):
        raise ValueError('Each source must have documents; refusing incremental indexing')
    prepare_documents(docs)
    primary = client_at(PRIMARY_PORT)
    changed_total = 0
    for source, group in _group(docs).items():
        col = primary.get_or_create_collection(name=COLLECTION_NAMES[source])
        existing = _fetch_all(col, include=('documents', 'metadatas'))
        deleted, changed = _changed_documents(group, existing)
        print(f'{source}: changed={len(changed)}, deleted={len(deleted)}')
        if changed:
            _write_batches(col, changed, BATCH_SIZE, upsert=True)
        for offset in range(0, len(deleted), BATCH_SIZE):
            col.delete(ids=deleted[offset:offset + BATCH_SIZE])
        changed_total += len(changed)
    verify_index()
    sync_start = time.perf_counter()
    sync_replicas(benchmark=benchmark)
    if benchmark:
        print(f"BENCHMARK | changed={changed_total} | sync+validation={time.perf_counter()-sync_start:.2f}s "
              f"| total={time.perf_counter()-start:.2f}s")
    return changed_total
def print_configuration():
    print(f'Embedding: {EMBEDDING_MODEL}')
    print(f'Chunking: fixed {CHUNK_SIZE} characters / {CHUNK_OVERLAP} overlap')
    print(f'Primary: {CHROMA_HOST}:{PRIMARY_PORT}')
    print(f'Replicas: {REPLICA_PORTS}')
    print(f'Collections: {COLLECTION_NAMES}')
def build_chroma_index():
    return full_rebuild()
def choose_indexing_mode(benchmark=False):
    print_header('ZOE KNOWLEDGE INDEX BUILDER')
    print_configuration()
    print('1. Full rebuild\n2. Batch indexing\n3. Incremental indexing')
    choice = input('Enter choice (1/2/3): ').strip()
    choices = {'1': full_rebuild, '2': batch_indexing, '3': incremental_indexing}
    if choice not in choices:
        print('Invalid choice')
        return None
    return choices[choice](benchmark=benchmark)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true', help='Print indexing and synchronization timings')
    choose_indexing_mode(benchmark=parser.parse_args().benchmark)
