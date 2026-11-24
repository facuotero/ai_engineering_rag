"""Carga persistente e idempotente de documentos UTF-8."""
import argparse
import hashlib
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings, make_embeddings


def source_files(settings):
    files = sorted(p for p in settings.data_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.md', '.txt'})
    if not files:
        raise ValueError('No hay documentos .txt o .md en data/.')
    return files


def fingerprint(settings):
    return {
        'embedding_model': settings.embedding_model,
        'collection': settings.collection,
        'chunk_size': settings.chunk_size,
        'chunk_overlap': settings.chunk_overlap,
        'encoding': 'cl100k_base',
        'files': {p.relative_to(settings.data_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files(settings)},
    }


def split_documents(settings):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name='cl100k_base', chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = []
    for path in source_files(settings):
        source = path.relative_to(settings.data_dir).as_posix()
        docs = splitter.split_documents([Document(page_content=path.read_text(encoding='utf-8'), metadata={'source': source})])
        for i, doc in enumerate(docs):
            doc.metadata['chunk_id'] = f'{source}:{i}'
        chunks.extend(docs)
    if not chunks:
        raise ValueError('Los documentos están vacíos.')
    return chunks


def store(settings, embeddings):
    return Chroma(collection_name=settings.collection, persist_directory=str(settings.persist_dir), embedding_function=embeddings)


def open_store(settings, embeddings=None):
    manifest = settings.persist_dir / 'manifest.json'
    if not manifest.exists():
        raise ValueError('Base ausente o incompleta. Ejecutá python ingest.py --rebuild.')
    recorded = json.loads(manifest.read_text())
    if recorded['fingerprint'] != fingerprint(settings):
        raise ValueError('Documentos o configuración cambiaron. Ejecutá python ingest.py --rebuild.')
    db = store(settings, embeddings if embeddings is not None else make_embeddings(settings))
    if len(db.get()['ids']) != recorded['count']:
        raise ValueError('Colección incompleta. Ejecutá python ingest.py --rebuild.')
    return db


def ingest(settings=None, embeddings=None, rebuild=False):
    settings = settings or Settings()
    manifest = settings.persist_dir / 'manifest.json'
    if manifest.exists() and not rebuild:
        open_store(settings, embeddings)
        return 0
    current = fingerprint(settings)
    chunks = split_documents(settings)
    db = store(settings, embeddings if embeddings is not None else make_embeddings(settings))
    if db.get()['ids'] and not rebuild:
        raise ValueError('Base sin manifiesto válido. Ejecutá python ingest.py --rebuild.')
    manifest.unlink(missing_ok=True)
    if rebuild:
        db.reset_collection()
    for start in range(0, len(chunks), 100):
        batch = chunks[start:start + 100]
        db.add_documents(batch, ids=[d.metadata['chunk_id'] for d in batch])
    temporary = manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps({'fingerprint': current, 'count': len(chunks)}, indent=2))
    temporary.replace(manifest)
    return len(chunks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='Reemplaza la colección de este proyecto.')
    args = parser.parse_args()
    try:
        count = ingest(rebuild=args.rebuild)
        print(f'Ingesta completada: {count} fragmentos nuevos.' if count else 'Base vigente: no se recalcularon embeddings.')
    except Exception as exc:
        parser.exit(1, f'Error de ingesta ({type(exc).__name__}). Revisá documentos, configuración y conexión.\n' + (f'{exc}\n' if isinstance(exc, ValueError) else ''))
