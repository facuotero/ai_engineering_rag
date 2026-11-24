import asyncio
import hashlib
import json
import re

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from config import Settings
from ingest import ingest, open_store, split_documents
from rag import RAGService


class LocalEmbeddings(Embeddings):
    calls = 0

    def embed_documents(self, texts):
        self.calls += len(texts)
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [float(x) / 255 for x in hashlib.sha256(text.encode()).digest()]


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'prestamos.md').write_text('El préstamo dura 14 días. ' * 300)
    return Settings(data_dir=data, persist_dir=tmp_path / 'db'), LocalEmbeddings()


def test_chunking_and_persistence(setup):
    settings, embeddings = setup
    import tiktoken
    chunks = split_documents(settings)
    assert len(chunks) > 1
    assert all(len(tiktoken.get_encoding('cl100k_base').encode(c.page_content)) <= 500 for c in chunks)
    assert all(c.metadata['source'] == 'prestamos.md' for c in chunks)
    assert ingest(settings, embeddings) > 0
    calls = embeddings.calls
    assert ingest(settings, embeddings) == 0
    assert embeddings.calls == calls
    assert open_store(settings, embeddings).similarity_search('préstamo', k=4)


def test_changed_dataset_and_model_rejected(setup):
    settings, embeddings = setup
    ingest(settings, embeddings)
    (settings.data_dir / 'otro.txt').write_text('Nuevo documento')
    with pytest.raises(ValueError, match='rebuild'):
        ingest(settings, embeddings)
    settings.embedding_model = 'otro-modelo'
    with pytest.raises(ValueError, match='rebuild'):
        open_store(settings, embeddings)


@pytest.mark.parametrize('answer', [
    {'texto': 'No lo sé', 'referencias': []},
    {'texto': '14 días', 'referencias': [{'archivo': 'prestamos.md', 'fragmento': 'prestamos.md:0'}]},
])
def test_async_chain(setup, answer):
    settings, embeddings = setup
    ingest(settings, embeddings)
    def respond(prompt):
        result = dict(answer)
        if result['referencias']:
            chunk_id = re.search(r'Fragmento: ([^\n]+)', prompt.to_string()).group(1)
            result['referencias'] = [{'archivo': 'prestamos.md', 'fragmento': chunk_id}]
        return json.dumps(result)
    llm = RunnableLambda(respond)
    service = RAGService(settings, embeddings, llm)
    result = asyncio.run(service.get_rag_response('¿Cuánto dura?'))
    assert result.texto == answer['texto']


def test_invalid_reference_and_empty_query(setup):
    settings, embeddings = setup
    ingest(settings, embeddings)
    llm = RunnableLambda(lambda prompt: json.dumps({'texto': '14 días', 'referencias': [{'archivo': 'inventado', 'fragmento': 'x'}]}))
    service = RAGService(settings, embeddings, llm)
    with pytest.raises(ValueError, match='referencia'):
        asyncio.run(service.get_rag_response('duración'))
    with pytest.raises(ValueError, match='vacía'):
        asyncio.run(service.get_rag_response(' '))
