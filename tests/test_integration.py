"""Pruebas reales: RUN_OPENAI_TESTS=1 python -m pytest -m integration -v"""
import asyncio
import os

import pytest
from config import Settings
from ingest import ingest
from rag import RAGService


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_OPENAI_TESTS') != '1', reason='Opt-in: requiere API y genera consumo')
def test_real_grounding(tmp_path):
    settings = Settings(persist_dir=tmp_path / 'vectorstore')
    ingest(settings)
    service = RAGService(settings)

    async def check():
        answer = await service.get_rag_response('¿Cuántos días dura el préstamo de libros?')
        assert '14' in answer.texto
        assert any(r.archivo == 'prestamos.md' for r in answer.referencias)
        trap = await service.get_rag_response('¿Cuál es la contraseña del Wi-Fi de la biblioteca?')
        assert trap.texto == 'No lo sé'
        assert trap.referencias == []
    asyncio.run(check())
