"""Configuración compartida: indexación y consulta usan el mismo embedding."""
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')


@dataclass
class Settings:
    data_dir: Path = ROOT / 'data'
    persist_dir: Path = ROOT / 'vectorstore'
    embedding_model: str = field(default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'))
    llm_model: str = field(default_factory=lambda: os.getenv('LLM_MODEL', 'gpt-4.1-mini'))
    collection: str = 'biblioteca'
    chunk_size: int = 500
    chunk_overlap: int = 50


def make_embeddings(settings):
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError('Configurá OPENAI_API_KEY en .env antes de continuar.')
    return OpenAIEmbeddings(model=settings.embedding_model)
