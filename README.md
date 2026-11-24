# Pre-entrega 3: recuperación semántica local (RAG)

Asistente de la **biblioteca ficticia Aurora**. Lee cuatro documentos Markdown, los fragmenta por tokens, persiste embeddings en ChromaDB y responde consultas mediante una cadena LCEL asíncrona con salida Pydantic.

## Requisitos e instalación

Python 3.12 y una clave de OpenAI con acceso a los modelos configurados. La base vectorial es local; los textos y preguntas se envían a OpenAI para embeddings y generación. Las llamadas tienen costo. No cargar información sensible en este ejemplo.

```bash
git clone https://github.com/facuotero/ai_engineering_rag.git
cd ai_engineering_rag
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

En Windows, activar con `.venv\Scripts\activate`. Completar `OPENAI_API_KEY` en `.env`; nunca subir ese archivo. Los modelos por defecto son `gpt-4.1-mini` y `text-embedding-3-small`. Se pueden cambiar con `LLM_MODEL` y `EMBEDDING_MODEL`. La primera ejecución de tiktoken descarga su vocabulario público; las siguientes usan caché.

## Ingesta y consultas

```bash
python ingest.py
python ingest.py  # reutiliza la colección sin recalcular embeddings
python rag.py "¿Cuántos días dura el préstamo de libros?"
python rag.py "¿Cuál es la contraseña del Wi-Fi de la biblioteca?"
```

La primera respuesta debe indicar **14 días** y citar `prestamos.md`. La segunda debe producir este resultado esperado (no una transcripción de una ejecución):

```json
{"texto": "No lo sé", "referencias": []}
```

Uso desde Python o notebook:

```python
from rag import get_rag_response

respuesta = await get_rag_response("¿Cuántos días dura el préstamo de libros?")
print(respuesta.texto)
print(respuesta.referencias)
```

En un script normal, usar `asyncio.run(get_rag_response(...))`. Para varias preguntas en el mismo proceso, reutilizar una instancia de `RAGService`.

## Cómo funciona

- `config.py`: carga `.env` y centraliza modelos y rutas relativas al proyecto.
- `ingest.py`: lee `.txt` y `.md` UTF-8, utiliza `RecursiveCharacterTextSplitter.from_tiktoken_encoder` con `cl100k_base`, tamaño 500 tokens y overlap configurado de 50. Los cortes respetan separadores; el solapamiento efectivo puede ser menor y el fragmento final puede tener menos de 500 tokens.
- `vectorstore/`: contiene ChromaDB y un manifiesto de archivos, modelo, chunking y cantidad de fragmentos. La persistencia es automática. Una colección parcial o incompatible se rechaza.
- `rag.py`: pregunta → retriever asíncrono (top_k=4) → contexto con identificadores → prompt → ChatOpenAI → PydanticOutputParser → validación de referencias. Se conserva la pregunta original y se recupera una sola vez por consulta.
- `RAGResponse`: contiene `texto` y `referencias`; cada referencia tiene `archivo` y `fragmento`. Las referencias ajenas al contexto y las respuestas afirmativas sin referencias se rechazan.

Si cambian los documentos, el modelo de embeddings o el chunking, reconstruir explícitamente:

```bash
python ingest.py --rebuild
```

Esto reemplaza la colección del proyecto. No ejecutar ingestas y consultas simultáneamente durante la reconstrucción. La ingesta normal es idempotente; no se implementa actualización incremental por documento.

El prompt exige responder solo con contexto y decir “No lo sé” cuando no alcanza. Validar referencias no demuestra por sí mismo que una afirmación esté respaldada: las pruebas detectan regresiones concretas, pero no garantizan ausencia absoluta de alucinaciones.

## Pruebas

```bash
python -m pytest -q
# Con .env configurado; genera consumo real:
RUN_OPENAI_TESTS=1 python -m pytest -m integration -v
```

Las pruebas locales usan embeddings deterministas y un modelo simulado: comprueban persistencia real en Chroma, chunking, idempotencia, incompatibilidad, cadena asíncrona y referencias. No evalúan calidad semántica del modelo de OpenAI. La prueba de integración crea una base temporal y verifica tanto la pregunta documentada (14 días con referencia) como la pregunta trampa (contraseña Wi-Fi, desconocida). Se omite por defecto; sin clave válida no puede verificarse el comportamiento real del LLM.

Los errores técnicos se reportan como errores, no como “No lo sé”. Revisar la clave, los modelos habilitados, el saldo y la conexión ante errores de API. Si falta la base o cambió el dataset, ejecutar la ingesta indicada.

## Entrega segura

Se versionan código, dataset, pruebas y dependencias fijadas. `.gitignore` excluye `.env`, `.venv`, `vectorstore/` y cachés. `.env.example` contiene únicamente nombres de variables y valores de ejemplo sin secretos.
