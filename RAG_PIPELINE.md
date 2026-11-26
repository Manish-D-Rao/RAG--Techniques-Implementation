# Advanced RAG Pipeline

End-to-end retrieval-augmented generation pipeline for the AsterPeak document
set. Each stage maps to a module/constant in the codebase.

## Pipeline Overview

```
Load → Clean → Chunk (parent/child) → Embed → Index (vector + BM25)
   → Retrieve (hybrid) → Rerank → Build context → Generate → Cite
```

## 1. Data Ingestion

**Module:** `src/data_ingestion.py`

- `load_documents_from_folder()` scans `data/` (non-recursive) for supported files.
- `load_document()` picks a loader by extension:
  - `.pdf` → `PyPDFLoader`
  - `.txt` → `TextLoader` (utf-8)
  - `.docx` → `Docx2txtLoader`
- Unsupported/corrupt files are skipped and logged.

**Config:** `SUPPORTED_EXTENSIONS = {.pdf, .txt, .docx}`

## 2. Text Cleaning

**Function:** `clean_text()`

- Converts form-feed page breaks (`\x0c`) to paragraph breaks.
- Drops non-printable control characters.
- Collapses spaces/tabs; caps blank-line runs.
- Preserves line breaks so headings/paragraphs survive for chunking.

## 3. Metadata & Citations

**Functions:** `add_metadata()`, `guess_section()`

- `add_metadata()` attaches `source`, `filename`, `doc_type`, and 1-based
  `page_number` to every document.
- `guess_section()` detects a heading-like line (ALL CAPS / Title Case /
  numbered) in the first 15 lines and returns it (number stripped) for
  citations.

## 4. Chunking (Parent–Child)

**Config:** `PARENT_CHUNK_SIZE`, `PARENT_CHUNK_OVERLAP`,
`CHILD_CHUNK_SIZE`, `CHILD_CHUNK_OVERLAP`

- Documents are split into large **parent** chunks for context and small
  **child** chunks for precise retrieval.
- Child chunks carry a reference to their parent so retrieved snippets can be
  expanded back to full context.

## 5. Embedding

**Module:** `llm_client.py` → `get_embedding_model()`

- `OpenAIEmbeddings` with `OPENAI_EMBEDDING_MODEL` (default
  `text-embedding-3-small`).
- Cached via `lru_cache(maxsize=1)`.

**Config:** `OPENAI_EMBEDDING_MODEL`, `NEXUS_BASE_URL`

## 6. Indexing

**Config:** `VECTOR_DB` (`faiss` | `pinecone`)

- **FAISS** — local vector files, no account needed.
- **Pinecone** — remote index (`PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`).
- A parallel **BM25** keyword index supports hybrid retrieval.

## 7. Retrieval (Hybrid)

**Config:** `VECTOR_TOP_K`, `BM25_TOP_K`, `HYBRID_TOP_K`,
`NUM_QUERY_VARIANTS`

- Generates `NUM_QUERY_VARIANTS` query rewrites.
- Runs dense (vector) + sparse (BM25) retrieval.
- Fuses results into a single candidate set of size `HYBRID_TOP_K`.

## 8. Reranking

**Config:** `RERANKER` (`cross-encoder` | `simple`), `RERANK_TOP_K`,
`MIN_RERANK_SCORE`, `CROSS_ENCODER_MODEL`

- `cross-encoder` — accurate, needs `sentence-transformers` + PyTorch.
- `simple` — keyword-overlap fallback, no extra deps.
- Keeps top `RERANK_TOP_K` results scoring above `MIN_RERANK_SCORE`.

## 9. Context Assembly

**Config:** `MAX_CONTEXT_CHARS`, `MEMORY_WINDOW`

- Expands child hits to parent chunks.
- Trims to `MAX_CONTEXT_CHARS`.
- Includes the last `MEMORY_WINDOW` turns of conversation history.

## 10. Generation

**Module:** `llm_client.py` → `get_chat_model()`

- `ChatOpenAI` with `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`) and
  `LLM_TEMPERATURE`.
- Cached via `lru_cache(maxsize=1)`.

**Config:** `OPENAI_CHAT_MODEL`, `LLM_TEMPERATURE`, `NEXUS_BASE_URL`

## 11. Citation

- Each answer cites `filename` + `page_number` + `section` (from
  `guess_section`) so claims trace back to source documents.

## Configuration

All values are loaded from `.env` in `config.py`. Required: `OPENAI_API_KEY`.
Pinecone keys are only required when `VECTOR_DB=pinecone`.