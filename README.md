# EMPHNET Policies Chatbot

A Streamlit chatbot for the EMPHNET Human Resources, Events Management, and Internal Communication documents.

The current version uses skeleton-defined semantic nodes as its retrieval units. It does not create arbitrary character chunks and does not use Chroma.

## Architecture

```text
PDFs + skeleton JSON files
        |
        v
PyMuPDF extraction and ordered boundary validation
        |
        v
Enriched trees with complete own_text per semantic node
        |
        +--> BM25 lexical retrieval
        +--> multilingual embedding retrieval
        |
        v
LLM navigation using outline + retrieval candidates
        |
        v
Union of recommended and semantic nodes
        |
        v
LLM grounded answer using fetched own_text
        |
        v
Answer + exact source paths and excerpts
```

A successful question uses two LLM calls:

1. Navigation: selects node IDs and whether the request is broad or narrow.
2. Generation: answers only from the fetched node text and preserves the question language.

BM25 and embedding retrieval run locally and do not add LLM calls.

## Project layout

```text
app.py                         Streamlit entry point
build_skeleton_data.py         Build enriched trees and outline
src/
  skeleton_pipeline.py         PDF extraction, tree building, corpus lookup, QA orchestration
  skeleton_retrieval.py        BM25 plus embedding node retrieval
  skeleton_generation.py       LLM navigation and grounded generation
  skeleton_pipeline_types.py   Shared Navigation type
data/
  pdf/                         Source PDFs
  files_as_nodes/              Ground-truth skeletons
  enriched_nodes/              Generated enriched trees
  outline.json                 Lightweight combined outline for navigation
tests/unit/
  test_skeleton_pipeline.py    Boundary extraction tests
  test_skeleton_retrieval.py   Hybrid retrieval tests
  test_skeleton_qa.py          Retrieval/navigation/generation orchestration tests
```

## Setup

Use Python 3.10 or newer. From PowerShell:

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

Configure `.env` using `.env.example`. The application can use either:

- Ollama through `OLLAMA_HOST` and `OLLAMA_MODEL`.
- Hugging Face Router through `HF_API_KEY` and `HF_MODEL`.

Never commit a real API key.

## Build document data

The skeletons are authoritative. The builder:

- Extracts PDF text with PyMuPDF.
- Removes repeated headers, footers, page numbers, and DocuSign lines.
- Skips the table of contents.
- Resolves headings in document order after the previous boundary.
- Handles wrapped PDF headings and common punctuation artifacts.
- Stops with the node ID if a required boundary cannot be found.
- Writes one enriched tree per document and one combined outline.

Run:

```powershell
venv\Scripts\python.exe build_skeleton_data.py
```

The generated files are:

```text
data/enriched_nodes/GL-ORG-01.json
data/enriched_nodes/ML-ORG-01.json
data/enriched_nodes/ML-HR-01.json
data/outline.json
```

## Run the app

Microsoft SSO is required to access the application. Before running it, fill in
`.streamlit/secrets.toml` with the real values from the Azure App Registration:

```toml
[auth]
redirect_uri = "http://localhost:8501/oauth2callback"
cookie_secret = "..."

[auth.microsoft]
client_id = "..."
client_secret = "..."
server_metadata_url = "https://login.microsoftonline.com/<tenant-id>/v2.0/.well-known/openid-configuration"
```

The `redirect_uri` must exactly match the redirect URI registered in Azure. Keep
`.streamlit/secrets.toml` private; it is ignored by Git. The app uses Streamlit's
native Authlib-based OIDC support through `st.login`, `st.logout`, and `st.user`.

Start Ollama first if no Hugging Face key is configured, then run:

```powershell
venv\Scripts\streamlit.exe run app.py
```

Test narrow questions, broad questions, questions whose wording differs from headings, and Arabic questions. The Sources expander shows the exact paths and excerpts supplied to the answer model.

## Tests

```powershell
venv\Scripts\python.exe -m pytest -q
```

## Rebuilding after source changes

If a PDF or skeleton changes, rerun `build_skeleton_data.py`. The retriever reads the regenerated enriched nodes at application startup. No vector database rebuild is required.

## Removed legacy path

The old regex chunker, arbitrary text splitter, Chroma database, chunk inspection files, legacy retriever, and legacy generation modules were removed. The project now has one maintained extraction path, one maintained retrieval path, and one maintained generation path.
