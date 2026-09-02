# EMPHNET Policies Chatbot

A retrieval-augmented generation (RAG) application for answering HR-policy questions from the EMPHNET policy manual while keeping answers grounded in the source text and citing the relevant sections.

## What this project does

- Parses the HR policy PDF into structured chunks with metadata
- Builds a hybrid retrieval index combining semantic vector search and BM25 keyword matching
- Maps user questions to policy structure such as sections, subsections, policies, and procedures
- Uses Ollama to generate responses grounded only in retrieved policy excerpts
- Exposes the system through a simple Streamlit UI

## System overview

```text
User query (EN / AR)
    ↓
Streamlit UI (app.py)
    ↓
Query intent + document structure matching
    ↓
Hybrid retrieval (vector + BM25)
    ↓
Grounded answer generation via Ollama
    ↓
Answer + source citations
```

## Project layout

```text
.
├── app.py                     # Streamlit application entry point
├── embed_chunks.py            # Rebuild the Chroma index from the inspection JSON
├── src/
│   ├── __init__.py            # Package metadata
│   ├── config.py              # Environment variable parsing and settings
│   ├── document_structure.py  # Section / subsection / policy detection logic
│   ├── generation.py          # Ollama generation and compatibility wrapper
│   ├── index_manager.py       # Explicit index lifecycle and verification
│   ├── ingestion.py           # PDF parsing, chunking, metadata extraction
│   ├── query_intent.py        # Query analysis exports
│   └── retrieval.py           # Hybrid vector + BM25 retriever
├── data/
│   ├── pdf/                  # Policy PDF source files
│   └── chunks_inspection.json # Generated inspection / chunk dataset
├── chroma_storage/            # Persistent ChromaDB index
├── tests/
│   ├── integration/           # Integration checks for retrieval and index state
│   └── unit/                 # Focused unit tests for config and generation
├── .env.example               # Environment template
├── .gitignore                 # Git exclusions
├── README.md                  # Project overview
├── requirements.txt           # Python dependencies
└── venv/                      # Local virtual environment
```

## Requirements

- Python 3.10+
- Ollama installed and running locally
- The HR policy PDF available under data/pdf/
- Access to the local model used by the project configuration

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure your environment by copying the template:

```bash
copy .env.example .env
```

4. Edit `.env` to set values such as:

```env
PDF_PATH=data/pdf/your_policy_file.pdf
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
CHROMA_DB_PATH=chroma_storage
CHROMA_COLLECTION_NAME=emphnet_policies
TOP_K_RESULTS=5
```

5. Start Ollama:

```bash
ollama serve
```

6. Generate the chunk inspection file and build the vector index:

```bash
python src/ingestion.py
python embed_chunks.py
```

7. Launch the app:

```bash
streamlit run app.py
```

## Runtime behavior

- The ingestion step extracts text, normalizes noisy PDF content, and assigns policy metadata to each chunk.
- The retrieval layer searches semantically and with keyword matching to reduce misses on policy-specific wording.
- The generation layer prompts the local LLM with only the relevant retrieved chunks and asks it to answer from that context only.
- The UI surfaces the final answer and the source excerpts used for grounding.

## Maintaining this project

This codebase is structured around the core pipeline rather than ad hoc experiments. When editing, prefer these modules:

- `src/ingestion.py` for PDFs, chunk construction, and metadata
- `src/document_structure.py` for section and intent matching
- `src/retrieval.py` for ranking and retrieval behavior
- `src/generation.py` for answer generation and compatibility wrappers
- `src/config.py` for configuration defaults and environment validation

For operational changes, prefer updating the source modules and relevant tests rather than adding one-off scripts at the repository root.

## Testing

Run the project checks with:

```bash
pytest -q
```

This repository keeps the active test suite focused on the maintained behavior rather than obsolete benchmarking or exploratory scripts.

**Output**: Answer text + source citations formatted for UI

### 5. Streamlit UI (`app.py`)

**Why:** Provide intuitive interface for staff

- **Input**: Question in EN or AR
- **Output**: Generated answer + collapsible sources section
- **Display**:
  - Plain-language response
  - Source document + section number
  - Verbatim excerpt from policy

## Working with the Document

### Document Structure

```
Section 01: Recruitment & Selection
+-- 1.1 Purpose
+-- 1.2 Scope
+-- 1.3 Responsibilities
+-- 1.4 Policies
�   +-- Vacancy Announcement
�   +-- Application Review
�   +-- Interview Process
+-- 1.5 Procedures
�   +-- ML-HR-01.P01 Vacancy Posting
�   +-- ML-HR-01.P02 Application Collection
�   +-- ...
+-- 1.6 Related Documents
+-- 1.7 References

Section 02-07: [Similar structure]

Annexes (Annex 01-0N)
Attachments (Attachment 01-0N) [Some bilingual EN/AR]
```

### Adding a New Document

To add another policy manual:

1. Place PDF in `data/pdf/`
2. Update `PDF_PATH` in `.env`
3. If document structure differs, edit regex patterns in `src/ingestion.py`:
   - `SECTION_PATTERN` - Section header detection
   - `SUBSECTION_PATTERN` - Subsection header detection
   - `POLICY_PATTERN` - Policy topic detection
   - `PROCEDURE_PATTERN` - Procedure with reference code detection
4. Run `python src/ingestion.py` again
5. Review `data/chunks_inspection.json` to verify correct parsing

## Troubleshooting

### "Connection refused" - Ollama not running
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it:
ollama serve
```

### "Model not found" - qwen2.5:14b not pulled
```bash
ollama pull qwen2.5:14b-instruct
```

### PDF not found during ingestion
```bash
# Verify PDF path matches your setup
ls data/pdf/

# Update PDF_PATH in .env and src/ingestion.py if needed
```

### Slow embeddings on first run
- **Normal**: Sentence-transformers downloads large model (~500MB) on first use
- **Solution**: Let it complete (5-10 min), future runs are cached

### Streamlit not opening in browser
```bash
# Explicitly specify URL
streamlit run app.py --server.address=localhost --server.port=8501

# Then open: http://localhost:8501
```

### GPU not detected for embeddings
```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available())"

# If false, embeddings run on CPU (slower but works)
# To enable GPU, install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Testing & Evaluation

### Included Test Questions (`data/eval_questions.json`)

Example format for manual testing:
```json
[
  {
    "question": "What is the policy on sick leave?",
    "language": "en",
    "expected_section": "2.4",
    "expected_keywords": ["sick leave", "medical certificate"]
  },
  ...
]
```

Run tests:
```bash
python src/evaluation.py --eval-file data/eval_questions.json
```

## Performance Optimization

### Embedding Time
- **First run**: ~2 minutes per 1000 pages (downloads model)
- **Subsequent runs**: Uses cached embeddings
- **GPU acceleration**: Add `EMBEDDING_DEVICE=cuda` to `.env` (requires NVIDIA GPU + CUDA)

### Retrieval Latency
- Vector search: ~50ms
- BM25 search: ~10ms
- LLM inference: 5-30s (depends on query length + hardware)
- **Total**: ~10-40 seconds per question

### Token Limits
- LLM context window: 32,000 tokens (qwen2.5:14b)
- Each retrieved chunk: ~100-500 tokens
- Allowing 5 chunks + question/answer: ~5,000 tokens used

## Development Notes

### Extending the Chatbot

**To add a new retrieval method:**
1. Create `src/retrieval_custom.py`
2. Implement retriever interface
3. Integrate in `src/retrieval.py`
4. Benchmark performance

**To customize prompts:**
1. Edit system prompt in `src/generation.py`
2. Add language-specific variants
3. Test on eval_questions.json

**To add more data sources:**
1. Create parallel ingestion scripts for each source
2. Merge chunks into single ChromaDB collection
3. Add source tracking metadata
4. Test hybrid document retrieval

## License & Legal

**Document Confidentiality**: The HR policy manual is proprietary to EMPHNET. Keep the PDF and generated chunks secure. Do not commit `data/pdf/` or `chroma_storage/` to public repositories.

**Dependencies**: All packages are open-source (MIT, Apache 2.0, etc.) See `requirements.txt` for details.

## Support

For issues or questions:
1. Check the **Troubleshooting** section above
2. Verify `data/chunks_inspection.json` shows correct parsing
3. Check logs from `python src/ingestion.py`
4. Ensure Ollama is running: `curl http://localhost:11434/api/tags`

## Next Steps

1. **Place the PDF** in `data/pdf/`
2. **Run ingestion** with `python src/ingestion.py`
3. **Review chunks** in `data/chunks_inspection.json`
4. **Launch UI** with `streamlit run app.py`
5. **Test questions** from `data/eval_questions.json`

---

**Built with:** LangChain, ChromaDB, Ollama, Streamlit, PyPDF, sentence-transformers Policies Chatbot
