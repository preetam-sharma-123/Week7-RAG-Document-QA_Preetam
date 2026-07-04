# Document Q&A with RAG

A command-line Retrieval-Augmented Generation (RAG) system that lets you ask
natural-language questions about any PDF document and get answers grounded
strictly in that document's content, with source page citations.

Built as part of the Celebal Excellence Internship '26 (Week 7).

## How it works

1. **Load** — the PDF is parsed page by page with `PyPDFLoader`.
2. **Chunk** — text is split into overlapping ~500-character chunks
   (`RecursiveCharacterTextSplitter`) so retrieval can find precise, relevant
   passages instead of whole pages.
3. **Embed** — each chunk is embedded using the
   `sentence-transformers/all-MiniLM-L6-v2` model.
4. **Store** — embeddings are indexed in a local FAISS vector store, cached
   to disk so re-running on the same document doesn't re-embed from scratch.
5. **Retrieve** — for each question, the top-3 most similar chunks are
   fetched via cosine similarity search.
6. **Generate** — the retrieved chunks are passed as context to
   `gemini-2.5-flash`, which is instructed to answer only from that context
   and say so explicitly if the answer isn't present.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Gemini API key:

```
GOOGLE_API_KEY=your_key_here
```

## Usage

```bash
python app.py path/to/document.pdf
```

Or run it without an argument and you'll be prompted for a file path:

```bash
python app.py
```

Then ask questions interactively:

```
Ask Question: What programming languages does the candidate know?
Answer: ...
(Source page(s): 1)

Ask Question: exit
```

## Example

The included `Preetam_Sharma_Resume_Inn.pdf` is used as a sample document —
try asking things like "What internships has this person completed?" or
"What projects are listed?"

See `screenshots/` for example runs.

## Notes / limitations

- Currently supports PDF input only.
- Answers are only as good as the retrieved chunks — very short or vague
  questions may retrieve less relevant context.
- The FAISS index is cached per-file in a `.faiss_index_<filename>` folder;
  delete it if you want to force a rebuild after editing the source PDF.
