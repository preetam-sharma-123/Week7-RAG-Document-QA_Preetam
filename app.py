"""
Document Q&A using Retrieval-Augmented Generation (RAG)
--------------------------------------------------------
Load any PDF, chunk it, embed it into a FAISS vector store, and answer
questions grounded only in the document's content (with page citations).

Usage:
    python app.py path/to/document.pdf
    python app.py                      # will prompt for a path
"""

import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3  # number of chunks retrieved per question


def get_pdf_path() -> str:
    """Get the PDF path from CLI args, or prompt the user for one."""
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = input("Enter path to a PDF document: ").strip()

    if not os.path.isfile(path):
        print(f"Error: file not found -> {path}")
        sys.exit(1)
    if not path.lower().endswith(".pdf"):
        print("Error: only PDF files are supported right now.")
        sys.exit(1)
    return path


def load_and_chunk(pdf_path: str):
    """Load a PDF and split it into overlapping text chunks."""
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        print(f"Error: could not read PDF ({e})")
        sys.exit(1)

    if not docs:
        print("Error: no readable text found in this PDF (is it scanned/image-only?)")
        sys.exit(1)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} page(s) -> split into {len(chunks)} chunk(s).")
    return chunks


def build_vectorstore(chunks, pdf_path: str):
    """Build a FAISS vector store, reusing a cached index if one exists for this file."""
    index_dir = f".faiss_index_{os.path.basename(pdf_path)}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if os.path.isdir(index_dir):
        print("Found existing index, loading from cache...")
        try:
            return FAISS.load_local(
                index_dir, embeddings, allow_dangerous_deserialization=True
            )
        except Exception:
            print("Cached index unreadable, rebuilding...")

    print("Building new vector index (embedding all chunks)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(index_dir)
    return vectorstore


def get_llm():
    if not os.getenv("GOOGLE_API_KEY"):
        print(
            "Error: GOOGLE_API_KEY not set. Add it to a .env file "
            "(GOOGLE_API_KEY=your_key_here) before running."
        )
        sys.exit(1)
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)


def answer_question(vectorstore, llm, question: str):
    results = vectorstore.similarity_search(question, k=TOP_K)

    if not results:
        return "No relevant content found in the document for that question.", []

    context = "\n\n".join(doc.page_content for doc in results)
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer is not contained in the context, say "
        "\"I couldn't find that in the document.\"\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        return f"Error calling the language model: {e}", []

    pages = sorted({doc.metadata.get("page", "?") for doc in results})
    return response.content, pages


def main():
    load_dotenv()
    pdf_path = get_pdf_path()
    chunks = load_and_chunk(pdf_path)
    vectorstore = build_vectorstore(chunks, pdf_path)
    llm = get_llm()

    print("\nDocument loaded. Ask questions about it (type 'exit' to quit).")
    while True:
        question = input("\nAsk Question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        answer, pages = answer_question(vectorstore, llm, question)
        print(f"\nAnswer: {answer}")
        if pages:
            page_list = ", ".join(str(p) for p in pages)
            print(f"(Source page(s): {page_list})")


if __name__ == "__main__":
    main()