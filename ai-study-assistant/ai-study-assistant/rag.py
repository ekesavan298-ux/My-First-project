"""
RAG (Retrieval-Augmented Generation) Pipeline.
Implements:
1. PyPDF text extraction
2. Text cleaning & chunking
3. Sentence Transformers embedding (all-MiniLM-L6-v2)
4. FAISS IndexFlatIP (cosine similarity with normalized embeddings)
5. Vector store persistence
6. Retrieval & Gemini-powered grounded Q&A with source attribution
"""

import os
import sys
import re
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Windows DLL directory resolution for PyTorch & FAISS in virtual environment
if sys.platform == "win32":
    try:
        site_packages = Path(sys.executable).parent.parent / "Lib" / "site-packages"
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))
    except Exception:
        pass

import pypdf

from config import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    generate_gemini_text
)
import memory

# Persistent index file paths
FAISS_INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
METADATA_FILE = VECTOR_STORE_DIR / "chunks_metadata.json"

# In-memory singletons
_EMBEDDING_MODEL = None
_FAISS_INDEX = None
_CHUNKS_METADATA: List[Dict[str, Any]] = []

def get_embedding_model():
    """
    Lazy-loads and caches the SentenceTransformer embedding model.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EMBEDDING_MODEL

def clean_text(text: str) -> str:
    """
    Cleans raw text by removing non-printable characters, 
    normalizing whitespace and line breaks.
    """
    if not text:
        return ""
    # Replace carriage returns and excessive tabs
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace more than two consecutive newlines with two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespaces
    return text.strip()

def extract_text_from_pdf(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from a PDF using PyPDF.
    Returns a list of dicts: [{"page": 1, "text": "..."}]
    Raises ValueError if the PDF is unreadable, empty, or scanned without OCR text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    pages_data = []
    total_text_length = 0

    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        raise ValueError(f"Could not read PDF file '{path.name}'. File may be corrupt or encrypted: {e}")

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise ValueError(f"PDF '{path.name}' contains 0 pages.")

    for idx, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        
        cleaned = clean_text(page_text)
        if cleaned:
            total_text_length += len(cleaned)
            pages_data.append({
                "page": idx,
                "text": cleaned
            })

    if total_text_length < 25:
        raise ValueError(
            f"The uploaded PDF '{path.name}' contains no extractable text. "
            "It appears to be a scanned document, image-only, or empty. "
            "Please upload a PDF containing selectable digital text."
        )

    return pages_data

def chunk_text(
    pages_data: List[Dict[str, Any]],
    file_name: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Splits page-level text into overlapping chunks while preserving page numbers and source metadata.
    """
    chunks = []
    chunk_counter = 0

    for item in pages_data:
        page_num = item["page"]
        text = item["text"]
        
        # If the page text is shorter than chunk size, store as a single chunk
        if len(text) <= chunk_size:
            chunk_counter += 1
            chunks.append({
                "chunk_id": chunk_counter,
                "file_name": file_name,
                "page": page_num,
                "text": text
            })
            continue

        # Sliding window chunking
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_str = text[start:end]

            # Try to break at a natural boundary (sentence or newline) if not at end
            if end < len(text):
                last_break = max(chunk_str.rfind(". "), chunk_str.rfind("\n"), chunk_str.rfind(" "))
                if last_break > chunk_size * 0.5:
                    chunk_str = chunk_str[:last_break + 1]
                    end = start + len(chunk_str)

            cleaned_chunk = chunk_str.strip()
            if len(cleaned_chunk) > 20:
                chunk_counter += 1
                chunks.append({
                    "chunk_id": chunk_counter,
                    "file_name": file_name,
                    "page": page_num,
                    "text": cleaned_chunk
                })

            start = end - chunk_overlap
            if start >= len(text):
                break

    return chunks

def save_vector_store():
    """
    Persists FAISS index and chunk metadata to disk.
    """
    global _FAISS_INDEX, _CHUNKS_METADATA
    import faiss

    if _FAISS_INDEX is not None:
        VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(_FAISS_INDEX, str(FAISS_INDEX_FILE))

        with open(METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(_CHUNKS_METADATA, f, indent=2, ensure_ascii=False)

def load_vector_store():
    """
    Loads FAISS index and chunk metadata from disk if available.
    """
    global _FAISS_INDEX, _CHUNKS_METADATA
    import faiss

    if _FAISS_INDEX is None and FAISS_INDEX_FILE.exists() and METADATA_FILE.exists():
        try:
            _FAISS_INDEX = faiss.read_index(str(FAISS_INDEX_FILE))
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                _CHUNKS_METADATA = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing vector store: {e}")
            _FAISS_INDEX = None
            _CHUNKS_METADATA = []

def clear_vector_store():
    """
    Clears the active in-memory and on-disk vector store.
    """
    global _FAISS_INDEX, _CHUNKS_METADATA
    _FAISS_INDEX = None
    _CHUNKS_METADATA = []
    if FAISS_INDEX_FILE.exists():
        try:
            FAISS_INDEX_FILE.unlink()
        except OSError:
            pass
    if METADATA_FILE.exists():
        try:
            METADATA_FILE.unlink()
        except OSError:
            pass

def process_and_index_pdf(pdf_path: str | Path, original_filename: str) -> Dict[str, Any]:
    """
    Full pipeline to ingest a PDF:
    1. Extract page text via PyPDF
    2. Split into overlapping chunks
    3. Generate SentenceTransformer embeddings
    4. Normalize vectors and insert into FAISS IndexFlatIP
    5. Persist index and metadata
    6. Record topic in student memory
    """
    global _FAISS_INDEX, _CHUNKS_METADATA
    import faiss

    # Extract text
    pages_data = extract_text_from_pdf(pdf_path)
    total_pages = len(pages_data)

    # Chunk text
    chunks = chunk_text(pages_data, original_filename)
    if not chunks:
        raise ValueError(f"No usable text chunks could be extracted from '{original_filename}'.")

    # Generate embeddings
    embedder = get_embedding_model()
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=False)

    # Cast to float32 and normalize for cosine similarity via inner product
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    # Initialize or append to FAISS index
    load_vector_store()
    if _FAISS_INDEX is None:
        _FAISS_INDEX = faiss.IndexFlatIP(dimension)
        _CHUNKS_METADATA = []

    _FAISS_INDEX.add(embeddings)
    _CHUNKS_METADATA.extend(chunks)

    # Save vector store to disk
    save_vector_store()

    # Track topic / document in student memory
    clean_name = Path(original_filename).stem.replace("_", " ").title()
    memory.add_topic_studied(clean_name)

    return {
        "file_name": original_filename,
        "total_pages": total_pages,
        "total_chunks": len(chunks),
        "total_index_size": len(_CHUNKS_METADATA),
        "status": "Success"
    }

def get_vector_store_status() -> Dict[str, Any]:
    """
    Returns summary statistics for the vector database.
    """
    load_vector_store()
    indexed_files = list({c.get("file_name", "Unknown") for c in _CHUNKS_METADATA})
    return {
        "is_ready": _FAISS_INDEX is not None and _FAISS_INDEX.ntotal > 0,
        "total_chunks": len(_CHUNKS_METADATA),
        "indexed_files": indexed_files
    }

def search_study_material(query: str, top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """
    Converts query into embedding, searches FAISS IndexFlatIP,
    and returns the top_k most relevant chunks with similarity scores.
    """
    global _FAISS_INDEX, _CHUNKS_METADATA
    import faiss

    load_vector_store()
    if _FAISS_INDEX is None or _FAISS_INDEX.ntotal == 0:
        return []

    embedder = get_embedding_model()
    query_vec = embedder.encode([query], convert_to_numpy=True)
    query_vec = query_vec.astype(np.float32)
    faiss.normalize_L2(query_vec)

    k = min(top_k, _FAISS_INDEX.ntotal)
    scores, indices = _FAISS_INDEX.search(query_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_CHUNKS_METADATA):
            continue
        chunk = _CHUNKS_METADATA[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)

    return results

def answer_question(query: str, top_k: int = TOP_K_RESULTS) -> Dict[str, Any]:
    """
    RAG QA pipeline:
    1. Search FAISS for relevant chunks
    2. Format prompt with context
    3. Generate answer via Gemini
    4. Record question in student memory
    5. Return answer and source citations
    """
    status = get_vector_store_status()
    if not status["is_ready"]:
        return {
            "answer": "No study materials have been uploaded yet. Please upload a PDF in the '📚 Study Materials' section first.",
            "sources": [],
            "retrieved_context": ""
        }

    relevant_chunks = search_study_material(query, top_k=top_k)

    # Check if retrieval returned usable results
    # With normalized IndexFlatIP, score is cosine similarity in [-1, 1]
    # If the top score is very poor (< 0.15), flag that material likely doesn't have it
    best_score = relevant_chunks[0]["score"] if relevant_chunks else -1.0

    if not relevant_chunks or best_score < 0.18:
        # Save question to memory
        memory.add_question(query, topic="General Study")
        return {
            "answer": "I couldn't find this information in the uploaded study material.",
            "sources": [],
            "retrieved_context": ""
        }

    # Build context string
    context_blocks = []
    for i, c in enumerate(relevant_chunks, start=1):
        context_blocks.append(
            f"[Source {i} - File: {c.get('file_name', 'Document')} | Page {c.get('page', '?')}]\n"
            f"{c.get('text', '')}"
        )
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "You are an expert AI Learning & Study Assistant. "
        "Answer the student's question accurately, clearly, and primarily using the provided study material context.\n\n"
        "CRITICAL RULES:\n"
        "1. If the answer cannot be found in or directly inferred from the provided context, clearly say:\n"
        "'I couldn't find this information in the uploaded study material.'\n"
        "2. Do NOT hallucinate or make up facts outside the material.\n"
        "3. Structure your response with clear explanations, bullet points, or definitions where appropriate to aid student learning.\n"
        "4. Be encouraging, concise, and academically sound."
    )

    prompt = (
        f"STUDY MATERIAL CONTEXT:\n"
        f"---------------------\n"
        f"{context_text}\n"
        f"---------------------\n\n"
        f"STUDENT QUESTION: {query}\n\n"
        f"ANSWER:"
    )

    try:
        answer_text = generate_gemini_text(prompt, system_instruction=system_prompt)
    except Exception as e:
        answer_text = f"Could not generate answer due to an AI model error: {e}"

    # Log question in student memory
    dominant_topic = relevant_chunks[0].get("file_name", "Study Material").replace(".pdf", "").replace("_", " ")
    memory.add_question(query, topic=dominant_topic)

    # Format sources
    sources = []
    for c in relevant_chunks:
        sources.append({
            "file_name": c.get("file_name", "Document"),
            "page": c.get("page", 1),
            "snippet": c.get("text", "")[:180] + "..." if len(c.get("text", "")) > 180 else c.get("text", ""),
            "score": round(c.get("score", 0.0), 3)
        })

    return {
        "answer": answer_text,
        "sources": sources,
        "retrieved_context": context_text
    }
