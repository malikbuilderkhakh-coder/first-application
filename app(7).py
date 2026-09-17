import os
import re
import io
import hashlib
from typing import List, Dict, Any

import fitz  # PyMuPDF
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide",
)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

# BGE models work best when queries use this instruction.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking defaults. Chunk size is measured approximately in words here;
# the embedding model performs its own subword tokenization internally.
CHUNK_SIZE_WORDS = 450
CHUNK_OVERLAP_WORDS = 80
TOP_K = 6
MIN_SIMILARITY = 0.25


# -----------------------------
# Cached models / clients
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_api_key() -> str:
    """Read Groq key from Streamlit Secrets first, then environment."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""

    if not key:
        key = os.getenv("GROQ_API_KEY", "")

    return key.strip()


def get_groq_client() -> Groq:
    key = get_api_key()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or "
            "set it as an environment variable."
        )
    return Groq(api_key=key)


# -----------------------------
# PDF extraction
# -----------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(uploaded_file) -> List[Dict[str, Any]]:
    """Extract text page-by-page and preserve file/page metadata."""
    data = uploaded_file.getvalue()
    doc = fitz.open(stream=data, filetype="pdf")

    pages = []
    try:
        for page_number, page in enumerate(doc, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(
                    {
                        "text": text,
                        "page": page_number,
                    }
                )
    finally:
        doc.close()

    return pages


# -----------------------------
# Chunking
# -----------------------------
def chunk_page_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> List[str]:
    """
    Paragraph-aware word chunking.

    The embedding model tokenizes each resulting chunk into subword tokens.
    Word-based chunking keeps chunks readable and predictable while avoiding
    dependency on a second tokenizer just for chunk construction.
    """
    if not text.strip():
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    # If PDF extraction did not preserve paragraphs, use the whole page.
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks = []
    current_words: List[str] = []

    for paragraph in paragraphs:
        words = paragraph.split()

        # Very large paragraph: split it directly.
        if len(words) > chunk_size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-overlap:]

            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                part = words[start:end]
                if part:
                    chunks.append(" ".join(part))
                if end >= len(words):
                    break
                start = max(end - overlap, start + 1)

            current_words = []
            continue

        if len(current_words) + len(words) <= chunk_size:
            current_words.extend(words)
        else:
            if current_words:
                chunks.append(" ".join(current_words))

            overlap_words = current_words[-overlap:] if overlap else []
            current_words = overlap_words + words

            # Safety for a combined overlap + paragraph.
            if len(current_words) > chunk_size:
                chunks.append(" ".join(current_words[:chunk_size]))
                current_words = current_words[chunk_size - overlap :]

    if current_words:
        chunks.append(" ".join(current_words))

    return [c.strip() for c in chunks if c.strip()]


def build_chunks(pages: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
    chunks = []
    for page in pages:
        page_chunks = chunk_page_text(page["text"])
        for idx, chunk in enumerate(page_chunks, start=1):
            chunks.append(
                {
                    "text": chunk,
                    "file": filename,
                    "page": page["page"],
                    "chunk": idx,
                }
            )
    return chunks


# -----------------------------
# Embeddings + FAISS
# -----------------------------
def create_faiss_index(
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
):
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def search_chunks(
    question: str,
    index,
    chunks: List[Dict[str, Any]],
    model: SentenceTransformer,
    top_k: int = TOP_K,
):
    query = QUERY_PREFIX + question

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = dict(chunks[int(idx)])
        item["score"] = float(score)

        # With normalized vectors + inner product, this is cosine similarity.
        if item["score"] >= MIN_SIMILARITY:
            results.append(item)

    return results


# -----------------------------
# Prompt / answer generation
# -----------------------------
def make_context(results: List[Dict[str, Any]]) -> str:
    blocks = []

    for i, item in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {i}]\n"
            f"File: {item['file']}\n"
            f"Page: {item['page']}\n"
            f"Content:\n{item['text']}"
        )

    return "\n\n".join(blocks)


def answer_question(
    question: str,
    results: List[Dict[str, Any]],
    model_name: str,
    chat_history: List[Dict[str, str]],
) -> str:
    client = get_groq_client()

    context = make_context(results)

    system_prompt = """You are a careful PDF question-answering assistant.

Rules:
1. Answer using ONLY information supported by the supplied PDF context.
2. If the answer is not present or cannot be reasonably determined from the
   context, say: "I couldn't find that information in the uploaded PDF."
3. Do not invent facts, citations, page numbers, names, or statistics.
4. Keep the answer clear and reasonably concise.
5. When useful, mention the source file and page number in the answer.
6. If the user asks something unrelated to the uploaded documents, explain
   that you can answer questions grounded in the uploaded PDFs.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Keep only recent conversation turns so context stays manageable.
    for message in chat_history[-6:]:
        messages.append(
            {
                "role": message["role"],
                "content": message["content"],
            }
        )

    user_prompt = f"""Answer this question using the PDF context below.

QUESTION:
{question}

PDF CONTEXT:
{context}
"""

    messages.append({"role": "user", "content": user_prompt})

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.1,
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 PDF RAG Assistant")
st.caption(
    "Upload one or more PDFs → extract text → chunk → embed → search with FAISS → "
    "answer with Groq."
)

with st.sidebar:
    st.header("⚙️ Settings")

    groq_model = st.text_input(
        "Groq model",
        value=DEFAULT_GROQ_MODEL,
        help="Enter a currently available Groq chat model.",
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=2,
        max_value=10,
        value=TOP_K,
        help="Number of relevant chunks sent to the LLM.",
    )

    st.divider()
    st.write("**Embedding model**")
    st.code(EMBEDDING_MODEL)

    st.write("**Chunk size**")
    st.write(f"{CHUNK_SIZE_WORDS} words, {CHUNK_OVERLAP_WORDS}-word overlap")

    if st.button("🗑️ Clear session", use_container_width=True):
        for key in [
            "faiss_index",
            "chunks",
            "documents_hash",
            "chat_history",
            "processed_files",
            "stats",
        ]:
            st.session_state.pop(key, None)
        st.rerun()

uploaded_files = st.file_uploader(
    "Upload PDF file(s)",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    current_hash = hashlib.sha256(
        b"".join(
            hashlib.sha256(f.getvalue()).digest()
            for f in uploaded_files
        )
    ).hexdigest()

    if st.session_state.get("documents_hash") != current_hash:
        with st.spinner("Processing PDFs..."):
            all_chunks = []
            processed_files = []
            total_pages = 0

            for uploaded_file in uploaded_files:
                try:
                    pages = extract_pdf(uploaded_file)
                    total_pages += len(pages)

                    if not pages:
                        st.warning(
                            f"⚠️ No selectable text found in '{uploaded_file.name}'. "
                            "This may be a scanned/image-only PDF and requires OCR."
                        )
                        continue

                    chunks = build_chunks(pages, uploaded_file.name)
                    all_chunks.extend(chunks)
                    processed_files.append(uploaded_file.name)

                except Exception as exc:
                    st.error(
                        f"Could not process '{uploaded_file.name}': {exc}"
                    )

            if not all_chunks:
                st.error(
                    "No text chunks were created. Please upload a text-based PDF."
                )
                st.stop()

            embedding_model = load_embedding_model()
            index = create_faiss_index(all_chunks, embedding_model)

            st.session_state["faiss_index"] = index
            st.session_state["chunks"] = all_chunks
            st.session_state["documents_hash"] = current_hash
            st.session_state["processed_files"] = processed_files
            st.session_state["chat_history"] = []
            st.session_state["stats"] = {
                "files": len(processed_files),
                "pages": total_pages,
                "chunks": len(all_chunks),
            }

    stats = st.session_state.get("stats", {})
    processed_files = st.session_state.get("processed_files", [])

    st.success(
        f"Indexed {stats.get('files', 0)} PDF(s), "
        f"{stats.get('pages', 0)} text page(s), "
        f"{stats.get('chunks', 0)} chunks."
    )

    if processed_files:
        with st.expander("📄 Processed files"):
            for name in processed_files:
                st.write(f"• {name}")

    st.divider()

    if "faiss_index" not in st.session_state:
        st.warning("The PDF index is not ready yet.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("🔎 Retrieved sources"):
                    for source in message["sources"]:
                        st.write(
                            f"**{source['file']} — page {source['page']}** "
                            f"(similarity: {source['score']:.3f})"
                        )

    question = st.chat_input("Ask a question about your PDF...")

    if question:
        st.session_state["chat_history"].append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                embedding_model = load_embedding_model()

                results = search_chunks(
                    question=question,
                    index=st.session_state["faiss_index"],
                    chunks=st.session_state["chunks"],
                    model=embedding_model,
                    top_k=top_k,
                )

                if not results:
                    answer = (
                        "I couldn't find relevant information in the uploaded PDF."
                    )
                    st.markdown(answer)
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                        }
                    )
                else:
                    with st.spinner("Generating answer..."):
                        answer = answer_question(
                            question=question,
                            results=results,
                            model_name=groq_model,
                            chat_history=st.session_state["chat_history"][:-1],
                        )

                    st.markdown(answer)

                    with st.expander("🔎 Retrieved sources"):
                        for source in results:
                            st.write(
                                f"**{source['file']} — page {source['page']}** "
                                f"(similarity: {source['score']:.3f})"
                            )
                            st.caption(source["text"][:600] + "...")

                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": results,
                        }
                    )

            except Exception as exc:
                error_message = f"Error: {exc}"
                st.error(error_message)
                st.session_state["chat_history"].append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )

else:
    st.info("👆 Upload a PDF to start.")

    st.markdown(
        """
### How this RAG app works

1. **PDF extraction** — PyMuPDF extracts text page by page.
2. **Chunking** — text is divided into overlapping chunks while preserving page/file metadata.
3. **Embeddings** — `BAAI/bge-small-en-v1.5` converts chunks into semantic vectors.
4. **FAISS** — normalized vectors are stored in a FAISS inner-product index for cosine-similarity retrieval.
5. **Retrieval** — the most relevant chunks are selected for each question.
6. **Generation** — Groq generates an answer from the retrieved PDF context.
7. **Grounding** — the prompt instructs the model not to invent information outside the retrieved context.

> Note: image-only/scanned PDFs need OCR before their text can be retrieved.
"""
    )
