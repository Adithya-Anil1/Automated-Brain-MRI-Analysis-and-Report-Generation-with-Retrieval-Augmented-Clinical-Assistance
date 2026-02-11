"""
Vector Store Builder — Persistent ChromaDB from Markdown Knowledge Base
========================================================================
This module reads verified medical-definition files from ``knowledge_base/``,
embeds them using a lightweight sentence-transformer model, and stores
them in a persistent ChromaDB collection under ``vector_store/``.

Design decisions
----------------
1. **One chunk per file** — Each knowledge-base article is short (< 500
   words) and covers a single well-defined medical concept.  Splitting
   it into smaller chunks would break the coherent explanation and
   reduce retrieval quality for a domain-specific QA system.

2. **Metadata improves retrieval** — We store the TITLE and KEYWORDS
   from each file as ChromaDB metadata fields.  ChromaDB can filter on
   metadata *before* similarity search, so a query like "What is midline
   shift?" can be matched both by semantic embedding similarity AND by
   keyword overlap in the metadata, improving recall and precision.

3. **Persistent storage** — The ``vector_store/`` directory keeps the
   ChromaDB data on disk so the collection does not need to be rebuilt
   on every application start.  ``load_vector_store()`` simply reconnects
   to the existing collection.

Usage
-----
    # Build (run once, or when knowledge_base/ files change)
    from RAG_Assistant.vector_store_builder import build_vector_store
    collection = build_vector_store()

    # Load (on every application start)
    from RAG_Assistant.vector_store_builder import load_vector_store
    collection = load_vector_store()

    # Query
    results = collection.query(query_texts=["What is peritumoral edema?"], n_results=2)
"""

import os
import glob
from typing import Tuple, List, Optional

import chromadb
from chromadb.utils import embedding_functions

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths are relative to the RAG_Assistant/ package directory so the
# module works correctly regardless of the caller's working directory.
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

KNOWLEDGE_BASE_DIR = os.path.join(_PACKAGE_DIR, "knowledge_base")
VECTOR_STORE_DIR   = os.path.join(_PACKAGE_DIR, "vector_store")

# ChromaDB collection name  (one collection for the whole knowledge base)
COLLECTION_NAME = "brain_mri_knowledge"

# Sentence-transformer model used for embedding.
# all-MiniLM-L6-v2 is a compact (80 MB), fast model that produces 384-dim
# embeddings — ideal for a college-project demo on a laptop.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================================
# 1.  MARKDOWN PARSER
# ============================================================================

def parse_md_file(filepath: str) -> Tuple[str, dict]:
    """
    Read a knowledge-base ``.md`` file and extract its content and metadata.

    Expected file format::

        TITLE: Some Title
        KEYWORDS: keyword1, keyword2, ...
        VERSION: 1.0
        ---
        <markdown body>

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the ``.md`` file.

    Returns
    -------
    tuple[str, dict]
        - **document_text** : The full combined text
          (``TITLE + KEYWORDS + body``) used for embedding.
          Combining these ensures the embedding captures both the
          topic label *and* the detailed explanation.
        - **metadata** : Dict with keys ``title``, ``keywords``,
          ``version``, and ``source`` (filename).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # --- Split header from body at the '---' separator --------------------
    metadata: dict = {
        "title": "",
        "keywords": "",
        "version": "",
        "source": os.path.basename(filepath),
    }
    body = raw  # fallback: whole file is the body

    if "---" in raw:
        parts = raw.split("---", maxsplit=1)
        header_block = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""

        # Parse key-value pairs from the header
        for line in header_block.splitlines():
            if line.startswith("TITLE:"):
                metadata["title"] = line[len("TITLE:"):].strip()
            elif line.startswith("KEYWORDS:"):
                metadata["keywords"] = line[len("KEYWORDS:"):].strip()
            elif line.startswith("VERSION:"):
                metadata["version"] = line[len("VERSION:"):].strip()

    # --- Combine title + keywords + body into one document ----------------
    # WHY: By prepending the title and keywords to the body text we ensure
    # that the embedding vector captures the *topic identity* of the
    # article, not just its detailed content.  This significantly improves
    # retrieval when the user's query is short (e.g., "midline shift").
    combined_parts = []
    if metadata["title"]:
        combined_parts.append(f"Title: {metadata['title']}")
    if metadata["keywords"]:
        combined_parts.append(f"Keywords: {metadata['keywords']}")
    combined_parts.append(body)

    document_text = "\n\n".join(combined_parts)

    return document_text, metadata


# ============================================================================
# 2.  BUILD VECTOR STORE
# ============================================================================

def build_vector_store(
    knowledge_dir: Optional[str] = None,
    store_dir: Optional[str] = None,
) -> chromadb.Collection:
    """
    Read all ``.md`` files from ``knowledge_base/``, embed them, and
    store them in a persistent ChromaDB collection.

    Each file is stored as **one chunk** (no splitting).

    WHY ONE CHUNK PER FILE?
    -----------------------
    Our knowledge-base files are short, self-contained definitions
    (< 500 words each).  Splitting them would:
      • Break coherent explanations mid-sentence.
      • Create tiny fragments that lose semantic context.
      • Add unnecessary complexity for a small, curated corpus.
    Keeping each file as a single document preserves the full
    meaning and ensures the retrieved context is always complete.

    Parameters
    ----------
    knowledge_dir : str, optional
        Path to the folder containing ``.md`` files.
        Defaults to ``knowledge_base/`` inside this package.
    store_dir : str, optional
        Path for the persistent ChromaDB storage.
        Defaults to ``vector_store/`` inside this package.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection with all documents indexed.
    """
    knowledge_dir = knowledge_dir or KNOWLEDGE_BASE_DIR
    store_dir     = store_dir or VECTOR_STORE_DIR

    # --- Discover .md files -----------------------------------------------
    md_pattern = os.path.join(knowledge_dir, "*.md")
    md_files = sorted(glob.glob(md_pattern))

    if not md_files:
        raise FileNotFoundError(
            f"No .md files found in '{knowledge_dir}'. "
            "Please add knowledge-base articles before building."
        )

    print(f"[VectorStore] Found {len(md_files)} .md file(s) in '{knowledge_dir}'")

    # --- Parse all files --------------------------------------------------
    documents: List[str]  = []
    metadatas: List[dict] = []
    ids: List[str]        = []

    for filepath in md_files:
        doc_text, meta = parse_md_file(filepath)
        documents.append(doc_text)
        metadatas.append(meta)
        # Use the filename (without extension) as a stable, unique ID
        doc_id = os.path.splitext(os.path.basename(filepath))[0]
        ids.append(doc_id)
        print(f"  ✓ Parsed: {meta['source']}  (title: {meta['title']})")

    # --- Create persistent ChromaDB client --------------------------------
    os.makedirs(store_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=store_dir)

    # --- Set up the embedding function ------------------------------------
    # sentence-transformers/all-MiniLM-L6-v2 produces 384-dimensional
    # embeddings.  It runs on CPU and is small enough for a laptop demo.
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    # --- Create (or overwrite) the collection -----------------------------
    # Using get_or_create ensures idempotency.  To force a rebuild, we
    # delete the existing collection first.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"[VectorStore] Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"description": "Brain MRI verified medical definitions"},
    )

    # --- Add all documents in one batch -----------------------------------
    # WHY METADATA IMPROVES RETRIEVAL:
    # ChromaDB stores metadata alongside each document.  During queries
    # you can filter by metadata fields (e.g., title or keywords) to
    # narrow down candidates *before* the similarity search.  Even
    # without explicit filtering, the metadata is available in the
    # returned results, which lets the downstream LLM see the topic
    # label and keywords — improving answer grounding.
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"[VectorStore] Indexed {collection.count()} document(s) "
          f"into collection '{COLLECTION_NAME}'")
    print(f"[VectorStore] Persistent store saved to '{store_dir}'")

    return collection


# ============================================================================
# 3.  LOAD VECTOR STORE
# ============================================================================

def load_vector_store(
    store_dir: Optional[str] = None,
) -> chromadb.Collection:
    """
    Load an existing persistent ChromaDB collection from disk.

    Call this on every application start instead of ``build_vector_store()``
    to avoid re-indexing.  The embeddings and metadata are read directly
    from the ``vector_store/`` directory.

    Parameters
    ----------
    store_dir : str, optional
        Path to the persistent ChromaDB storage.
        Defaults to ``vector_store/`` inside this package.

    Returns
    -------
    chromadb.Collection
        The loaded ChromaDB collection, ready for ``.query()`` calls.

    Raises
    ------
    ValueError
        If the collection does not exist (call ``build_vector_store()`` first).
    """
    store_dir = store_dir or VECTOR_STORE_DIR

    if not os.path.isdir(store_dir):
        raise ValueError(
            f"Vector store directory '{store_dir}' does not exist. "
            "Run build_vector_store() first."
        )

    client = chromadb.PersistentClient(path=store_dir)

    # Use the same embedding function so queries are embedded consistently
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception as e:
        raise ValueError(
            f"Collection '{COLLECTION_NAME}' not found in '{store_dir}'. "
            "Run build_vector_store() first."
        ) from e

    print(f"[VectorStore] Loaded collection '{COLLECTION_NAME}' "
          f"with {collection.count()} document(s)")

    return collection


# ============================================================================
# 4.  EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("  Vector Store Builder — Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: BUILD the vector store from knowledge_base/ .md files
    # ------------------------------------------------------------------
    print("\n📦 Building vector store from knowledge_base/ ...\n")
    collection = build_vector_store()

    # ------------------------------------------------------------------
    # Step 2: LOAD the vector store (simulating an application restart)
    # ------------------------------------------------------------------
    print("\n📂 Loading vector store from disk ...\n")
    collection = load_vector_store()

    # ------------------------------------------------------------------
    # Step 3: QUERY the vector store with sample questions
    # ------------------------------------------------------------------
    sample_queries = [
        "What is midline shift?",
        "Tell me about peritumoral edema on FLAIR",
        "How is tumor volume calculated?",
        "What does enhancement mean on MRI?",
        "What are the different MRI sequences used?",
    ]

    print("\n🔍 Running sample queries ...\n")
    for query in sample_queries:
        results = collection.query(
            query_texts=[query],
            n_results=2,  # Retrieve top-2 most relevant documents
        )
        print(f"Q: {query}")
        for i, (doc_id, distance) in enumerate(
            zip(results["ids"][0], results["distances"][0])
        ):
            # ChromaDB returns L2 distances; lower = more similar
            metadata = results["metadatas"][0][i]
            title = metadata.get("title", doc_id)
            print(f"   {i+1}. {title}  (distance: {distance:.4f})")
        print()

    print("=" * 70)
    print("  Done!  Vector store is ready at: vector_store/")
    print("=" * 70)
