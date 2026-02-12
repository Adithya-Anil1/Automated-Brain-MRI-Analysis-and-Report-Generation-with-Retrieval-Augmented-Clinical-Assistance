"""
RAG-Based Educational Assistant for Brain MRI Reports
======================================================
A controlled retrieval-augmented generation (RAG) module that answers
questions about a patient's MRI report using ONLY:

    Source A — the generated patient MRI report  (injected as context)
    Source B — a small verified medical-definitions knowledge base
               (retrieved from an in-memory vector store)

Safety constraints
------------------
* Keyword-based question gating blocks clinical queries BEFORE the LLM.
* The prompt explicitly forbids diagnosis, prognosis, or treatment advice.
* If the answer cannot be grounded in the provided context the LLM is
  instructed to return a standard refusal string.

Usage
-----
    from RAG_Assistant.rag_assistant import answer_query

    response = answer_query(
        user_query="What does midline shift mean in my report?",
        patient_report_text="<full report text>"
    )
    print(response)
"""

import os
import warnings
import numpy as np
from typing import List, Tuple

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from project root
except ImportError:
    pass  # python-dotenv not installed, will use os.environ directly

# ============================================================================
# 1.  REFUSAL TEMPLATES  (hard-coded safety strings)
# ============================================================================
# These are returned verbatim without ever calling the LLM.

REFUSAL_CLINICAL = (
    "I cannot answer clinical questions regarding diagnosis, prognosis, "
    "or treatment. Please consult a doctor."
)

REFUSAL_DATA = (
    "This information is not present in the generated report "
    "or verified definitions."
)

# ============================================================================
# 2.  BLOCKED KEYWORDS  (used by the question-gating step)
# ============================================================================
# Any query containing one of these words (case-insensitive) is immediately
# refused — the LLM is never invoked.

BLOCKED_KEYWORDS = [
    "treatment", "therapy", "surgery", "medication", "drug",
    "prognosis", "survival", "outcome", "chemotherapy", "radiation",
]

# ============================================================================
# 3.  GEMINI CONFIGURATION
# ============================================================================
# The API key is read from the environment.  Set it via:
#   export GEMINI_API_KEY=your_key_here        (Linux / macOS)
#   set    GEMINI_API_KEY=your_key_here        (Windows CMD)
#   $env:GEMINI_API_KEY = "your_key_here"      (PowerShell)
#
# or place it in a .env file at the project root.

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

GEMINI_AVAILABLE = False
try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # Gemini SDK not installed


# ============================================================================
# 4.  DUMMY VECTOR STORE  (in-memory, no external files)
# ============================================================================
# A minimal vector store built on NumPy for the college-project demo.
# Each "document" is a short, verified medical definition with NO treatment
# or prognosis language.

# --- Verified medical definitions (Source B) --------------------------------

MEDICAL_DEFINITIONS: List[dict] = [
    {
        "term": "Midline shift",
        "text": (
            "Midline shift refers to the displacement of brain structures "
            "from their normal central position to one side. It is measured "
            "in millimeters on axial MRI or CT images and indicates "
            "asymmetric mass effect within the cranial cavity."
        ),
    },
    {
        "term": "Peritumoral edema",
        "text": (
            "Peritumoral edema is the accumulation of excess fluid in the "
            "brain tissue surrounding a tumor. On T2-weighted and FLAIR MRI "
            "sequences it appears as a region of high signal intensity "
            "around the tumor margin."
        ),
    },
    {
        "term": "Enhancing tumor",
        "text": (
            "An enhancing tumor is a region that shows increased signal "
            "intensity on post-contrast T1-weighted MRI images after "
            "gadolinium administration. Enhancement indicates areas where "
            "the blood-brain barrier is disrupted, allowing contrast agent "
            "to accumulate."
        ),
    },
]


class DummyVectorStore:
    """
    Minimal in-memory vector store using TF-IDF-style bag-of-words
    embeddings and cosine similarity.

    This avoids any dependency on heavyweight libraries (FAISS,
    sentence-transformers, etc.) so the demo runs immediately.
    """

    def __init__(self, documents: List[dict] | None = None):
        """
        Parameters
        ----------
        documents : list[dict]
            Each dict must have a ``"text"`` key and optionally a ``"term"`` key.
        """
        self.documents: List[dict] = documents or []
        self.vocab: List[str] = []
        self.vectors: np.ndarray = np.array([])
        if self.documents:
            self._build_index()

    # --- internal helpers ---------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase and split on non-alpha characters."""
        import re
        return re.findall(r"[a-z]+", text.lower())

    def _build_index(self) -> None:
        """Build a simple bag-of-words vector for every document."""
        # 1. Collect vocabulary from all documents
        all_tokens: List[List[str]] = []
        vocab_set: set = set()
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            all_tokens.append(tokens)
            vocab_set.update(tokens)
        self.vocab = sorted(vocab_set)
        word_to_idx = {w: i for i, w in enumerate(self.vocab)}

        # 2. Vectorize each document (term-frequency)
        matrix = np.zeros((len(self.documents), len(self.vocab)))
        for row, tokens in enumerate(all_tokens):
            for tok in tokens:
                matrix[row, word_to_idx[tok]] += 1
        # L2-normalise for cosine similarity
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # avoid division by zero
        self.vectors = matrix / norms

    def _query_vector(self, query: str) -> np.ndarray:
        """Convert a query string into the same vector space."""
        vec = np.zeros(len(self.vocab))
        word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        for tok in self._tokenize(query):
            if tok in word_to_idx:
                vec[word_to_idx[tok]] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    # --- public API ---------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 2) -> List[Tuple[dict, float]]:
        """
        Retrieve the *top_k* most relevant documents for *query*.

        Returns
        -------
        list[(document_dict, similarity_score)]
        """
        if not self.documents:
            return []
        q_vec = self._query_vector(query)
        # Cosine similarity = dot product (vectors are already L2-normalised)
        scores = self.vectors @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[i], float(scores[i])) for i in top_indices]


def init_vector_store() -> DummyVectorStore:
    """
    Initialize the dummy in-memory vector store with verified
    medical definitions.  No external files are required.

    Returns
    -------
    DummyVectorStore
        Ready-to-query vector store containing 3 safe definitions.
    """
    return DummyVectorStore(documents=MEDICAL_DEFINITIONS)


# ============================================================================
# 5.  QUESTION GATING  (keyword-based safety filter)
# ============================================================================

def is_clinical_query(user_query: str) -> bool:
    """
    Check whether *user_query* contains any blocked clinical keyword.

    This runs **before** the LLM is called.  If it returns True the
    caller must return ``REFUSAL_CLINICAL`` immediately.

    Parameters
    ----------
    user_query : str
        The raw question from the user.

    Returns
    -------
    bool
        True if the query matches a blocked keyword.
    """
    query_lower = user_query.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in query_lower:
            return True
    return False


# ============================================================================
# 6.  PROMPT CONSTRUCTION  (strict template)
# ============================================================================

PROMPT_TEMPLATE = """\
You are an Educational MRI Assistant with expertise in neuroradiology.

You support probabilistic clinical reasoning — explaining typical radiologic
associations — while strictly refusing to diagnose the patient.

### CONTEXT 1: PATIENT REPORT
{patient_report}

### CONTEXT 2: KNOWLEDGE BASE
{definitions}

### KNOWLEDGE SOURCE RULES
- Patient-specific findings must come ONLY from Context 1 (Patient Report).
  Do NOT invent, assume, or extrapolate findings beyond what is described.
- Medical associations and definitions must come ONLY from Context 2
  (Knowledge Base). If no relevant association was retrieved, state:
  "This association is not described in the available verified knowledge."
- Do NOT draw on unrestricted internal medical knowledge.

### PROBABILISTIC REASONING (Allowed)
When discussing imaging findings you MAY use language such as:
  "is commonly associated with", "raises suspicion for",
  "is frequently seen in", "is characteristic of",
  "suggests but does not confirm".
The tone must remain objective and educational.

### CLINICAL LIMITATION RULE
When an explanation touches on aggressiveness, tumor grade, or tumor type:
  Naturally clarify that imaging findings alone do not establish a
  definitive diagnosis and that histopathologic confirmation is required.
  Integrate this clarification contextually — do NOT append it mechanically.

### FORBIDDEN — Hard Safety Boundary
You must NEVER:
  - Diagnose the patient ("This patient has…", "This confirms…",
    "This is definitively…", "The tumor is Grade…").
  - State a prognosis ("The prognosis is…", "Survival is…").
  - Recommend any treatment, therapy, medication, or surgery.
If the user asks for any of the above, respond ONLY with:
  "I cannot answer clinical questions regarding diagnosis, prognosis,
   or treatment. Please consult a doctor."

### RESPONSE RULES

1. Directness
   - Begin with a HEADLINE: one clear, direct sentence answering the user's question.
   - Do not use filler phrases (e.g., "Based on the context", "The report indicates").

2. Structure
   - After the HEADLINE, provide SUPPORTING DETAILS as a bulleted list.
   - Each bullet must contain exactly one idea.
   - **Bold** all measurements, volumes, and anatomical locations (e.g., **12.4 cm³**, **Right Temporal Lobe**).
   - Do not write paragraph-style prose inside bullets.

3. Content Logic (The "Anchor & Explain" Pattern)
   - First bullet(s): Extract specific findings from the PATIENT REPORT (Context 1).
   - Next bullet(s): Explain the mechanism or association using the KNOWLEDGE BASE (Context 2).
   - Explicitly connect the patient's specific value to the general concept.

4. Imaging Sign Questions
   - First, confirm if the sign is PRESENT or ABSENT in the report.
   - Then, explain the radiologic mechanism (e.g., "This indicates blood-brain barrier breakdown...").
   - Use probabilistic language for clinical associations (e.g., "commonly associated with").

5. Safety & Grounding
   - If the answer is not in the context, output ONLY: "This information is not present in the generated report or verified definitions."
   - If the user asks for diagnosis/prognosis, output ONLY: "I cannot answer clinical questions regarding diagnosis, prognosis, or treatment. Please consult a doctor."

### USER QUESTION
{user_query}
"""


def build_prompt(
    user_query: str,
    patient_report: str,
    retrieved_definitions,
) -> str:
    """
    Assemble the strict RAG prompt from the patient report and the
    retrieved medical definitions.

    Parameters
    ----------
    user_query : str
        The user's question.
    patient_report : str
        Full text of the patient's generated MRI report.
    retrieved_definitions
        Either ChromaDB query results (dict with 'documents', 'metadatas')
        or legacy DummyVectorStore results (list of tuples).

    Returns
    -------
    str
        The complete prompt string ready for the LLM.
    """
    # Format retrieved definitions into a readable block
    def_lines: List[str] = []

    if isinstance(retrieved_definitions, dict):
        # ChromaDB results format
        docs = retrieved_definitions.get("documents", [[]])[0]
        metas = retrieved_definitions.get("metadatas", [[]])[0]
        for doc_text, meta in zip(docs, metas):
            title = meta.get("title", "Definition")
            def_lines.append(f"- {title}:\n{doc_text}")
    else:
        # Legacy DummyVectorStore format: list of (doc_dict, score)
        for doc, score in retrieved_definitions:
            term = doc.get("term", "Definition")
            text = doc["text"]
            def_lines.append(f"- {term}: {text}")

    definitions_block = "\n\n".join(def_lines) if def_lines else "No definitions retrieved."

    prompt = PROMPT_TEMPLATE.format(
        patient_report=patient_report.strip(),
        definitions=definitions_block,
        user_query=user_query.strip(),
    )
    return prompt


# ============================================================================
# 7.  GEMINI LLM CALL
# ============================================================================

def call_gemini(prompt: str) -> str:
    """
    Send *prompt* to Gemini 1.5 Flash with low temperature and return
    the response text.

    Parameters
    ----------
    prompt : str
        The fully assembled RAG prompt.

    Returns
    -------
    str
        The model's response text, or ``REFUSAL_DATA`` on failure.
    """
    if not GEMINI_AVAILABLE:
        return "[Error] google-generativeai SDK is not installed."

    # Always re-read the API key from environment (in case .env was updated)
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "[Error] GEMINI_API_KEY environment variable is not set."

    # Configure the SDK with the API key
    genai.configure(api_key=api_key)

    # Use gemini-2.5-flash with low temperature for factual responses
    model = genai.GenerativeModel(model_name="gemini-2.5-flash")
    generation_config = genai.types.GenerationConfig(
        temperature=0.1,       # Low temperature — factual, deterministic
        max_output_tokens=2048, # Enough for structured answers with bullet lists
    )

    try:
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[RAG] Gemini API error: {e}")
        return REFUSAL_DATA


# ============================================================================
# 8.  MAIN ENTRY POINT — answer_query()
# ============================================================================

def answer_query(user_query: str, patient_report_text: str) -> str:
    """
    Answer a user question about a patient MRI report using a
    controlled RAG pipeline with strict safety constraints.

    Pipeline steps
    --------------
    1. **Keyword gating** — block clinical queries before LLM call.
    2. **Retrieve** top-2 definitions from the in-memory vector store.
    3. **Build prompt** — inject the patient report and definitions.
    4. **Call Gemini** — get a short, factual, explanatory answer.
    5. **Return** the response (or a safe fallback).

    Parameters
    ----------
    user_query : str
        The user's natural-language question.
    patient_report_text : str
        Full text of the generated patient MRI report.

    Returns
    -------
    str
        The assistant's answer, or a refusal string.
    """

    # ------------------------------------------------------------------
    # Step 1: SAFETY GATING — check for blocked clinical keywords
    # ------------------------------------------------------------------
    # This happens BEFORE any LLM call.  If the query is clinical, we
    # return the hard-coded refusal immediately.
    if is_clinical_query(user_query):
        return REFUSAL_CLINICAL

    # ------------------------------------------------------------------
    # Step 2: RETRIEVE relevant medical definitions (Source B)
    # ------------------------------------------------------------------
    # Try ChromaDB vector store first (full knowledge base),
    # fall back to dummy in-memory store if ChromaDB is unavailable.
    retrieved = None
    try:
        from RAG_Assistant.vector_store_builder import load_vector_store
        collection = load_vector_store()
        retrieved = collection.query(
            query_texts=[user_query],
            n_results=2,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # Fall back to dummy vector store
        vector_store = init_vector_store()
        retrieved = vector_store.retrieve(query=user_query, top_k=2)

    # ------------------------------------------------------------------
    # Step 3: BUILD PROMPT with both data sources
    # ------------------------------------------------------------------
    prompt = build_prompt(
        user_query=user_query,
        patient_report=patient_report_text,
        retrieved_definitions=retrieved,
    )

    # ------------------------------------------------------------------
    # Step 4: CALL GEMINI (gemini-2.5-flash, low temperature)
    # ------------------------------------------------------------------
    response = call_gemini(prompt)

    # ------------------------------------------------------------------
    # Step 5: RETURN the response (safe fallback on empty)
    # ------------------------------------------------------------------
    if not response or not response.strip():
        return REFUSAL_DATA

    return response


# ============================================================================
# 9.  STANDALONE TEST / DEMO
# ============================================================================
# Run this file directly to see the pipeline in action with a patient report.
#
# Usage:
#   python -m RAG_Assistant.rag_assistant                          (default patient)
#   python -m RAG_Assistant.rag_assistant <path/to/report.txt>     (custom report)
#   python -m RAG_Assistant.rag_assistant results/BraTS-GLI-00020-000  (patient folder)

if __name__ == "__main__":

    import sys
    import os

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

    # --- Resolve report path from CLI argument or default ---------------
    _REPORT_PATH = None

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        candidate = os.path.abspath(arg)

        if os.path.isfile(candidate):
            # Direct path to a .txt report file
            _REPORT_PATH = candidate
        elif os.path.isdir(candidate):
            # Patient results folder — look inside feature_extraction/
            _REPORT_PATH = os.path.join(candidate, "feature_extraction", "radiology_report.txt")
            if not os.path.isfile(_REPORT_PATH):
                # Maybe it's a raw patient folder — try results/<case_id>/ instead
                case_name = os.path.basename(candidate)
                _REPORT_PATH = os.path.join(
                    _PROJECT_DIR, "results", case_name,
                    "feature_extraction", "radiology_report.txt"
                )
                if not os.path.isfile(_REPORT_PATH):
                    print(f"  ⚠ No report found for {case_name}")
                    print(f"     Tried: {candidate}/feature_extraction/")
                    print(f"     Tried: results/{case_name}/feature_extraction/")
                    sys.exit(1)
        else:
            # Maybe it's a case ID like BraTS-GLI-00020-000
            _REPORT_PATH = os.path.join(
                _PROJECT_DIR, "results", arg,
                "feature_extraction", "radiology_report.txt"
            )
            if not os.path.isfile(_REPORT_PATH):
                print(f"  ⚠ Could not find report for: {arg}")
                print(f"     Tried: {_REPORT_PATH}")
                sys.exit(1)
    else:
        # Default patient
        _REPORT_PATH = os.path.join(
            _PROJECT_DIR, "results", "BraTS-GLI-00020-000",
            "feature_extraction", "radiology_report.txt"
        )

    # --- Load the report ------------------------------------------------
    try:
        with open(_REPORT_PATH, "r", encoding="utf-8") as f:
            SAMPLE_REPORT = f.read()
        print(f"  Loaded report: {_REPORT_PATH}")
    except FileNotFoundError:
        print(f"  ⚠ Report not found at {_REPORT_PATH}")
        sys.exit(1)

    # --- Extract case ID from path for display --------------------------
    _parts = os.path.normpath(_REPORT_PATH).split(os.sep)
    _case_id = "Unknown"
    for i, p in enumerate(_parts):
        if p == "feature_extraction" and i > 0:
            _case_id = _parts[i - 1]
            break

    # --- Interactive loop -----------------------------------------------
    print("=" * 70)
    print("  RAG Educational Assistant — Interactive Mode")
    print("=" * 70)
    print(f"\n  Patient: {_case_id}")
    print(f"  Report:  {os.path.basename(_REPORT_PATH)}")
    print(f"\n  💡 Ask questions about the patient's MRI findings.")
    print(f"  🚫 Clinical questions (treatment, prognosis) are blocked.")
    print(f"  ⌨  Type 'quit' or 'exit' to stop")
    print("=" * 70)

    import time
    while True:
        print("\n" + "-" * 70)
        user_input = input("\n💬 Your question: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Done!")
            break

        ans = answer_query(user_query=user_input, patient_report_text=SAMPLE_REPORT)
        print(f"\n📚 Answer:\n{ans}")
        print("-" * 70)
