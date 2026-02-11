# RAG — Educational Assistant for Brain MRI Reports

A controlled **Retrieval-Augmented Generation (RAG)** module that answers
questions about a patient's MRI report with strict safety constraints.

## Architecture

```
User Question
     │
     ▼
┌──────────────────┐   matches keyword?   ┌──────────────────┐
│  Question Gating │ ──── YES ──────────► │ REFUSAL_CLINICAL │
│  (keyword check) │                      └──────────────────┘
└────────┬─────────┘
         │ NO
         ▼
┌──────────────────┐
│  Retrieve Top-2  │  ◄── Dummy Vector Store (in-memory)
│  Definitions     │      3 verified medical definitions
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Build Prompt    │  ◄── Patient Report (Source A) +
│  (strict rules)  │      Retrieved Definitions (Source B)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Gemini 1.5 Flash│  temperature = 0.1
│  (LLM call)      │  max_tokens = 512
└────────┬─────────┘
         │
         ▼
     Response
```

## Quick Start

```python
from RAG.rag_assistant import answer_query

report = "..."  # Full MRI report text
answer = answer_query("What does midline shift mean?", report)
print(answer)
```

### Standalone demo

```bash
cd "AI-Powered Brain MRI Assistant"
python -m RAG.rag_assistant
```

## Safety Constraints

| Layer | Mechanism | Outcome |
|-------|-----------|---------|
| **Pre-LLM** | Keyword gating (`treatment`, `prognosis`, …) | Hard-coded refusal — LLM never called |
| **Prompt** | Strict rules section forbids external knowledge | LLM constrained to provided context |
| **Fallback** | `REFUSAL_DATA` constant | Returned when context is insufficient |

## Files

| File | Purpose |
|------|---------|
| `rag_assistant.py` | Main module — `answer_query()`, vector store, gating, prompt, Gemini call |
| `__init__.py` | Package marker |

## Requirements

- `numpy` (for the dummy vector store)
- `google-generativeai` (for Gemini API calls)
- Environment variable `GEMINI_API_KEY` must be set
