"""
FastAPI backend for the Brain MRI Analysis Pipeline.

Wraps the CLI-based pipeline (run_full_pipeline.py) and exposes
REST endpoints for uploading scans, tracking progress, retrieving
reports/metrics, and chatting with the RAG assistant.
"""

import json
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Job store (in-memory, guarded by a lock)
# ---------------------------------------------------------------------------
JOB_STORE: dict = {}
JOB_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Brain MRI Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.resolve()
SESSIONS_DIR = BASE_DIR / "sessions"

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str

# ---------------------------------------------------------------------------
# Blocked keywords for the /chat endpoint
# ---------------------------------------------------------------------------
BLOCKED_KEYWORDS = [
    "treatment", "prognosis", "diagnose", "diagnosis",
    "should i", "will the patient", "survival",
    "chemotherapy", "radiation", "surgery",
    "grade", "malignant", "benign", "cancer",
]

CLINICAL_REFUSAL = (
    "This question requires clinical judgment. "
    "Please consult a qualified radiologist."
)

# ---------------------------------------------------------------------------
# Stage detection helpers
# ---------------------------------------------------------------------------
STAGE_MAP = {
    "STAGE:done":        ("done",        "done",        100),
    "STAGE:error":       ("error",       "error",       None),
    "STAGE:exporting":   ("running",     "exporting",   90),
    "STAGE:generating":  ("running",     "generating",  70),
    "STAGE:extracting":  ("running",     "extracting",  40),
    "STAGE:segmenting":  ("running",     "segmenting",  10),
}

# Order matters: check from latest stage to earliest so the *last*
# marker found in the log wins.
STAGE_ORDER = [
    "STAGE:done",
    "STAGE:error",
    "STAGE:exporting",
    "STAGE:generating",
    "STAGE:extracting",
    "STAGE:segmenting",
]


def _parse_log(log_path: Path) -> dict:
    """Parse the pipeline log and return a status dict."""
    if not log_path.exists():
        return {
            "status": "running",
            "stage": "segmenting",
            "progress_pct": 0,
            "error_message": None,
        }

    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.strip().splitlines()

    # Walk from earliest to latest so we always keep the *most recent* marker.
    detected_status = "running"
    detected_stage = "segmenting"
    detected_pct = 10
    error_message = None

    for marker in STAGE_ORDER[::-1]:          # segmenting → … → done/error
        if marker in text:
            status, stage, pct = STAGE_MAP[marker]
            detected_status = status
            detected_stage = stage
            if pct is not None:
                detected_pct = pct

    # Re-scan for the latest marker to override with the most recent one
    latest_marker = None
    latest_pos = -1
    for marker in STAGE_ORDER:
        pos = text.rfind(marker)
        if pos > latest_pos:
            latest_pos = pos
            latest_marker = marker

    if latest_marker is not None:
        status, stage, pct = STAGE_MAP[latest_marker]
        detected_status = status
        detected_stage = stage
        if pct is not None:
            detected_pct = pct

    if detected_status == "error" and lines:
        error_message = lines[-1]

    return {
        "status": detected_status,
        "stage": detected_stage,
        "progress_pct": detected_pct,
        "error_message": error_message,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_case_folder(input_dir: Path) -> Path:
    """
    After extracting a zip into input_dir, locate the actual BraTS case folder.

    Three layouts are handled:
      1. Zip contained a single subfolder  → use that subfolder
      2. Zip files were placed directly     → use input_dir itself
      3. Multiple subfolders               → use the first one that contains NIfTI files
    """
    # Direct NIfTI files in input_dir?
    if list(input_dir.glob("*.nii.gz")) or list(input_dir.glob("*.nii")):
        return input_dir

    subdirs = [d for d in sorted(input_dir.iterdir()) if d.is_dir()]
    if not subdirs:
        return input_dir

    if len(subdirs) == 1:
        return subdirs[0]

    # Multiple subdirs – prefer the first one that has NIfTI files
    for d in subdirs:
        if list(d.glob("*.nii.gz")) or list(d.glob("*.nii")):
            return d

    return subdirs[0]  # last-resort fallback


def get_output_dir(job_id: str) -> Path:
    """
    Return the feature-extraction output directory for this job:
        <project_root>/results/<case_id>/feature_extraction
    All pipeline output files (report, PDF, JSON) live here.
    """
    with JOB_LOCK:
        case_id = JOB_STORE.get(job_id, {}).get("case_id", "")
    return BASE_DIR / "results" / case_id / "feature_extraction"


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(job_id: str, case_folder: Path, log_path: Path):
    """Execute run_full_pipeline.py in a subprocess; update JOB_STORE on exit."""
    try:
        with open(log_path, "w", encoding="utf-8") as log_fh:
            proc = subprocess.run(
                [
                    sys.executable,
                    str(BASE_DIR / "run_full_pipeline.py"),
                    str(case_folder),   # positional arg — matches main()'s 'case_folder'
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(BASE_DIR),
            )

        # Determine final status from log
        result = _parse_log(log_path)
        final_status = result["status"]
        final_stage = result["stage"]

        # If subprocess exited with non-zero and we haven't already
        # recorded an error marker, force error status.
        if proc.returncode != 0 and final_status != "error":
            final_status = "error"
            final_stage = "error"

        with JOB_LOCK:
            JOB_STORE[job_id]["status"] = final_status
            JOB_STORE[job_id]["stage"] = final_stage

    except Exception as exc:
        with JOB_LOCK:
            JOB_STORE[job_id]["status"] = "error"
            JOB_STORE[job_id]["stage"] = "error"
        # Append to log if possible
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\nSTAGE:error\n{exc}\n")
        except OSError:
            pass


# =========================================================================
# ENDPOINTS
# =========================================================================

# POST /api/analyze --------------------------------------------------------

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Upload a .zip of a BraTS folder, start the pipeline."""

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Upload must be a .zip file.")

    job_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / job_id
    input_dir = session_dir / "input"
    output_dir = session_dir / "output"
    log_path = session_dir / "pipeline.log"

    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save & extract zip
    zip_path = session_dir / "upload.zip"
    contents = await file.read()
    zip_path.write_bytes(contents)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(input_dir)
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid zip archive.")

    # Locate the actual BraTS case folder and derive case_id
    case_folder = _find_case_folder(input_dir)
    case_id = case_folder.name

    # Register job (store case_id so result-path helpers work)
    with JOB_LOCK:
        JOB_STORE[job_id] = {
            "status": "running",
            "stage": "segmenting",
            "case_id": case_id,
        }

    # Launch pipeline in a background thread
    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, case_folder, log_path),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id}


# GET /api/status/{job_id} -------------------------------------------------

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    """Return current pipeline status by inspecting the log file."""

    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Job not found.")

    log_path = SESSIONS_DIR / job_id / "pipeline.log"
    return _parse_log(log_path)


# GET /api/report/{job_id} -------------------------------------------------

@app.get("/api/report/{job_id}")
async def report_text(job_id: str):
    """Return the plain-text radiology report."""

    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Job not found.")

    # Pipeline writes to  results/<case_id>/feature_extraction/radiology_report.txt
    report_path = get_output_dir(job_id) / "radiology_report.txt"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not generated yet.")

    text = report_path.read_text(encoding="utf-8")
    return Response(content=text, media_type="text/plain")


# GET /api/report/{job_id}/pdf ----------------------------------------------

@app.get("/api/report/{job_id}/pdf")
async def report_pdf(job_id: str):
    """Return the PDF radiology report."""

    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Job not found.")

    # Pipeline writes to  results/<case_id>/feature_extraction/radiology_report.pdf
    pdf_path = get_output_dir(job_id) / "radiology_report.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not generated yet.")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename="radiology_report.pdf",
    )


# GET /api/metrics/{job_id} ------------------------------------------------

@app.get("/api/metrics/{job_id}")
async def metrics(job_id: str):
    """Return selected tumour metrics from the pipeline's llm_ready_summary.json."""

    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Job not found.")

    # Pipeline writes  results/<case_id>/feature_extraction/llm_ready_summary.json
    summary_path = get_output_dir(job_id) / "llm_ready_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail="Metrics not available yet.")

    data = json.loads(summary_path.read_text(encoding="utf-8"))

    # --- map nested pipeline keys to flat API response ---
    # tumor_characteristics
    tc = data.get("tumor_characteristics", {})
    # location
    loc = data.get("location", {})
    # mass_effect
    me = data.get("mass_effect", {})
    # quality_metrics
    qm = data.get("quality_metrics", {})

    return {
        # 'volume_cm3' is Whole Tumor in the pipeline's llm_ready_summary
        "whole_tumor_volume_cm3": float(tc.get("volume_cm3", 0.0)),
        "enhancing_volume_cm3":   float(tc.get("enhancing_volume_cm3", 0.0)),
        "necrotic_volume_cm3":    float(tc.get("necrotic_volume_cm3", 0.0)),
        "edema_volume_cm3":       float(tc.get("edema_volume_cm3", 0.0)),
        # location fields
        "hemisphere":             loc.get("hemisphere", "unknown"),
        "lobe":                   loc.get("primary_lobe", "unknown"),
        # mass_effect.midline_shift_mm
        "midline_shift_mm":       float(me.get("midline_shift_mm", 0.0)),
        # no dice score in pipeline output — always null
        "dice_score":             None,
        # quality_metrics.segmentation_grade  (e.g. "Good", "Fair", …)
        "segmentation_quality":   qm.get("segmentation_grade", "unknown"),
    }


# POST /api/chat/{job_id} --------------------------------------------------

@app.post("/api/chat/{job_id}")
async def chat(job_id: str, body: ChatRequest):
    """Answer a question about the patient report via the RAG assistant."""

    with JOB_LOCK:
        if job_id not in JOB_STORE:
            raise HTTPException(status_code=404, detail="Job not found.")

    question = body.question

    # Backend safety gate — reject clinical queries
    q_lower = question.lower()
    for keyword in BLOCKED_KEYWORDS:
        if keyword in q_lower:
            raise HTTPException(status_code=400, detail=CLINICAL_REFUSAL)

    # Read the report from the real pipeline output location
    report_path = get_output_dir(job_id) / "radiology_report.txt"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not available yet.")

    report_text = report_path.read_text(encoding="utf-8")

    # Import and invoke the RAG assistant
    try:
        from RAG_Assistant.rag_assistant import answer_query
        answer = answer_query(
            user_query=question,
            patient_report_text=report_text,
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="RAG assistant module not available.",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG assistant error: {exc}",
        )

    return {"answer": answer}
