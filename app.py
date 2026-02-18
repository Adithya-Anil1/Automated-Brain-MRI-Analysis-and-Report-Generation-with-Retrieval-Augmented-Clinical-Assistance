"""
Streamlit frontend for the Brain MRI Analysis system.
Connects to the FastAPI backend at http://localhost:8000.
"""

import time
from datetime import datetime

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000"

STAGE_LABELS = {
    "segmenting": "Tumor Segmentation (nnUNet)",
    "extracting": "Feature Extraction (6 steps)",
    "generating": "Report Generation (Gemini)",
    "exporting": "PDF Export",
}

STAGE_ORDER = ["segmenting", "extracting", "generating", "exporting"]

BLOCKED_KEYWORDS = [
    "treatment", "prognosis", "diagnose", "diagnosis", "should i",
    "survival", "chemotherapy", "radiation", "surgery", "grade",
    "malignant", "benign", "cancer",
]

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_defaults = {
    "page": "upload",
    "job_id": None,
    "chat_history": [],
    "report_text": None,
    "metrics": None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="BraTS MRI Report Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* page background */
    .stApp {
        background-color: #f4f6f9;
    }

    /* white panel cards */
    .card {
        background: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 16px;
    }

    /* accent colour */
    .accent {
        color: #1a5276;
    }
    .accent-border {
        border-left: 4px solid #1a5276;
        padding-left: 12px;
    }

    /* monospace report */
    .report-box {
        background: #f8f8f8;
        font-family: 'Courier New', Courier, monospace;
        max-height: 420px;
        overflow-y: auto;
        padding: 16px;
        border-radius: 6px;
        white-space: pre-wrap;
        font-size: 13px;
        line-height: 1.5;
    }

    /* deep-blue buttons */
    .stButton > button {
        background-color: #1a5276;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #154360;
        color: #ffffff;
    }
    .stDownloadButton > button {
        background-color: #1a5276;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #154360;
        color: #ffffff;
    }

    /* chat bubbles */
    .chat-user {
        background: #d6eaf8;
        border-radius: 12px 12px 0 12px;
        padding: 10px 14px;
        margin: 6px 0;
        text-align: right;
        max-width: 85%;
        margin-left: auto;
    }
    .chat-assistant {
        background: #ffffff;
        border-left: 4px solid #1a5276;
        border-radius: 0 12px 12px 12px;
        padding: 10px 14px;
        margin: 6px 0;
        max-width: 85%;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }

    /* stage checklist */
    .stage-done   { color: #1a5276; font-weight: 600; }
    .stage-active { color: #1a5276; font-weight: 700; }
    .stage-pending{ color: #aaaaaa; }

    /* hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Helper utilities
# =========================================================================

def _reset_session():
    """Reset everything back to the upload page."""
    st.session_state["page"] = "upload"
    st.session_state["job_id"] = None
    st.session_state["chat_history"] = []
    st.session_state["report_text"] = None
    st.session_state["metrics"] = None


def _stage_index(stage: str) -> int:
    """Return the numeric index of a pipeline stage (-1 if not found)."""
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return -1


# =========================================================================
# PAGE 1 — Upload
# =========================================================================

def page_upload():
    # Centre the content
    _, col, _ = st.columns([1, 2, 1])

    with col:
        st.markdown(
            '<h1 class="accent" style="text-align:center;margin-bottom:0;">'
            "🧠 BraTS MRI Report Assistant</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center;color:#555;font-size:15px;">'
            "AI-powered tumor segmentation · Feature extraction · "
            "Radiology report generation · Clinical Q&amp;A</p>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="card">', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload BraTS Case (.zip)",
            type=["zip"],
            accept_multiple_files=False,
        )

        st.markdown(
            '<p style="font-size:13px;color:#777;">'
            "Zip must contain: t1.nii.gz, t1ce.nii.gz, t2.nii.gz, flair.nii.gz"
            "</p>",
            unsafe_allow_html=True,
        )

        analyze_clicked = st.button(
            "Analyze Case",
            disabled=(uploaded is None),
            use_container_width=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        if analyze_clicked and uploaded is not None:
            with st.spinner("Uploading…"):
                try:
                    resp = requests.post(
                        f"{API_BASE}/api/analyze",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/zip")},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state["job_id"] = data["job_id"]
                    st.session_state["page"] = "processing"
                    st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Upload failed: {exc}")


# =========================================================================
# PAGE 2 — Processing
# =========================================================================

def page_processing():
    job_id = st.session_state["job_id"]

    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown(
            '<h2 class="accent" style="text-align:center;">'
            "⏳ Analyzing MRI Case…</h2>",
            unsafe_allow_html=True,
        )

        # ---- Fetch status ------------------------------------------------
        status_data = {
            "status": "running",
            "stage": "segmenting",
            "progress_pct": 0,
            "error_message": None,
        }
        try:
            resp = requests.get(f"{API_BASE}/api/status/{job_id}", timeout=10)
            resp.raise_for_status()
            status_data = resp.json()
        except requests.RequestException as exc:
            st.error(f"Could not reach backend: {exc}")

        current_stage = status_data.get("stage", "segmenting")
        progress_pct = status_data.get("progress_pct", 0)
        status = status_data.get("status", "running")

        # ---- Stage checklist ---------------------------------------------
        current_idx = _stage_index(current_stage)

        st.markdown('<div class="card">', unsafe_allow_html=True)

        for i, stage_key in enumerate(STAGE_ORDER):
            label = STAGE_LABELS[stage_key]
            if status == "done" or i < current_idx:
                st.markdown(
                    f'<p class="stage-done">✅ {label}</p>',
                    unsafe_allow_html=True,
                )
            elif i == current_idx and status == "running":
                st.markdown(
                    f'<p class="stage-active">⏳ {label}</p>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<p class="stage-pending">⬜ {label}</p>',
                    unsafe_allow_html=True,
                )

        st.progress(min(progress_pct, 100) / 100)

        st.markdown(
            '<p style="font-size:12px;font-style:italic;color:#888;">'
            "Segmentation runs on CPU and typically takes 5–6 minutes. "
            "Please keep this tab open.</p>",
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

        # ---- Handle terminal states --------------------------------------
        if status == "done":
            try:
                rpt = requests.get(
                    f"{API_BASE}/api/report/{job_id}", timeout=30
                )
                rpt.raise_for_status()
                st.session_state["report_text"] = rpt.text
            except requests.RequestException:
                st.session_state["report_text"] = "(Report could not be fetched.)"

            st.session_state["page"] = "dashboard"
            st.rerun()

        if status == "error":
            st.error(
                status_data.get("error_message") or "Pipeline encountered an error."
            )
            if st.button("Start Over"):
                _reset_session()
                st.rerun()
            return  # stop polling

        # ---- Poll again in 4 s -------------------------------------------
        time.sleep(4)
        st.rerun()


# =========================================================================
# PAGE 3 — Dashboard
# =========================================================================

def page_dashboard():
    job_id = st.session_state["job_id"]

    left, right = st.columns([6, 4])

    # =====================================================================
    # LEFT COLUMN
    # =====================================================================
    with left:
        # ----- Metrics card ----------------------------------------------
        if st.session_state["metrics"] is None:
            try:
                resp = requests.get(
                    f"{API_BASE}/api/metrics/{job_id}", timeout=15
                )
                resp.raise_for_status()
                st.session_state["metrics"] = resp.json()
            except requests.RequestException as exc:
                st.error(f"Could not load metrics: {exc}")
                st.session_state["metrics"] = {}

        m = st.session_state["metrics"] or {}

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h4 class="accent">Tumor Metrics</h4>', unsafe_allow_html=True)

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Whole Tumor", f"{m.get('whole_tumor_volume_cm3', 0):.1f} cm³")
        mc2.metric("Enhancing", f"{m.get('enhancing_volume_cm3', 0):.1f} cm³")
        mc3.metric("Edema", f"{m.get('edema_volume_cm3', 0):.1f} cm³")
        mc4.metric("Necrotic", f"{m.get('necrotic_volume_cm3', 0):.1f} cm³")

        mc5, mc6, mc7 = st.columns(3)
        hemisphere = m.get("hemisphere", "—")
        lobe = m.get("lobe", "—")
        mc5.metric("Location", f"{hemisphere} / {lobe}")
        mc6.metric("Midline Shift", f"{m.get('midline_shift_mm', 0):.1f} mm")
        mc7.metric("Quality", m.get("segmentation_quality", "—"))

        st.markdown("</div>", unsafe_allow_html=True)

        # ----- Report card -----------------------------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h4 class="accent">Radiology Report</h4>', unsafe_allow_html=True
        )

        report_text = st.session_state.get("report_text") or ""
        st.markdown(
            f'<div class="report-box">{report_text}</div>',
            unsafe_allow_html=True,
        )

        # PDF download
        try:
            pdf_resp = requests.get(
                f"{API_BASE}/api/report/{job_id}/pdf", timeout=30
            )
            if pdf_resp.status_code == 200:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_resp.content,
                    file_name="radiology_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.info("PDF report not available yet.")
        except requests.RequestException:
            st.info("PDF report not available yet.")

        st.markdown("</div>", unsafe_allow_html=True)

        # ----- Clinical Q&A card -----------------------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h4 class="accent">Ask About This Report</h4>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="font-size:13px;color:#c0392b;">'
            "⚠️ For educational reference only. "
            "Not a substitute for radiologist review.</p>",
            unsafe_allow_html=True,
        )

        # Chat history
        for msg in st.session_state["chat_history"]:
            st.markdown(
                f'<div class="chat-user">{msg["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-assistant">{msg["answer"]}</div>',
                unsafe_allow_html=True,
            )

        question = st.text_input(
            "Type your question here…", key="chat_input", label_visibility="collapsed",
            placeholder="Type your question here…",
        )
        send_clicked = st.button("Send", key="send_btn")

        if send_clicked and question:
            # Client-side keyword gate
            q_lower = question.lower()
            blocked = any(kw in q_lower for kw in BLOCKED_KEYWORDS)

            if blocked:
                st.warning(
                    "This question requires clinical judgment. "
                    "Please consult a qualified radiologist."
                )
            else:
                try:
                    chat_resp = requests.post(
                        f"{API_BASE}/api/chat/{job_id}",
                        json={"question": question},
                        timeout=60,
                    )
                    if chat_resp.status_code == 400:
                        detail = chat_resp.json().get("detail", "Request rejected.")
                        st.warning(detail)
                    else:
                        chat_resp.raise_for_status()
                        answer = chat_resp.json().get("answer", "")
                        st.session_state["chat_history"].append(
                            {"question": question, "answer": answer}
                        )
                        st.rerun()
                except requests.RequestException as exc:
                    st.error(f"Chat request failed: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================================
    # RIGHT COLUMN
    # =====================================================================
    with right:
        # ----- MRI Viewer placeholder ------------------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h4 class="accent">MRI Viewer</h4>', unsafe_allow_html=True
        )
        st.markdown(
            '<div style="background:#eef2f7;border-radius:6px;padding:40px 20px;'
            'text-align:center;color:#888;margin-bottom:12px;">'
            "MRI slice visualization will be available in the next version."
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:13px;'>"
            "🔴 Necrotic Core &nbsp;&nbsp; 🟢 Edema &nbsp;&nbsp; 🔵 Enhancing Tumor"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='font-size:12px;color:#888;font-style:italic;'>"
            "Segmentation by nnUNet BraTS 2021 winning model (KAIST MRI Lab)"
            "</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ----- Case Info card --------------------------------------------
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<h4 class="accent">Case Info</h4>', unsafe_allow_html=True
        )
        short_id = (job_id or "")[:8]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.markdown(
            f"<p><strong>Job ID:</strong> {short_id}</p>"
            f"<p><strong>Analyzed:</strong> {ts}</p>"
            "<p style='font-size:12px;color:#888;font-style:italic;'>"
            "All outputs require review by a qualified radiologist "
            "before clinical use.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ----- New Analysis button ----------------------------------------
        if st.button("New Analysis"):
            _reset_session()
            st.rerun()


# =========================================================================
# Router
# =========================================================================

page = st.session_state["page"]

if page == "upload":
    page_upload()
elif page == "processing":
    page_processing()
elif page == "dashboard":
    page_dashboard()
else:
    _reset_session()
    st.rerun()
