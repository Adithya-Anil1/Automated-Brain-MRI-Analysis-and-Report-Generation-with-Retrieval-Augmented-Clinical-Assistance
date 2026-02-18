"""
Streamlit frontend for the AI-Powered Brain MRI Assistant.
Provides MRI upload, structured radiology report display,
RAG-based clinical Q&A, and a visualization workspace placeholder.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Brain MRI Assistant",
    page_icon="🧠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
ENABLE_VISUALIZATION = False

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "report_text" not in st.session_state:
    st.session_state.report_text = ""
if "rag_answer" not in st.session_state:
    st.session_state.rag_answer = ""


# =========================================================================
# RAG helper (placeholder — replace with real backend call later)
# =========================================================================

def query_rag(question: str) -> str:
    """Send *question* to the RAG backend and return the answer.

    Currently returns an empty string.  Replace the body with a call to
    /api/chat when the backend is ready.
    """
    return ""


# =========================================================================
# LEFT PANEL — Upload
# =========================================================================

def render_upload_section():
    """Four modality uploaders + Run Analysis button."""

    st.header("Upload MRI Modalities")

    t1_file = st.file_uploader("T1 (.nii.gz)", type=["nii.gz"], key="up_t1")
    t1ce_file = st.file_uploader("T1ce (.nii.gz)", type=["nii.gz"], key="up_t1ce")
    t2_file = st.file_uploader("T2 (.nii.gz)", type=["nii.gz"], key="up_t2")
    flair_file = st.file_uploader("FLAIR (.nii.gz)", type=["nii.gz"], key="up_flair")

    all_uploaded = all([t1_file, t1ce_file, t2_file, flair_file])

    if st.button("Run Analysis", disabled=not all_uploaded, use_container_width=True):
        with st.spinner("Running analysis..."):
            # Placeholder — swap with real /api/analyze call later
            st.session_state.report_text = (
                "STRUCTURED RADIOLOGY REPORT\n"
                "===========================\n\n"
                "Patient ID:          BraTS-GLI-XXXXX-000\n"
                "Study Date:          2026-02-19\n"
                "Modalities:          T1, T1ce, T2, FLAIR\n\n"
                "--- FINDINGS ---\n\n"
                "1. Tumor Location\n"
                "   Hemisphere:       Left\n"
                "   Lobe:             Temporal\n\n"
                "2. Tumor Volumes\n"
                "   Whole Tumor:      45.2 cm³\n"
                "   Enhancing:        12.8 cm³\n"
                "   Edema:            25.1 cm³\n"
                "   Necrotic Core:     7.3 cm³\n\n"
                "3. Mass Effect\n"
                "   Midline Shift:     3.2 mm\n"
                "   Ventricle Compression: Moderate\n\n"
                "4. Morphology\n"
                "   Shape:            Irregular\n"
                "   Margins:          Ill-defined\n"
                "   Enhancement:      Heterogeneous ring-enhancing\n\n"
                "5. Signal Characteristics\n"
                "   T1:               Hypointense\n"
                "   T1ce:             Peripheral enhancement\n"
                "   T2:               Hyperintense\n"
                "   FLAIR:            Hyperintense with peritumoral edema\n\n"
                "6. Additional Observations\n"
                "   Segmentation Quality: Good\n"
                "   Multiplicity:     Solitary lesion\n\n"
                "--- IMPRESSION ---\n\n"
                "Large heterogeneously enhancing intra-axial mass in the\n"
                "left temporal lobe with surrounding vasogenic edema and\n"
                "central necrosis. Findings are concerning for high-grade\n"
                "glioma. Clinical correlation and histopathological\n"
                "confirmation recommended.\n\n"
                "--- DISCLAIMER ---\n"
                "This report is AI-generated and requires review by a\n"
                "qualified radiologist before clinical use."
            )
            # Clear any previous RAG answer when a new analysis runs
            st.session_state.rag_answer = ""


# =========================================================================
# LEFT PANEL — Report
# =========================================================================

def render_report_section():
    """Display the structured radiology report with a download button."""

    st.header("Structured Radiology Report")

    if not st.session_state.report_text:
        st.info("No report generated yet.")
        return

    st.text_area(
        "Report",
        value=st.session_state.report_text,
        height=500,
        disabled=True,
        label_visibility="collapsed",
    )

    st.download_button(
        label="Download Report",
        data=st.session_state.report_text.encode("utf-8"),
        file_name="report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# =========================================================================
# LEFT PANEL — RAG Q&A
# =========================================================================

def render_rag_section():
    """Clinical question input and answer display."""

    st.header("Ask About This Case")

    question = st.text_input(
        "Enter your clinical question",
        placeholder="e.g. What regions show enhancement?",
        key="rag_input",
    )

    if st.button("Ask", use_container_width=True):
        if question:
            answer = query_rag(question)
            st.session_state.rag_answer = answer

    if st.session_state.rag_answer:
        st.markdown(st.session_state.rag_answer)


# =========================================================================
# RIGHT PANEL — Visualization workspace
# =========================================================================

def render_visualization_panel():
    """MRI slice viewer placeholder with controls."""

    st.header("MRI Visualization Workspace")

    # ----- Slice display area (only permitted use of unsafe_allow_html) ---
    st.markdown(
        "<div style='border:1px solid #ccc; padding:20px; "
        "border-radius:8px; text-align:center;'>"
        "MRI slice rendering will appear here.</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ----- Slice navigation -----------------------------------------------
    st.slider("Slice Index", 0, 155, 0, disabled=True, key="slice_idx")

    st.divider()

    # ----- Tumor region overlay controls ----------------------------------
    st.subheader("Tumor Region Overlays")
    st.checkbox("Enhancing Tumor (ET)", value=False, disabled=True, key="ov_et")
    st.checkbox("Edema (ED)", value=False, disabled=True, key="ov_ed")
    st.checkbox("Necrotic Core (NCR)", value=False, disabled=True, key="ov_ncr")

    st.divider()

    # ----- Overlay transparency -------------------------------------------
    st.slider("Overlay Transparency", 0, 100, 50, disabled=True, key="ov_alpha")


# =========================================================================
# Main layout
# =========================================================================

col_left, col_right = st.columns([9, 11])

with col_left:
    render_upload_section()
    st.divider()
    render_report_section()
    st.divider()
    render_rag_section()

with col_right:
    render_visualization_panel()
