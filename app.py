"""
MedAssist - Clinical Decision Support Agent
Streamlit UI
"""

import streamlit as st
import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

# ── Fix import paths ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
load_dotenv()

from agents.clinical_agent import run_clinical_agent
from utils.risk_scorer import RiskScorer

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="MedAssist — Clinical Decision Support",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS — Clean Readable Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #1a1a2e !important;
    }

    .stApp {
        background-color: #f0f4f8 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #1a1a2e !important;
        border-right: 2px solid #2d2d5e;
    }

    [data-testid="stSidebar"] * {
        color: #e0e6f0 !important;
    }

    /* ── Header ── */
    .header-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        padding: 1.8rem 2rem;
        border-radius: 14px;
        border-left: 6px solid #e94560;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }

    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 0;
    }

    .header-sub {
        color: #a8b8d8 !important;
        font-size: 0.88rem;
        margin-top: 0.4rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Section titles ── */
    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1a1a2e !important;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #e94560;
    }

    /* ── Cards ── */
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.3rem;
    }

    /* ── Pipeline node cards in sidebar ── */
    .node-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
    }

    .node-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #e94560 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .node-desc {
        font-size: 0.78rem;
        color: #a8b8d8 !important;
        margin-top: 0.2rem;
    }

    /* ── Disclaimer ── */
    .disclaimer {
        background: #fff8e6;
        border: 1px solid #f6ad55;
        border-left: 4px solid #ed8936;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-top: 1rem;
        font-size: 0.82rem;
        color: #744210 !important;
        line-height: 1.5;
    }

    /* ── Report output area ── */
    .report-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        color: #1a1a2e !important;
        font-size: 0.92rem;
        line-height: 1.8;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff;
        border-radius: 10px;
        padding: 0.3rem;
        gap: 0.3rem;
        border: 1px solid #e2e8f0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #64748b !important;
        font-weight: 500;
        font-size: 0.88rem;
        padding: 0.4rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: #1a1a2e !important;
        color: #ffffff !important;
    }

    /* ── Text area ── */
    div[data-testid="stTextArea"] textarea {
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #1a1a2e !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
        padding: 0.75rem !important;
    }

    div[data-testid="stTextArea"] textarea:focus {
        border-color: #e94560 !important;
        box-shadow: 0 0 0 3px rgba(233,69,96,0.1) !important;
    }

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] > div {
        background: #ffffff !important;
        border: 2px solid #cbd5e1 !important;
        border-radius: 10px !important;
        color: #1a1a2e !important;
    }

    /* ── Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #c62a47) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #ff5c75, #e94560) !important;
        box-shadow: 0 6px 20px rgba(233,69,96,0.45) !important;
        transform: translateY(-1px) !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #f8fafc !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
        font-weight: 500 !important;
        border: 1px solid #e2e8f0 !important;
    }

    .streamlit-expanderContent {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-top: none !important;
        color: #374151 !important;
        font-size: 0.88rem !important;
        line-height: 1.7 !important;
    }

    /* ── All text in main area readable ── */
    .main p, .main li, .main span,
    .main div, .main label {
        color: #1a1a2e !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background: #e94560 !important;
    }

    /* ── Alerts ── */
    .stSuccess {
        background: #f0fdf4 !important;
        border: 1px solid #86efac !important;
        color: #166534 !important;
        border-radius: 8px !important;
    }

    .stError {
        background: #fef2f2 !important;
        border: 1px solid #fca5a5 !important;
        color: #991b1b !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: #fffbeb !important;
        border: 1px solid #fcd34d !important;
        color: #92400e !important;
        border-radius: 8px !important;
    }

    .stInfo {
        background: #eff6ff !important;
        border: 1px solid #93c5fd !important;
        color: #1e40af !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────

st.markdown("""
<div class="header-box">
    <div class="header-title">🏥 MedAssist</div>
    <div class="header-sub">
        Clinical Decision Support Agent &nbsp;·&nbsp;
        LangGraph + RAG + Groq Llama3 &nbsp;·&nbsp;
        GTX 1650 GPU Embeddings
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 Agent Pipeline")
    st.markdown("---")

    nodes = [
        ("🔍", "Input Parser",      "Extracts symptoms, history, medications"),
        ("📚", "RAG Retrieval",     "FAISS search — Kaggle + clinical docs"),
        ("🧠", "Symptom Analyzer",  "Evidence-based pattern recognition"),
        ("⚕️", "Diagnosis Support", "Differential diagnosis generation"),
        ("⚠️", "Risk Flagging",     "Red flags, drug interactions, urgency"),
        ("📋", "Report Generator",  "Structured clinical decision report"),
    ]

    for icon, name, desc in nodes:
        st.markdown(f"""
        <div class="node-card">
            <div class="node-title">{icon} {name}</div>
            <div class="node-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## ⚙️ Stack")
    st.markdown("""
    <div style='font-size:0.82rem; line-height:2.0;'>
    🔗 <b style='color:#e94560'>LangGraph</b> — State machine<br>
    🤖 <b style='color:#e94560'>Groq Llama3</b> — FREE LLM<br>
    📚 <b style='color:#e94560'>FAISS</b> — Vector search<br>
    🎯 <b style='color:#e94560'>HuggingFace</b> — GPU embeddings<br>
    📊 <b style='color:#e94560'>Kaggle</b> — 41 diseases dataset<br>
    ⚡ <b style='color:#e94560'>GTX 1650</b> — CUDA 12.8
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # GPU Status
    try:
        import torch
        if torch.cuda.is_available():
            gpu  = torch.cuda.get_device_name(0)
            vram = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            st.success(f"✅ GPU: {gpu} ({vram}GB)")
        else:
            st.warning("⚠️ CPU Mode — GPU not detected")
    except Exception:
        pass


# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────

col1, col2 = st.columns([1, 1], gap="large")

# ── Left — Input ──
with col1:
    st.markdown('<div class="section-title">📝 Patient Report</div>', unsafe_allow_html=True)

    sample_cases = {
        "Select a sample case...": "",
        "🫀 Cardiac — Chest Pain (High Risk)":
            "65-year-old male, smoker, hypertensive. Presenting with crushing chest pain "
            "radiating to left arm for 30 minutes, associated with sweating and nausea. "
            "Current medications: Lisinopril 10mg, Aspirin 81mg. History of hypercholesterolemia.",
        "🫁 Respiratory — Shortness of Breath":
            "52-year-old female with 3-day history of progressive shortness of breath, "
            "productive cough with yellow sputum, fever 38.5C. Former smoker (10 pack-years). "
            "No chest pain. Oxygen saturation 94% on room air.",
        "🧠 Neurological — Sudden Headache":
            "38-year-old male with sudden onset severe headache described as worst headache "
            "of my life, started 2 hours ago. Neck stiffness and photophobia. "
            "No prior headache history. BP 160/95 on presentation.",
        "🩸 Endocrine — Diabetes Follow-up":
            "58-year-old female with Type 2 diabetes for 12 years. HbA1c 9.2%. "
            "Medications: Metformin 1000mg BD, Glipizide 5mg. Increased thirst, "
            "frequent urination, blurred vision for 2 weeks. CT scan with contrast planned.",
    }

    selected     = st.selectbox("Quick load sample case:", list(sample_cases.keys()))
    default_text = sample_cases[selected]

    patient_input = st.text_area(
        "Patient symptoms, history and current medications:",
        value=default_text,
        height=220,
        placeholder=(
            "e.g. 55-year-old male with chest pain radiating to left arm, "
            "shortness of breath, history of hypertension. "
            "Currently on Metformin and Lisinopril..."
        )
    )

    analyze_btn = st.button("🔬 Run Clinical Analysis", use_container_width=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Medical Disclaimer:</b> MedAssist is a clinical <i>decision support</i> tool only.
        All AI-generated outputs must be reviewed by a licensed healthcare professional.
        Never use as a substitute for qualified medical judgment.
    </div>
    """, unsafe_allow_html=True)


# ── Right — Output ──
with col2:
    st.markdown('<div class="section-title">📊 Clinical Analysis</div>', unsafe_allow_html=True)

    if analyze_btn and patient_input.strip():

        scorer       = RiskScorer()
        progress_bar = st.progress(0)
        status_text  = st.empty()

        steps = [
            "🔍 Parsing patient input...",
            "📚 Retrieving clinical guidelines...",
            "🧠 Analyzing symptoms...",
            "⚕️ Generating differential diagnoses...",
            "⚠️  Assessing risk factors...",
            "📋 Compiling clinical report...",
        ]

        for i, step in enumerate(steps):
            status_text.info(step)
            progress_bar.progress((i + 1) / len(steps))
            time.sleep(0.25)

        try:
            with st.spinner("Running LangGraph pipeline..."):
                result = run_clinical_agent(patient_input)

            progress_bar.empty()
            status_text.empty()

            # ── Metrics Row ──
            risk_score = result.get("risk_score", 0.0)
            confidence = result.get("confidence", 0.0)
            num_docs   = len(result.get("retrieved_docs", []))

            risk_color = (
                "#dc2626" if risk_score > 0.7 else
                "#d97706" if risk_score > 0.4 else
                "#16a34a"
            )

            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:{risk_color};">{risk_score:.0%}</div>
                    <div class="metric-label">Risk Score</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#1a1a2e;">{confidence:.0%}</div>
                    <div class="metric-label">Confidence</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#1a1a2e;">{num_docs}</div>
                    <div class="metric-label">Sources Used</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Risk Badge ──
            risk_label = scorer.get_risk_label(risk_score)
            badge_bg   = (
                "#fef2f2" if risk_score > 0.7 else
                "#fffbeb" if risk_score > 0.4 else
                "#f0fdf4"
            )
            badge_border = (
                "#fca5a5" if risk_score > 0.7 else
                "#fcd34d" if risk_score > 0.4 else
                "#86efac"
            )
            st.markdown(
                f'<div style="background:{badge_bg}; color:{risk_color}; '
                f'border:2px solid {badge_border}; padding:0.5rem 1.2rem; '
                f'border-radius:25px; display:inline-block; '
                f'font-weight:700; font-size:0.95rem; margin-bottom:1rem;">'
                f'{risk_label}</div>',
                unsafe_allow_html=True
            )

            # ── Output Tabs ──
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Report", "🤖 ML Classifier", "🔬 Symptoms", "⚠️ Risk Flags", "📚 Sources"
            ])

            with tab1:
                final = result.get("final_response", "No report generated.")
                st.markdown(
                    f'<div class="report-box">{final}</div>',
                    unsafe_allow_html=True
                )

            with tab2:
                clf_result = result.get("classifier_result", {})
                if clf_result and clf_result.get("classifier_available"):
                    preds    = clf_result.get("top_predictions", [])
                    acc      = clf_result.get("model_accuracy", 0)
                    detected = clf_result.get("detected_symptoms", [])
                    shap_exp = clf_result.get("shap_explanation", "")

                    st.markdown(f"""
                    <div style='background:#f0fdf4; border:1px solid #86efac;
                                border-left:4px solid #16a34a; border-radius:8px;
                                padding:0.8rem 1rem; margin-bottom:1rem; font-size:0.88rem; color:#166534;'>
                        🤖 <b>XGBoost Classifier</b> — Trained on Kaggle Dataset &nbsp;|&nbsp;
                        Model Accuracy: <b>{acc:.1%}</b> &nbsp;|&nbsp;
                        Symptoms Detected: <b>{len(detected)}</b>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**🏆 Top Disease Predictions:**")
                    for p in preds:
                        rank  = p["rank"]
                        dis   = p["disease"]
                        conf  = p["confidence"]
                        color = "#dc2626" if rank == 1 else "#d97706" if rank == 2 else "#16a34a"
                        bar_w = int(conf * 100)
                        st.markdown(f"""
                        <div style='background:#ffffff; border:1px solid #e2e8f0;
                                    border-radius:10px; padding:0.9rem 1.2rem;
                                    margin-bottom:0.6rem; box-shadow:0 1px 4px rgba(0,0,0,0.05);'>
                            <div style='display:flex; justify-content:space-between; align-items:center;'>
                                <span style='font-weight:700; color:#1a1a2e; font-size:0.95rem;'>
                                    #{rank} &nbsp; {dis}
                                </span>
                                <span style='font-weight:700; color:{color}; font-size:1rem;
                                             font-family:JetBrains Mono, monospace;'>
                                    {conf:.1%}
                                </span>
                            </div>
                            <div style='background:#f1f5f9; border-radius:4px;
                                        height:6px; margin-top:0.5rem; overflow:hidden;'>
                                <div style='background:{color}; width:{bar_w}%;
                                            height:100%; border-radius:4px;'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")
                    st.markdown("**🔍 XAI Explanation (SHAP):**")
                    st.markdown(
                        f'<div class="report-box" style="font-family:JetBrains Mono,monospace; '
                        f'font-size:0.85rem; white-space:pre-wrap;">{shap_exp}</div>',
                        unsafe_allow_html=True
                    )

                    if detected:
                        st.markdown("---")
                        st.markdown("**💊 Detected Symptoms:**")
                        chips = " ".join([
                            f'<span style="background:#eff6ff; color:#1e40af; '
                            f'border:1px solid #93c5fd; border-radius:12px; '
                            f'padding:0.2rem 0.7rem; font-size:0.78rem; '
                            f'display:inline-block; margin:0.2rem;">'
                            f'{s.replace("_"," ").title()}</span>'
                            for s in detected
                        ])
                        st.markdown(chips, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ ML Classifier not available.")
                    st.info("Run: **python train_classifier.py** to train the model first.")

            with tab3:
                symptoms = result.get("symptom_analysis", "No symptom analysis available.")
                st.markdown(
                    f'<div class="report-box">{symptoms}</div>',
                    unsafe_allow_html=True
                )
                st.markdown("---")
                st.markdown("**🔬 Diagnosis Support:**")
                diagnosis = result.get("diagnosis_support", "Not available.")
                st.markdown(
                    f'<div class="report-box">{diagnosis}</div>',
                    unsafe_allow_html=True
                )

            with tab4:
                risk_flags = result.get("risk_flags", [])
                if risk_flags:
                    for flag in risk_flags:
                        st.error(f"🚩 {flag}")
                else:
                    st.success("✅ No critical red flags identified.")

                st.markdown("---")
                st.markdown("**XAI Risk Score Breakdown:**")
                st.code(scorer.get_score_explanation({}), language=None)

            with tab5:
                retrieved = result.get("retrieved_docs", [])
                if retrieved:
                    st.markdown(f"**{len(retrieved)} clinical sources retrieved:**")
                    for i, doc in enumerate(retrieved, 1):
                        src     = doc.get("source", "Unknown")
                        cat     = doc.get("category", "general")
                        rel     = doc.get("relevance_score", 0)
                        disease = doc.get("disease", "")
                        label   = f"{src}" + (f" — {disease}" if disease else "")
                        with st.expander(f"📄 [{i}] {label}  (relevance: {rel:.2f})"):
                            st.markdown(f"`Category: {cat}`")
                            content = doc.get("content", "")[:600]
                            st.markdown(
                                f'<div style="color:#374151; font-size:0.88rem; '
                                f'line-height:1.7; background:#f8fafc; '
                                f'border-radius:8px; padding:0.75rem;">'
                                f'{content}...</div>',
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No sources retrieved.")

        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"⚠️ Error: {str(e)}")
            err = str(e).lower()
            if "groq_api_key" in err or "api key" in err:
                st.info("💡 Check your GROQ_API_KEY in the .env file")
            elif "decommission" in err:
                st.info("💡 Update model to: llama-3.3-70b-versatile in clinical_agent.py")
            elif "faiss" in err or "index" in err:
                st.info("💡 Run: python data\\ingest.py to rebuild the FAISS index")

    elif analyze_btn:
        st.warning("⚠️ Please enter a patient report to analyze.")

    else:
        st.markdown("""
        <div style="text-align:center; padding:3.5rem 2rem;
                    background:#ffffff; border-radius:14px;
                    border:2px dashed #cbd5e1;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);">
            <div style="font-size:3.5rem; margin-bottom:1rem;">🏥</div>
            <div style="font-size:1.15rem; font-weight:700;
                        color:#1a1a2e; margin-bottom:0.5rem;">
                Ready for Analysis
            </div>
            <div style="font-size:0.9rem; color:#64748b; line-height:2.0;">
                1️⃣ &nbsp;Select a sample case or type a patient report<br>
                2️⃣ &nbsp;Click <b style='color:#e94560'>Run Clinical Analysis</b><br>
                3️⃣ &nbsp;Watch the 6-node LangGraph pipeline run<br>
                4️⃣ &nbsp;Review Report · Symptoms · Risks · Sources
            </div>
        </div>
        """, unsafe_allow_html=True)
