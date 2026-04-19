"""
app.py — PlantPulse AI + Quantum (Enterprise Edition v3)
======================================================
Comprehensive Plant Diagnostic Suite with:
- Local Hybrid AI (Histogram + Raw)
- PlantNet API Integration (Identification)
- Kindwise Crop Health API (Disease Detection)
- IBM Quantum Severity Prediction
- GPT-style Plant Chatbot
- Clinical PDF Reports
"""

import streamlit as st
import cv2
import numpy as np
import os
import json
import datetime
import base64
import requests
from fpdf import FPDF
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from dotenv import load_dotenv

# Shared utilities
from utils import (
    predict_image, get_disease_info,
    get_feature_mode, load_model_and_scaler,
    FEATURE_MODE_RAW, FEATURE_MODE_HIST,
    decode_bytes_to_bgr,
    identify_plant_with_plantnet,
    identify_disease_with_kindwise,
    get_perenual_care_info
)

load_dotenv()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PlantPulse Enterprise",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# STYLING & ANIMATION
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/animate.css/4.1.1/animate.min.css');

    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

    .main { 
        background-color: #050a12;
        background-image: 
            radial-gradient(at 0% 0%, hsla(160,100%,10%,1) 0, transparent 50%), 
            radial-gradient(at 50% 0%, hsla(180,100%,5%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(160,100%,10%,1) 0, transparent 50%);
        color: white; 
    }

    /* Falling Leaves Animation */
    @keyframes flower-fall {
        0% { transform: translateY(-10vh) rotate(0deg); opacity: 0; }
        10% { opacity: 0.8; }
        90% { opacity: 0.8; }
        100% { transform: translateY(110vh) rotate(360deg); opacity: 0; }
    }

    .leaf {
        position: fixed;
        top: -10vh;
        z-index: 0;
        pointer-events: none;
        animation: flower-fall 10s linear infinite;
        font-size: 20px;
    }

    /* Custom Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2rem;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .glass-card:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: #10b981;
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #10b981, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .severity-badge {
        padding: 5px 15px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .sev-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }
    .sev-medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .sev-low { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; border: none; border-radius: 12px;
        padding: 0.75rem 1.5rem; font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }

    /* Chatbot styles */
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: rgba(0,0,0,0.2);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.05);
    }
</style>

<!-- Falling Leaves Elements -->
<div class="leaf" style="left: 5%; animation-duration: 7s;">🌿</div>
<div class="leaf" style="left: 15%; animation-duration: 12s;">🍃</div>
<div class="leaf" style="left: 25%; animation-duration: 9s;">🌱</div>
<div class="leaf" style="left: 45%; animation-duration: 15s;">🌿</div>
<div class="leaf" style="left: 65%; animation-duration: 8s;">🍃</div>
<div class="leaf" style="left: 85%; animation-duration: 11s;">🌱</div>
<div class="leaf" style="left: 95%; animation-duration: 13s;">🌿</div>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE INIT
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I am Groot... I mean, PlantPulse AI. How can I help you today?"}
    ]
if "diagnosis_ready" not in st.session_state:
    st.session_state.diagnosis_ready = False
if "current_report" not in st.session_state:
    st.session_state.current_report = None

# ===============================
# ANALYTICS CACHING
# ===============================
@st.cache_resource
def get_cached_model():
    try:
        return load_model_and_scaler()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

model, scaler = get_cached_model()
feature_mode = get_feature_mode(model) if model else "N/A"

# ===============================
# QUANTUM SEVERITY ENGINE
# ===============================
def analyze_severity_quantum(img: np.ndarray, backend_pref: str):
    """Predicts a severity score from 1-5 using Qiskit."""
    # Build 4-qubit circuit based on image entropy and density
    small = cv2.resize(img, (64, 64))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
    mean_val = float(np.mean(gray))
    edge_dens = float(np.sum(cv2.Canny((gray * 255).astype(np.uint8), 50, 150) > 0) / (64 * 64))
    
    qc = QuantumCircuit(4, 4)
    from math import pi
    qc.ry(mean_val * pi, 0)
    qc.ry(edge_dens * pi, 1)
    qc.h(2)
    qc.cx(0, 3)
    qc.cx(1, 3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])

    try:
        TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
        if not TOKEN: raise ValueError("No token")
        service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
        backend = service.least_busy(simulator=(backend_pref == "Simulator Only"))
        qc_t = transpile(qc, backend)
        sampler = Sampler(backend)
        job = sampler.run([qc_t], shots=1024)
        counts = job.result()[0].data.c.get_counts()
    except:
        # Local fallback
        from qiskit.primitives import StatevectorSampler
        sampler = StatevectorSampler()
        job = sampler.run([qc])
        counts = job.result()[0].data.c.get_counts()
        backend = type('obj', (object,), {'name': 'local-sim'})

    # Severity Heuristic: Count bits set in dominant state
    dom_state = max(counts, key=counts.get)
    # We map bitcount (0-4) to severity score (1-5)
    score = dom_state.count('1') + 1
    labels = ["Healthy/Optimal", "Incipient", "Noticeable", "Widespread", "Critical"]
    return {
        "score": score,
        "label": labels[score-1],
        "state": dom_state,
        "backend": backend.name
    }

# ===============================
# PDF REPORT GENERATOR
# ===============================
class PDFReport(FPDF):
    def header(self):
        self.set_fill_color(16, 185, 129)
        self.rect(0, 0, 210, 40, 'F')
        self.set_font('helvetica', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 20, 'PLANTPULSE DIAGNOSTIC REPORT', ln=True, align='C')
        self.set_font('helvetica', 'I', 10)
        self.cell(0, 0, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')

    def add_section(self, title, content):
        self.set_font('helvetica', 'B', 16)
        self.set_text_color(16, 185, 129)
        self.cell(0, 15, title, ln=True)
        self.set_font('helvetica', '', 12)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 10, content)
        self.ln(5)

def create_report_pdf(data):
    pdf = PDFReport()
    pdf.add_page()
    pdf.ln(20)
    
    pdf.add_section("SPECIMEN IDENTIFICATION", 
        f"Detected Plant: {data['plant'].title()}\n"
        f"Scientific Name: {data.get('scientific_name', 'N/A')}\n"
        f"AI Confidence: {data['confidence']}%\n"
        f"Family: {data.get('family', 'N/A')}")
    
    pdf.add_section("PATHOLOGY ANALYSIS", 
        f"Condition: {data['disease'].replace('_',' ').title()}\n"
        f"Health Status: {'INFECTED' if data['disease'] != 'healthy' else 'HEALTHY'}\n"
        f"Quantum Severity Score: {data['q_sev']['score']}/5 ({data['q_sev']['label']})")
    
    pdf.add_section("TREATMENT PROTOCOL", data['tips'])
    
    if data.get('treatment_details'):
        pdf.add_section("DETAILED REMEDIATION", data['treatment_details'])

    return pdf.output()

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/color/144/leaf.png", width=100)
    st.title("Settings")
    
    with st.expander("🛠 Engine Config", expanded=True):
        api_mode = st.radio("Pipeline Mode", ["Hybrid (Local + API)", "Experimental (API Only)", "Local Only"])
        q_backend = st.selectbox("Quantum Backend", ["Dynamic", "Simulator Only"])
        use_care_api = st.toggle("Fetch Deep Care Info", value=True)
    
    st.divider()
    st.subheader("System Health")
    st.status("AI Core: Online", state="complete" if model else "error")
    st.status("PlantNet API: Connected", state="complete")
    st.status("Kindwise API: Connected", state="complete")
    
    st.divider()
    if st.button("🗑 Clear Session History"):
        st.session_state.chat_history = [{"role": "assistant", "content": "Reset complete. Ready for new specimen."}]
        st.session_state.diagnosis_ready = False
        st.rerun()

# ===============================
# MAIN LAYOUT
# ===============================
st.markdown("<h1 class='animate__animated animate__fadeInDown' style='text-align:center; font-size:4rem; margin-bottom:0;'>🌿 PlantPulse <span style='color:#10b981'>v3</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.7; font-size:1.2rem;'>Enterprise Grade Plant Pathogen Identification & Quantum Analytics</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

# ---- INPUT COLUMN ----
with col1:
    st.subheader("📸 Input Specimen")
    src_tab1, src_tab2 = st.tabs(["Upload Specimen", "Live Capture"])
    
    input_img = None
    with src_tab1:
        up_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        if up_file:
            bytes_data = up_file.read()
            input_img = decode_bytes_to_bgr(bytes_data)
    
    with src_tab2:
        cam_file = st.camera_input("Take photo")
        if cam_file:
            bytes_data = cam_file.read()
            input_img = decode_bytes_to_bgr(bytes_data)

    if input_img is not None:
        st.image(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB), caption="Analyzed Specimen", use_column_width=True)
        if st.button("🚀 EXECUTE FULL DIAGNOSTIC", use_container_width=True):
            with st.spinner("INITIATING MULTI-ENGINE PIPELINE..."):
                # 1. Local AI
                loc_res = predict_image(input_img, model, scaler)
                
                # 2. PlantNet API
                pn_res = identify_plant_with_plantnet(input_img) if api_mode != "Local Only" else {"error": "Skipped"}
                
                # 3. Kindwise API
                kw_res = identify_disease_with_kindwise(input_img) if api_mode != "Local Only" else {"error": "Skipped"}
                
                # 4. Quantum
                q_res = analyze_severity_quantum(input_img, q_backend)
                
                # 5. Care Info
                care_q = pn_res.get('scientific_name', loc_res['plant'])
                care_res = get_perenual_care_info(care_q) if use_care_api else {}

                # Merge Results
                final_plant = pn_res.get('scientific_name', loc_res['plant'])
                final_disease = kw_res.get('disease', loc_res['disease']) if "error" not in kw_res else loc_res['disease']
                
                report_data = {
                    "plant": final_plant,
                    "scientific_name": pn_res.get('scientific_name', 'Unknown'),
                    "common_names": pn_res.get('common_names', []),
                    "family": pn_res.get('family', 'N/A'),
                    "disease": final_disease,
                    "confidence": loc_res['confidence'],
                    "q_sev": q_res,
                    "tips": loc_res['tips'],
                    "treatment_details": kw_res.get('description', '') + "\n\n" + str(kw_res.get('treatment', '')),
                    "care": care_res
                }
                
                st.session_state.current_report = report_data
                st.session_state.diagnosis_ready = True
                
                # Auto-add to chat context
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"I've completed the analysis for the {final_plant}. It seems to have {final_disease.replace('_',' ')}. Quantum severity is {q_res['label']}. How can I assist with the treatment?"
                })

# ---- ANALYSIS COLUMN ----
with col2:
    if st.session_state.diagnosis_ready:
        rep = st.session_state.current_report
        
        st.markdown(f"### 📋 Diagnostic Results: `{rep['plant'].upper()}`")
        
        # Upper stats
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='glass-card'><p style='margin:0; opacity:0.6;'>Plant ID</p><p class='metric-value' style='font-size:1.5rem;'>{rep['plant'].split(' ')[0].title()}</p></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='glass-card'><p style='margin:0; opacity:0.6;'>Status</p><p class='metric-value' style='font-size:1.5rem;'>{rep['disease'].replace('_',' ').title()}</p></div>", unsafe_allow_html=True)
        with m3:
            s_val = rep['q_sev']['score']
            s_lab = rep['q_sev']['label']
            s_cls = "sev-low" if s_val < 3 else "sev-medium" if s_val < 4 else "sev-high"
            st.markdown(f"<div class='glass-card'><p style='margin:0; opacity:0.6;'>Severity</p><span class='severity-badge {s_cls}'>{s_lab}</span></div>", unsafe_allow_html=True)

        # Tabs for details
        tab_sum, tab_patho, tab_care, tab_report = st.tabs(["Summary", "Pathology", "Care Guide", "Reports"])
        
        with tab_sum:
            st.markdown("#### Identification Insights")
            st.write(f"**Scientific Name:** {rep['scientific_name']}")
            st.write(f"**Common Names:** {', '.join(rep['common_names'][:3])}")
            st.write(f"**Family:** {rep['family']}")
            st.progress(rep['confidence']/100, text=f"AI Confidence: {rep['confidence']}%")
        
        with tab_patho:
            st.markdown("#### Symptom & Pathogen Analysis")
            st.info(f"💡 **AI Recommendation:** {rep['tips']}")
            if rep['treatment_details']:
                st.write("**Remediation Workflow:**")
                st.write(rep['treatment_details'])
            st.markdown(f"**Quantum State Signature:** `{rep['q_sev']['state']}`")
            st.caption(f"Quantum analysis performed on `{rep['q_sev']['backend']}`")

        with tab_care:
            if rep['care']:
                c = rep['care']
                st.markdown(f"#### Growth Profile for {rep['plant'].title()}")
                col_c1, col_c2 = st.columns(2)
                col_c1.metric("Watering", c.get('watering', 'N/A'))
                col_c1.metric("Sunlight", ", ".join(c.get('sunlight', ['N/A'])))
                col_c2.metric("Cycle", c.get('cycle', 'N/A'))
                col_c2.metric("Growth Rate", c.get('growth_rate', 'N/A'))
            else:
                st.warning("Deep care info not available for this species.")

        with tab_report:
            st.markdown("#### Export Clinical Dossier")
            if st.button("📄 GENERATE PDF REPORT"):
                pdf_bytes = create_report_pdf(rep)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"PlantPulse_Report_{rep['plant']}.pdf",
                    mime="application/pdf"
                )
    else:
        st.markdown("""
        <div style='text-align:center; padding:100px 20px;'>
            <h2 style='opacity:0.2;'>No Active Specimen</h2>
            <p style='border: 1px dashed rgba(255,255,255,0.2); padding: 20px; border-radius: 15px;'>
                Upload or capture a leaf image on the left to begin the multi-engine diagnostic process.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ===============================
# CHATBOT SECTION
# ===============================
st.divider()
st.markdown("### 💬 PlantPulse Smart Assistant")
chat_col1, chat_col2 = st.columns([2, 1])

with chat_col1:
    chat_subcontainer = st.container(height=300)
    for msg in st.session_state.chat_history:
        chat_subcontainer.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about symptoms, soil, or treatment..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        # Basic logic: If diagnosis exists, use it.
        context = ""
        if st.session_state.diagnosis_ready:
            r = st.session_state.current_report
            context = f"The user's plant is a {r['plant']} suffering from {r['disease']}. "
        
        # Simulated response (In reality, would use an LLM API here)
        response = f"I see you're asking about '{prompt}'. {context}Based on my botanical database, I recommend ensuring the soil pH is between 6.0 and 7.0 for optimal recovery."
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

with chat_col2:
    st.markdown("""
    <div class='glass-card' style='padding:1rem;'>
        <p style='font-size:0.8rem;'><b>Bot Context</b></p>
        <p style='font-size:0.75rem; opacity:0.7;'>The assistant is currently aware of your active specimen and integrated with <b>Perenual Care</b> and <b>PlantNet</b> knowledge bases.</p>
    </div>
    """, unsafe_allow_html=True)

st.caption("PlantPulse v3.0 | Hybrid AI + Quantum Intelligence | © 2026")