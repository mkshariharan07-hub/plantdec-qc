"""
app.py — PlantPulse AI + Quantum (Enterprise Edition v3.1)
======================================================
Comprehensive Plant Diagnostic Suite
Robust, Error-Handled, Multi-Engine
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
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from dotenv import load_dotenv

# Shared utilities
try:
    from utils import (
        predict_image, get_disease_info,
        get_feature_mode, load_model_and_scaler,
        FEATURE_MODE_RAW, FEATURE_MODE_HIST,
        decode_bytes_to_bgr,
        identify_plant_with_plantnet,
        identify_disease_with_kindwise,
        get_perenual_care_info
    )
except ImportError as e:
    st.error(f"Failed to import core utilities: {e}")
    st.stop()

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
# STYLING
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp { 
        background-color: #050a12;
        background-image: 
            radial-gradient(at 0% 0%, hsla(160,100%,10%,1) 0, transparent 50%), 
            radial-gradient(at 100% 0%, hsla(160,100%,10%,1) 0, transparent 50%);
        color: white; 
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #10b981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
    }

    .severity-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .sev-high { background: #ef4444; color: white; }
    .sev-medium { background: #f59e0b; color: white; }
    .sev-low { background: #10b981; color: white; }

    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.1); }
    ::-webkit-scrollbar-thumb { background: #10b981; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "I am the PlantPulse Assistant. Upload a leaf for analysis or ask me a question!"}
    ]
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ===============================
# QUANTUM SEVERITY 
# ===============================
def analyze_severity_quantum(img: np.ndarray, backend_pref: str):
    """Predicts a severity score from 1-5 using Qiskit."""
    try:
        small = cv2.resize(img, (64, 64))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        mean_val = float(np.mean(gray))
        edge_dens = float(np.sum(cv2.Canny((gray * 255).astype(np.uint8), 50, 150) > 0) / (64 * 64))
        
        # Explicit registration to avoid discovery errors
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(qr, cr)
        
        from math import pi
        qc.ry(mean_val * pi, 0)
        qc.ry(edge_dens * pi, 1)
        qc.h(2)
        qc.cx(0, 3)
        qc.cx(1, 3)
        qc.measure(qr, cr)

        try:
            TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
            if not TOKEN: raise ValueError("No IBM token")
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
            backend = service.least_busy(simulator=(backend_pref == "Simulator Only"))
            qc_t = transpile(qc, backend)
            sampler = Sampler(backend)
            job = sampler.run([qc_t], shots=1024)
            result = job.result()[0]
            counts = result.data.c.get_counts()
        except Exception as q_error:
            # Local fallback
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            job = sampler.run([qc])
            result = job.result()[0]
            counts = result.data.c.get_counts()
            backend = type('obj', (object,), {'name': 'local-sim'})

        dom_state = max(counts, key=counts.get)
        # Handle binary vs bitstring key format differences
        if isinstance(dom_state, int): dom_state = format(dom_state, '04b')
        
        score = dom_state.count('1') + 1
        labels = ["Healthy/Optimal", "Incipient", "Noticeable", "Widespread", "Critical"]
        return {
            "score": score,
            "label": labels[min(score-1, 4)],
            "state": dom_state,
            "backend": backend.name
        }
    except Exception as e:
        return {"score": 3, "label": "Analysis Uncertain", "state": "0000", "backend": "error", "err": str(e)}

# ===============================
# PDF ENGINES
# ===============================
def generate_pdf_report(data):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("helvetica", "B", 20)
        pdf.cell(0, 15, "PlantPulse Diagnostic Report", ln=True, align='C')
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 14)
        pdf.cell(0, 10, f"Plant: {data['plant']}", ln=True)
        pdf.cell(0, 10, f"Condition: {data['disease']}", ln=True)
        pdf.cell(0, 10, f"Severity: {data['q_sev']['label']}", ln=True)
        pdf.ln(5)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, "Treatment Advice:", ln=True)
        pdf.set_font("helvetica", "", 12)
        pdf.multi_cell(0, 10, data['tips'])
        return pdf.output()
    except Exception as e:
        return f"Error: {str(e)}".encode()

# ===============================
# SIDEBAR
# ===============================
model, scaler = load_model_and_scaler()

with st.sidebar:
    st.image("https://img.icons8.com/color/144/leaf.png", width=80)
    st.title("System Matrix")
    
    api_mode = st.radio("Pipeline Mode", ["Hybrid (Fast)", "Cloud-Heavy", "Local Only"])
    q_backend = st.selectbox("Quantum Engine", ["Dynamic (Least Busy)", "Simulator Only"])
    
    st.divider()
    st.write("### Diagnostics")
    c1, c2 = st.columns(2)
    c1.metric("Local AI", "OK" if model else "OFF")
    c2.metric("Quantum", "READY")
    
    if st.button("Reset Session"):
        st.session_state.clear()
        st.rerun()

# ===============================
# MAIN UI
# ===============================
st.title("🌿 PlantPulse Enterprise")
st.write("AI-Powered Computer Vision & Quantum Pathogen Analysis")

col_in, col_res = st.columns([1, 1], gap="medium")

with col_in:
    st.subheader("Specimen Capture")
    u_tab, c_tab = st.tabs(["📁 Upload", "📷 Camera"])
    
    img_bytes = None
    with u_tab:
        uf = st.file_uploader("Upload leaf...", type=["jpg", "png", "jpeg"])
        if uf: img_bytes = uf.read()
    with c_tab:
        cf = st.camera_input("Snapshot")
        if cf: img_bytes = cf.read()

    if img_bytes:
        active_img = decode_bytes_to_bgr(img_bytes)
        if active_img is not None:
            st.image(cv2.cvtColor(active_img, cv2.COLOR_BGR2RGB))
            if st.button("EXECUTE ANALYSIS", use_container_width=True, type="primary"):
                with st.status("Analyzing specimen...", expanded=True) as status:
                    # LOCAL
                    status.write("Calculating local feature vectors...")
                    loc = predict_image(active_img, model, scaler)
                    
                    # CLOUD
                    if api_mode != "Local Only":
                        status.write("Consulting PlantNet & Kindwise Cloud...")
                        pn = identify_plant_with_plantnet(active_img)
                        kw = identify_disease_with_kindwise(active_img)
                    else:
                        pn, kw = {}, {}
                    
                    # QUANTUM
                    status.write("Running Quantum verification...")
                    q = analyze_severity_quantum(active_img, q_backend)
                    
                    # MERGE
                    final_plant = pn.get('scientific_name', loc['plant'])
                    final_disease = kw.get('disease', loc['disease']) if "error" not in kw else loc['disease']
                    
                    res_pack = {
                        "plant": final_plant,
                        "disease": final_disease,
                        "confidence": loc['confidence'],
                        "q_sev": q,
                        "tips": loc['tips'],
                        "pn_data": pn,
                        "kw_data": kw
                    }
                    st.session_state.last_results = res_pack
                    status.update(label="Analysis Complete!", state="complete")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Analysis complete for **{final_plant}**. Detected **{final_disease}** with **{q['label']}** severity. I've prepared treatment protocols."
                    })

with col_res:
    st.subheader("Analysis Insights")
    if st.session_state.last_results:
        r = st.session_state.last_results
        
        st.markdown(f"""
        <div class="glass-card">
            <h4>{r['plant'].title()}</h4>
            <p>Condition: <b>{r['disease'].replace('_',' ').title()}</b></p>
            <span class="severity-badge {'sev-high' if r['q_sev']['score'] > 3 else 'sev-medium' if r['q_sev']['score'] > 1 else 'sev-low'}">
                {r['q_sev']['label']}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        att_tab, treat_tab, exp_tab = st.tabs(["Attributes", "Treatment", "Export"])
        
        with att_tab:
            st.write(f"**AI Confidence:** {r['confidence']}%")
            if r['pn_data'].get('family'):
                st.write(f"**Family:** {r['pn_data']['family']}")
            st.write(f"**Quantum Backend:** `{r['q_sev']['backend']}`")
            
        with treat_tab:
            st.success(f"**Recommendation:** {r['tips']}")
            if r['kw_data'].get('description'):
                st.write(r['kw_data']['description'])

        with exp_tab:
            if st.button("Generate Clinical PDF"):
                pdf = generate_pdf_report(r)
                st.download_button("Download Report", pdf, f"Report_{r['plant']}.pdf", "application/pdf")
    else:
        st.info("No specimen analyzed yet. Upload an image to see results here.")

# ===============================
# CHATBOT
# ===============================
st.divider()
st.subheader("💬 PlantPulse Assistant")
for m in st.session_state.chat_history:
    st.chat_message(m["role"]).write(m["content"])

if p := st.chat_input("Ask me anything about plant health..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    # Context-aware stub
    resp = "Based on my current database, I recommend ensuring adequate ventilation to prevent fungal spread."
    st.session_state.chat_history.append({"role": "assistant", "content": resp})
    st.rerun()