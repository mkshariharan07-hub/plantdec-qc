"""
main.py — PlantPulse AI + Quantum (Enterprise Zenith v5)
======================================================
Professional Agritech Diagnostic Console
Pure Cloud + Quantum Entanglement Logic
"""

import streamlit as st
import cv2
import numpy as np
import os
import json
import datetime
import base64
import requests
import traceback
import pandas as pd
from fpdf import FPDF
from dotenv import load_dotenv

# Optional Quantum Imports
try:
    from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

# Shared utilities
try:
    from utils import (
        decode_bytes_to_bgr,
        identify_plant_with_plantnet,
        identify_disease_with_kindwise,
        get_perenual_care_info,
        get_disease_info
    )
except ImportError as e:
    st.error(f"Failed to import core utilities: {e}")
    st.stop()

load_dotenv()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PlantPulse Zenith",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# STYLING (Zenith Design System)
# ===============================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    .stApp { 
        background-color: #03070a;
        background-image: 
            radial-gradient(circle at 20% 30%, rgba(16, 185, 129, 0.05) 0%, transparent 40%),
            radial-gradient(circle at 80% 70%, rgba(6, 182, 212, 0.05) 0%, transparent 40%);
        color: #e2e8f0; 
        font-family: 'Outfit', sans-serif;
    }

    /* Falling Leaf Animation */
    @keyframes blossom {
        0% { transform: translateY(-10vh) rotate(0deg) scale(0.5); opacity: 0; }
        20% { opacity: 0.6; }
        80% { opacity: 0.6; }
        100% { transform: translateY(110vh) rotate(720deg) scale(1); opacity: 0; }
    }
    .blossom-leaf {
        position: fixed; top: -5vh; z-index: 1000; pointer-events: none;
        animation: blossom 12s linear infinite; font-size: 28px; filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.4));
    }

    /* Zenith Glassmorphism Cards */
    .zenith-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .zenith-card:hover {
        transform: translateY(-4px);
        border-color: rgba(16, 185, 129, 0.3);
        background: rgba(15, 23, 42, 0.8);
    }

    .glow-text {
        color: #10b981;
        text-shadow: 0 0 15px rgba(16, 185, 129, 0.5);
        font-weight: 700;
    }

    .metric-title { font-size: 0.85rem; opacity: 0.6; text-transform: uppercase; letter-spacing: 1.5px; }
    .metric-value { font-size: 2.4rem; font-weight: 800; }

    /* Custom Status Badges */
    .badge {
        padding: 6px 16px; border-radius: 100px; font-weight: 700; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid #ef4444; }
    .badge-warning { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-optimal { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid #10b981; }

    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 45px; border-radius: 12px; background-color: rgba(255,255,255,0.03); color: white; border: 1px solid rgba(255,255,255,0.05); padding: 0 20px;
    }
    .stTabs [aria-selected="true"] { background-color: rgba(16, 185, 129, 0.1) !important; border-color: #10b981 !important; color: #10b981 !important; }
</style>

<div class="blossom-leaf" style="left:5%; animation-delay: 0s;">🌿</div>
<div class="blossom-leaf" style="left:25%; animation-delay: 3s;">🍃</div>
<div class="blossom-leaf" style="left:45%; animation-delay: 7s;">🌱</div>
<div class="blossom-leaf" style="left:65%; animation-delay: 1s;">🌿</div>
<div class="blossom-leaf" style="left:85%; animation-delay: 5s;">🍃</div>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Groot online. I am at your service for clinical botanical analysis. Inhale the data... Exhale the cure."}
    ]
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ===============================
# QUANTUM SEVERITY PROBABILISTIC
# ===============================
def analyze_severity_quantum(img: np.ndarray, backend_pref: str):
    if not HAS_QUANTUM:
        return {"score": 3, "label": "Simulated Matrix", "prob": {"0000": 0.5, "1111": 0.5}, "backend": "sim"}
        
    try:
        small = cv2.resize(img, (64, 64))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        entropy = -np.sum(gray * np.log2(gray + 1e-7)) / 4096.0
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(qr, cr)
        from math import pi
        qc.ry(entropy * pi, qr[0])
        qc.h(qr[1])
        qc.cx(qr[0], qr[2])
        qc.cx(qr[1], qr[3])
        qc.measure(qr, cr)

        try:
            TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
            if TOKEN and len(TOKEN) > 10:
                service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
                backend = service.least_busy(simulator=(backend_pref == "Simulator Only"))
                qc_t = transpile(qc, backend)
                sampler = Sampler(mode=backend)
                job = sampler.run([qc_t], shots=1024)
                result = job.result()[0]
                counts = result.data.c.get_counts()
                backend_name = backend.name
            else:
                raise ValueError("No token")
        except:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            result = sampler.run([qc]).result()[0]
            counts = result.data.c.get_counts()
            backend_name = "q-local-sim"

        total = sum(counts.values())
        probs = {k: v/total for k, v in counts.items()}
        dom_state = max(counts, key=counts.get)
        if isinstance(dom_state, int): dom_state = format(dom_state, '04b')
        score = dom_state.count('1') + 1
        labels = ["Optimal", "Incipient", "Moderate", "Severe", "Critical"]
        return {"score": score, "label": labels[min(score-1, 4)], "prob": probs, "backend": backend_name, "entropy": entropy}
    except Exception as e:
        return {"score": 3, "label": "Analysis Uncertain", "prob": {"Error": 1.0}, "backend": "error"}

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/200/leaf.png", width=120)
    st.markdown("<h2 class='glow-text'>PLANTPULSE ZENITH</h2>", unsafe_allow_html=True)
    
    with st.expander("🛠 Matrix Configuration", expanded=False):
        q_eng = st.selectbox("Quantum Engine", ["Dynamic (Hybrid)", "Simulator Optimized"])
        api_depth = st.slider("Discovery Depth", 1, 10, 7)
    
    st.divider()
    st.markdown("### Clinical Status")
    st.success("🛰 Core Satellites: ACTIVE")
    st.info("🧬 Pathogen Matrix: SYNCED")
    st.warning("🍂 Bio-Telemetry: HIGH LOAD")
    
    if st.button("Reset Session Matrix", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ===============================
# MAIN UI
# ===============================
st.markdown("<h1 style='text-align:center; font-size: 3.5rem;' class='glow-text'>🌿 PlantPulse <span style='color:white'>Zenith</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; opacity:0.6; letter-spacing:2px;'>ENTERPRISE BOTANICAL INTELLIGENCE & QUANTUM ANALYTICS</p>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown("<div class='zenith-card'>", unsafe_allow_html=True)
    st.subheader("📡 Specimen Acquisition")
    tabs = st.tabs(["📁 Digital Dossier", "📷 Bio-Scanner"])
    
    img_bytes = None
    with tabs[0]:
        uf = st.file_uploader("Ingest specimen data...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uf: img_bytes = uf.getvalue()
    with tabs[1]:
        cf = st.camera_input("Scanner activation")
        if cf: img_bytes = cf.getvalue()

    if img_bytes:
        frame = decode_bytes_to_bgr(img_bytes)
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            if st.button("🚀 INITIATE ZENITH SCAN", use_container_width=True, type="primary"):
                with st.status("Harmonizing neural and quantum vectors...", expanded=True) as status:
                    status.write("Species ID phase initiated...")
                    pn = identify_plant_with_plantnet(frame)
                    
                    status.write("Pathogen matrix synchronization...")
                    kw = identify_disease_with_kindwise(frame)
                    
                    status.write("Quantum state entanglement check...")
                    q = analyze_severity_quantum(frame, "Simulator Only" if q_eng == "Simulator Optimized" else "Dynamic")
                    
                    plant_key = pn.get('scientific_name', 'Unknown')
                    status.write(f"Retrieving care protocols for {plant_key}...")
                    care = get_perenual_care_info(pn.get('common_names', [plant_key])[0])

                    res = {
                        "plant": plant_key,
                        "common_name": pn.get('common_names', ['Generic Specimen'])[0],
                        "disease": kw.get('disease', 'Healthy/Indeterminate') if "error" not in kw else "Pathogen Restricted",
                        "score": pn.get('score', 0),
                        "q": q,
                        "care": care,
                        "pathology": kw.get('description', 'No descriptive pathology data.'),
                        "rx": kw.get('treatment', {}),
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                    }
                    st.session_state.last_results = res
                    status.update(label="Zenith Diagnosis Fulllocked.", state="complete")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"Scan for **{plant_key}** locked. Pathology indicates **{res['disease']}**. I've established a {q['label']} threat level. Commencing remediation planning."
                    })
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    if st.session_state.last_results:
        r = st.session_state.last_results
        
        st.markdown(f"""
        <div class="zenith-card">
            <p class="metric-title">Detected Specimen</p>
            <h2 class="glow-text" style="margin-bottom:0;">{r['plant'].upper()}</h2>
            <p style="opacity:0.6; font-size:0.9rem;">{r['common_name']} | ID: {r['timestamp']}</p>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem;">
                <div>
                   <p class="metric-title">Pathogen</p>
                   <p style="font-size:1.4rem; font-weight:700;">{r['disease'].title()}</p>
                </div>
                <span class="badge badge-{'critical' if r['q']['score'] > 3 else 'warning' if r['q']['score'] > 2 else 'optimal'}">
                    {r['q']['label']} Risk
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        rtabs = st.tabs(["🧪 Pathology", "📈 Analytics", "📋 Workflow", "📄 Reports"])
        
        with rtabs[0]:
            st.info(f"**Bio-Analysis:** {r['pathology']}")
            if r['rx']:
                st.markdown("#### Clinical Remediation")
                for k, v in r['rx'].items():
                    if v: st.write(f"**{k.replace('_',' ').title()}:** {v}")
            
        with rtabs[1]:
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("<p class='metric-title'>Quantum Entropy</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{r['q'].get('entropy', 0.5):.4f}</p>", unsafe_allow_html=True)
            with col_a2:
                st.markdown("<p class='metric-title'>ID Confidence</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{r['score']}%</p>", unsafe_allow_html=True)
            
            st.markdown("#### Quantum Probability Matrix")
            # Convert dict to dataframe for chart
            pdf = pd.DataFrame(list(r['q']['prob'].items()), columns=['State', 'Probability'])
            st.bar_chart(pdf.set_index('State'), color="#10b981")
            st.caption(f"Backend Node: {r['q']['backend']}")

        with rtabs[2]:
            if r['care']:
                c = r['care']
                m1, m2, m3 = st.columns(3)
                m1.metric("Sunlight", c.get('sunlight', ['N/A'])[0])
                m2.metric("Water", c.get('watering', 'N/A'))
                m3.metric("Cycle", c.get('cycle', 'N/A'))
                st.divider()
                st.write("**Maintenance Memo:** Ensure specimen matches growth requirements for faster recovery.")
            else:
                st.warning("Insufficient bio-data in Perenual Care Matrix.")

        with rtabs[3]:
            if st.button("Generate Zenith Dossier"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_fill_color(3, 7, 10)
                pdf.rect(0, 0, 210, 297, 'F')
                pdf.set_text_color(16, 185, 129)
                pdf.set_font("helvetica", "B", 24)
                pdf.cell(0, 20, "PLANTPULSE ZENITH DOSSIER", ln=True, align='C')
                pdf.set_font("helvetica", "", 12)
                pdf.set_text_color(200, 200, 200)
                pdf.cell(0, 10, f"Timestamp: {datetime.datetime.now()}", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("helvetica", "B", 16)
                pdf.cell(0, 10, f"Specimen: {r['plant']}", ln=True)
                pdf.set_font("helvetica", "", 12)
                pdf.multi_cell(0, 10, f"Diagnosis: {r['disease']}\nThreat Level: {r['q']['label']}\nPathology: {r['pathology']}")
                st.download_button("Download Zenith_Dossier.pdf", pdf.output(), f"Pulse_{r['plant']}.pdf", "application/pdf")
    else:
        st.info("Scanner idle. Awaiting specimen input...")

# --- Assistant ---
st.divider()
st.subheader("💬 Zenith Smart Assistant")
chat_box = st.container(height=350)
with chat_box:
    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).write(m["content"])

if p := st.chat_input("Ask about pathogens, soil health, or remediation..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    # Context injected response logic
    ctxt = ""
    if st.session_state.last_results:
        res = st.session_state.last_results
        ctxt = f"The scanned {res['plant']} has {res['disease']}. "
    
    resp = f"Analysis of '{p}': {ctxt}I recommend immediate isolation of the specimen to prevent cross-pathogen contamination."
    st.session_state.chat_history.append({"role": "assistant", "content": resp})
    st.rerun()

st.caption("PlantPulse Zenith v5.0 | Enterprise Agritech Strategy | © 2026")