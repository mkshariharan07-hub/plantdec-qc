"""
app.py — PlantPulse AI + Quantum (Full-Cloud Enterprise v4)
======================================================
Comprehensive Plant Diagnostic Suite
Pure Cloud-API Pipeline: PlantNet + Kindwise
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
        get_disease_info # Still useful for emojis/colors if needed
    )
except ImportError as e:
    st.error(f"Failed to import core utilities: {e}")
    st.stop()

load_dotenv()

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PlantPulse Cloud Console",
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

    /* Falling Leaf Animation */
    @keyframes fall {
        0% { transform: translateY(-100px) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(100vh) rotate(360deg); opacity: 0; }
    }
    .leaf-pix {
        position: fixed; top: -50px; z-index: 1000; pointer-events: none;
        animation: fall 10s linear infinite; font-size: 24px;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
    }

    .severity-badge {
        padding: 4px 12px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .sev-high { background: #ef4444; }
    .sev-medium { background: #f59e0b; }
    .sev-low { background: #10b981; }
</style>
<div class="leaf-pix" style="left:10%; animation-delay: 0s;">🌿</div>
<div class="leaf-pix" style="left:30%; animation-delay: 2s;">🍃</div>
<div class="leaf-pix" style="left:70%; animation-delay: 1s;">🌱</div>
<div class="leaf-pix" style="left:90%; animation-delay: 4s;">🌿</div>
""", unsafe_allow_html=True)

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to PlantPulse Cloud. I use PlantNet and Kindwise for high-accuracy diagnostics."}
    ]
if "last_results" not in st.session_state:
    st.session_state.last_results = None

# ===============================
# QUANTUM SEVERITY 
# ===============================
def analyze_severity_quantum(img: np.ndarray, backend_pref: str):
    if not HAS_QUANTUM:
        return {"score": 3, "label": "Cloud Baseline", "state": "1010", "backend": "sim"}
        
    try:
        small = cv2.resize(img, (64, 64))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        mean_val = float(np.mean(gray))
        
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        qc = QuantumCircuit(qr, cr)
        from math import pi
        qc.ry(mean_val * pi, qr[0])
        qc.h(qr[1])
        qc.cx(qr[0], qr[2])
        qc.measure(qr, cr)

        counts = {}
        try:
            TOKEN = os.getenv("IBM_QUANTUM_TOKEN", "")
            if TOKEN and len(TOKEN) > 10:
                service = QiskitRuntimeService(channel="ibm_quantum_platform", token=TOKEN)
                backend = service.least_busy(simulator=(backend_pref == "Simulator Only"))
                qc_t = transpile(qc, backend)
                sampler = Sampler(mode=backend)
                job = sampler.run([qc_t], shots=1024)
                counts = job.result()[0].data.c.get_counts()
                backend_name = backend.name
            else:
                raise ValueError("No token")
        except:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            result = sampler.run([qc]).result()[0]
            counts = result.data.c.get_counts()
            backend_name = "local-sim"

        dom_state = max(counts, key=counts.get)
        if isinstance(dom_state, int): dom_state = format(dom_state, '04b')
        score = dom_state.count('1') + 1
        labels = ["Healthy Condition", "Early Warning", "Moderate Infection", "Advanced Decay", "Critical Failure"]
        return {"score": score, "label": labels[min(score-1, 4)], "state": dom_state, "backend": backend_name}
    except Exception as e:
        return {"score": 3, "label": "Local Fallback", "state": "0000", "backend": "error", "err": str(e)}

# ===============================
# UI PANELS
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/color/144/leaf.png", width=80)
    st.title("Cloud Intelligence")
    
    q_backend = st.selectbox("Quantum Engine", ["Dynamic (Real HW)", "Simulator Only"])
    
    st.divider()
    st.write("### Cloud Status")
    st.status("PlantNet V2: Live", state="complete")
    st.status("Kindwise Crop: Live", state="complete")
    st.status("Perenual Care: Live", state="complete")
    
    if st.button("Reset Matrix"):
        st.session_state.clear()
        st.rerun()

st.title("🌿 PlantPulse Cloud")
st.write("Professional Diagnostic Engine | Powered by PlantNet & Kindwise")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Image Ingestion")
    tabs = st.tabs(["📁 Local Upload", "📷 Camera Capture"])
    img_bytes = None
    with tabs[0]:
        uf = st.file_uploader("Upload leaf...", type=["jpg", "png", "jpeg"])
        if uf: img_bytes = uf.getvalue()
    with tabs[1]:
        cf = st.camera_input("Snapshot")
        if cf: img_bytes = cf.getvalue()

    if img_bytes:
        frame = decode_bytes_to_bgr(img_bytes)
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            if st.button("EXECUTE CLOUD DIAGNOSTIC", use_container_width=True, type="primary"):
                with st.status("Connecting to global botanical servers...", expanded=True) as status:
                    # 1. Identification
                    status.write("Running PlantNet Species Identification...")
                    pn = identify_plant_with_plantnet(frame)
                    
                    # 2. Disease
                    status.write("Running Kindwise Pathogen Analysis...")
                    kw = identify_disease_with_kindwise(frame)
                    
                    # 3. Quantum
                    status.write("Calculating Quantum Severity Level...")
                    q = analyze_severity_quantum(frame, q_backend)
                    
                    # 4. Care Info
                    plant_name = pn.get('scientific_name', 'Unknown Plant')
                    status.write(f"Fetching species profile for {plant_name}...")
                    care = get_perenual_care_info(pn.get('common_names', [plant_name])[0])

                    res = {
                        "plant": plant_name,
                        "common_name": pn.get('common_names', ['N/A'])[0],
                        "disease": kw.get('disease', 'Healthy/Undetected') if "error" not in kw else "Unknown",
                        "confidence": pn.get('score', 0),
                        "q_sev": q,
                        "care": care,
                        "description": kw.get('description', 'No detailed description available.'),
                        "treatment": kw.get('treatment', {})
                    }
                    st.session_state.last_results = res
                    status.update(label="Diagnostic Completed Successfully!", state="complete")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"I've identified this as **{plant_name}**. The condition appears to be **{res['disease']}** with a **{q['label']}** severity. How can I help with the treatment?"
                    })

with col_right:
    st.subheader("Diagnostic Bio-Feed")
    if st.session_state.last_results:
        r = st.session_state.last_results
        
        st.markdown(f"""
        <div class="glass-card">
            <h4>{r['plant']}</h4>
            <p style='color:#94a3b8; font-size:0.9rem;'>{r['common_name']}</p>
            <p>Condition: <b>{r['disease'].title()}</b></p>
            <span class="severity-badge {'sev-high' if r['q_sev']['score'] > 3 else 'sev-medium' if r['q_sev']['score'] > 1 else 'sev-low'}">
                {r['q_sev']['label']}
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        rtabs = st.tabs(["Pathology", "Care Profile", "Export"])
        
        with rtabs[0]:
            st.write(f"**Species Score:** {r['confidence']}%")
            st.info(f"**Diagnostic Description:** {r['description']}")
            if r['treatment']:
                st.write("**Treatment Protocols:**")
                st.write(r['treatment'])
            
        with rtabs[1]:
            if r['care']:
                st.write(f"**Watering:** {r['care'].get('watering', 'N/A')}")
                st.write(f"**Sunlight:** {', '.join(r['care'].get('sunlight', ['N/A']))}")
                st.write(f"**Lifecycle:** {r['care'].get('cycle', 'N/A')}")
            else:
                st.warning("Perenual Care database returned no specific matches for this species.")

        with rtabs[2]:
            if st.button("Build Detailed PDF"):
                # Simplified PDF generator
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("helvetica", "B", 16)
                pdf.cell(0, 10, f"PlantPulse Report: {r['plant']}", ln=True)
                pdf.set_font("helvetica", "", 12)
                pdf.cell(0, 10, f"Condition: {r['disease']}", ln=True)
                pdf.cell(0, 10, f"Severity: {r['q_sev']['label']}", ln=True)
                pdf.ln(5)
                pdf.multi_cell(0, 10, f"Description: {r['description']}")
                report_bytes = pdf.output()
                st.download_button("Download clinical_file.pdf", report_bytes, "PlantPulse_Report.pdf", "application/pdf")
    else:
        st.info("Awaiting specimen in the diagnostic chamber...")

# --- Chatbot ---
st.divider()
st.subheader("💬 PlantPulse Assistant")
for m in st.session_state.chat_history:
    st.chat_message(m["role"]).write(m["content"])

if p := st.chat_input("Ask about symptoms or soil..."):
    st.session_state.chat_history.append({"role": "user", "content": p})
    st.rerun()

st.caption("PlantPulse v4.0 | Pure Cloud Intelligence | © 2026")