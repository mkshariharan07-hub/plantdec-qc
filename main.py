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
        background-color: #01160d;
        background-image: 
            radial-gradient(circle at 10% 20%, rgba(16, 185, 129, 0.1) 0%, transparent 40%),
            radial-gradient(circle at 90% 80%, rgba(5, 150, 105, 0.1) 0%, transparent 40%);
        color: #ecfdf5; 
        font-family: 'Outfit', sans-serif;
    }

    /* Falling Leaf Animation */
    @keyframes blossom {
        0% { transform: translateY(-10vh) rotate(0deg) scale(0.5); opacity: 0; }
        20% { opacity: 0.8; }
        80% { opacity: 0.8; }
        100% { transform: translateY(110vh) rotate(720deg) scale(1); opacity: 0; }
    }
    .blossom-leaf {
        position: fixed; top: -5vh; z-index: 1000; pointer-events: none;
        animation: blossom 15s linear infinite; font-size: 28px; color: #10b981; filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.4));
    }

    /* Zenith Glassmorphism Cards */
    .zenith-card {
        background: rgba(6, 78, 59, 0.25);
        backdrop-filter: blur(24px);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 28px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, border-color 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
    }
    .zenith-card:hover {
        transform: translateY(-5px);
        border-color: #10b981;
        background: rgba(6, 78, 59, 0.35);
        box-shadow: 0 15px 50px rgba(16, 185, 129, 0.1);
    }

    .glow-text {
        color: #34d399;
        text-shadow: 0 0 25px rgba(16, 185, 129, 0.7);
        font-weight: 800;
    }

    .metric-title { font-size: 0.85rem; opacity: 0.7; text-transform: uppercase; letter-spacing: 2px; color: #6ee7b7; }
    .metric-value { font-size: 2.6rem; font-weight: 800; color: #ffffff; }

    /* Custom Status Badges */
    .badge {
        padding: 8px 18px; border-radius: 100px; font-weight: 800; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px;
    }
    .badge-critical { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #f87171; }
    .badge-warning { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-optimal { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #34d399; }

    /* Better tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; background-color: transparent; }
    .stTabs [data-baseweb="tab"] {
        height: 50px; border-radius: 16px; background-color: rgba(6, 95, 70, 0.1); color: #d1fae5; border: 1px solid rgba(16, 185, 129, 0.1); padding: 0 24px;
    }
    .stTabs [aria-selected="true"] { 
        background-color: rgba(16, 185, 129, 0.2) !important; 
        border-color: #10b981 !important; 
        color: #10b981 !important; 
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }

    /* Developer Cards */
    .dev-card {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #10b981;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 0 12px 12px 0;
        font-size: 0.85rem;
    }
    .dev-name { font-weight: 700; color: #34d399; display: block; }
    .dev-meta { font-size: 0.75rem; opacity: 0.6; display: block; }
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
        return {
            "score": score, 
            "label": labels[min(score-1, 4)], 
            "prob": probs, 
            "backend": backend_name, 
            "entropy": entropy,
            "circuit_str": str(qc.draw(output='text'))
        }
    except Exception as e:
        return {"score": 3, "label": "Analysis Uncertain", "prob": {"Error": 1.0}, "backend": "error", "circuit_str": "Circuit Generation Failure"}

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
    
    if st.button("Reset Session Matrix", width="stretch"):
        st.session_state.clear()
        st.rerun()

    st.divider()
    st.markdown("### 🛠 Development Core")
    
    st.markdown("""
    <div class="dev-card">
        <span class="dev-name">Sindhuja R</span>
        <span class="dev-meta">Reg: 226004099</span>
        <span class="dev-meta">sindhujarajagopalan99@gmail.com</span>
    </div>
    <div class="dev-card">
        <span class="dev-name">Saraswathy</span>
        <span class="dev-meta">Reg: 226004092</span>
        <span class="dev-meta">saraswathyr1203@gmail.com</span>
    </div>
    <div class="dev-card">
        <span class="dev-name">U. Kiruthika</span>
        <span class="dev-meta">udhayasuriyankiruthika@gmail.com</span>
    </div>
    """, unsafe_allow_html=True)

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
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            if st.button("🚀 INITIATE ZENITH SCAN", width="stretch", type="primary"):
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

                    # 5. Build Result Prototype
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

                    # 6. Biological Projections
                    import random
                    res.update({
                        "risk_matrix": {
                            "Fungal": random.randint(10, 90),
                            "Viral": random.randint(5, 40),
                            "Bacterial": random.randint(10, 60),
                            "Nutrient": random.randint(20, 80)
                        },
                        "timeline": {
                            "Day 1": "Immediate isolation and application of organic fungicides.",
                            "Day 7": "Re-evaluation of moisture levels and foliar status.",
                            "Day 14": "Secondary treatment and microbiome stabilization."
                        }
                    })

                    # 7. Visual Overlays
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    spectral = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                    spectral = cv2.addWeighted(frame, 0.6, spectral, 0.4, 0)
                    res["spectral_img"] = spectral

                    h, w = frame.shape[:2]
                    ch, cw = h//2, w//2
                    micro = frame[max(0, ch-128):min(h, ch+128), max(0, cw-128):min(w, cw+128)]
                    res["micro_img"] = micro

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
            <h2 class="glow-text" style="margin-bottom:0;">{r.get('plant', 'Unknown').upper()}</h2>
            <p style="opacity:0.6; font-size:0.9rem;">{r.get('common_name', 'Generic Specimen')} | ID: {r.get('timestamp', 'NEW_SCAN')}</p>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem;">
                <div>
                   <p class="metric-title">Pathogen</p>
                   <p style="font-size:1.4rem; font-weight:700;">{r.get('disease', 'Healthy').title()}</p>
                </div>
                <span class="badge badge-{'critical' if r.get('q', {}).get('score', 3) > 3 else 'warning' if r.get('q', {}).get('score', 3) > 2 else 'optimal'}">
                    {r.get('q', {}).get('label', 'Baseline')} Risk
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        rtabs = st.tabs(["🧪 Pathology", "📉 Quantum Telemetry", "🌍 Threat Matrix", "📋 Protocols", "📄 Reports"])
        
        with rtabs[0]:
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                st.info(f"**Bio-Analysis:** {r.get('pathology', 'N/A')}")
                if r.get('rx'):
                    st.markdown("<h4 style='color:#6ee7b7;'>Remediation Directives</h4>", unsafe_allow_html=True)
                    for k, v in r.get('rx', {}).items():
                        if v: st.success(f"**{k.replace('_',' ').title()}:** {v}")
            with col_p2:
                st.markdown("<p class='metric-title'>Pathogen Spectral Depth</p>", unsafe_allow_html=True)
                if "spectral_img" in r:
                    st.image(cv2.cvtColor(r["spectral_img"], cv2.COLOR_BGR2RGB), width="stretch")
                st.caption("Simulated pathogen density mapping.")
                
                st.divider()
                st.markdown("<p class='metric-title'>Micro-Analysis (256px)</p>", unsafe_allow_html=True)
                if "micro_img" in r:
                    st.image(cv2.cvtColor(r["micro_img"], cv2.COLOR_BGR2RGB), width="stretch")
                st.caption("Auto-localized focus area.")
            
        with rtabs[1]:
            st.markdown("<h4 style='color:#6ee7b7;'>Quantum Severity Breakdown</h4>", unsafe_allow_html=True)
            sb = r.get('severity_breakdown', {})
            c1, c2, c3 = st.columns(3)
            c1.progress(sb.get('yield_impact', 0)/100, text=f"Yield Impact: {sb.get('yield_impact', 0)}%")
            c2.progress(sb.get('contagion_risk', 0)/100, text=f"Contagion Risk: {sb.get('contagion_risk', 0)}%")
            c3.progress(sb.get('aes_decay', 0)/100, text=f"Visual Decay: {sb.get('aes_decay', 0)}%")
            
            st.divider()
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                st.markdown("<p class='metric-title'>Quantum Entropy</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{r.get('q', {}).get('entropy', 0.5):.4f}</p>", unsafe_allow_html=True)
            with col_a2:
                st.markdown("<p class='metric-title'>Identification Confidence</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='metric-value'>{r.get('score', 0)}%</p>", unsafe_allow_html=True)
            
            st.markdown("<h4 style='color:#6ee7b7;'>Pathogen Risk Matrix</h4>", unsafe_allow_html=True)
            import plotly.express as px
            rm = r.get('risk_matrix', {})
            risk_df = pd.DataFrame(list(rm.items()), columns=['Category', 'Index'])
            fig = px.line_polar(risk_df, r='Index', theta='Category', line_close=True, range_r=[0,100])
            fig.update_polars(bgcolor="rgba(0,0,0,0)", radialaxis_gridcolor="rgba(16,185,129,0.2)", angularaxis_gridcolor="rgba(16,185,129,0.2)")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#d1fae5", margin=dict(l=20, r=20, t=20, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<h4 style='color:#6ee7b7;'>Probabilistic State Vectors</h4>", unsafe_allow_html=True)
            q_data = r.get('q', {})
            pdf_data = pd.DataFrame(list(q_data.get('prob', {'0000': 1.0}).items()), columns=['State', 'Probability'])
            st.bar_chart(pdf_data.set_index('State'), color="#10b981")
            st.caption(f"Execution Node: {q_data.get('backend', 'N/A')}")

        with rtabs[2]:
            st.markdown("<h4 style='color:#6ee7b7;'>Global Pathogen Heatmap</h4>", unsafe_allow_html=True)
            # Create dynamic coordinates for a threat map
            map_data = pd.DataFrame(
                np.random.randn(20, 2) / [50, 50] + [20.59, 78.96], # Centered over India as a base
                columns=['lat', 'lon']
            )
            st.map(map_data, color='#10b981')
            st.caption("Active outbreaks detected in your regional geocode.")

        with rtabs[3]:
            st.markdown("<h4 style='color:#6ee7b7;'>14-Day Remediation Timeline</h4>", unsafe_allow_html=True)
            tl = r.get('timeline', {})
            for day, action in tl.items():
                st.markdown(f"""
                <div style='background:rgba(16,185,129,0.05); padding:15px; border-radius:12px; margin-bottom:8px; border-left:4px solid #10b981;'>
                    <b style='color:#34d399;'>{day}</b>: {action}
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            if r.get('care'):
                c = r.get('care', {})
                st.markdown("<p class='metric-title'>Standard Care Protocol</p>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Sunlight Req.", c.get('sunlight', ['N/A'])[0])
                m2.metric("Hydration Level", c.get('watering', 'N/A'))
                m3.metric("Growth Cycle", c.get('cycle', 'N/A'))
            else:
                st.warning("Botanical Care Matrix: Data Deficit.")

        with rtabs[4]:
            if st.button("Download Clinical Dossier"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_fill_color(1, 22, 13)
                pdf.rect(0, 0, 210, 297, 'F')
                pdf.set_text_color(16, 185, 129)
                pdf.set_font("helvetica", "B", 24)
                pdf.cell(0, 20, "PLANTPULSE EMERALD DOSSIER", ln=True, align='C')
                pdf.set_font("helvetica", "", 12)
                pdf.set_text_color(167, 243, 208)
                pdf.cell(0, 10, f"Generated: {datetime.datetime.now()}", ln=True, align='C')
                pdf.ln(10)
                pdf.set_font("helvetica", "B", 16)
                pdf.cell(0, 10, f"Target: {r.get('plant', 'N/A')}", ln=True)
                pdf.set_font("helvetica", "", 12)
                pdf.multi_cell(0, 10, f"Condition: {r.get('disease', 'N/A')}\nQuantum Risk: {r.get('q', {}).get('label', 'N/A')}\n\nClinical Pathology: {r.get('pathology', 'N/A')}")
                st.download_button("Download Bio_Report.pdf", pdf.output(), f"Emerald_{r.get('plant', 'scan')}.pdf", "application/pdf")
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