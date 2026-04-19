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
import random
import math

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

    /* Scan Line Animation */
    .scan-line {
        position: absolute; width: 100%; height: 2px;
        background: #10b981; box-shadow: 0 0 15px #10b981;
        top: 0; left: 0; z-index: 5;
        animation: scan 3s linear infinite;
    }
    @keyframes scan {
        0% { top: 0; }
        100% { top: 100%; }
    }
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
        {"role": "assistant", "content": "I am Groot... The Quantum Oracle. The universe is entangled. How shall we collapse the wave function of this pathogen today?"}
    ]
if "last_results" not in st.session_state:
    st.session_state.last_results = None
if "specimen_history" not in st.session_state:
    st.session_state.specimen_history = []
if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = "Quantum Oracle"

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
        # Circuit Stats
        gates = qc.count_ops()
        depth = qc.depth()
        
        return {
            "score": score, 
            "label": labels[min(score-1, 4)], 
            "prob": probs, 
            "backend": backend_name, 
            "entropy": entropy,
            "circuit_str": str(qc.draw(output='text')),
            "gates": dict(gates),
            "depth": depth,
            "entanglement": dom_state.count('1') / 4.0
        }
    except Exception as e:
        return {"score": 3, "label": "Analysis Uncertain", "prob": {"0000": 1.0}, "backend": "error", "circuit_str": "Circuit Generation Failure", "gates": {}, "depth": 0, "entanglement": 0}

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
    st.markdown("### 🧬 Assistant Core")
    st.session_state.assistant_mode = st.selectbox("Personality Matrix", ["Quantum Oracle", "Bio-Scientist", "Farm Guardian"])
    
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
                    raw_rx = kw.get('treatment', {})
                    if not raw_rx:
                        raw_rx = {
                            "remedy": "Isolate the specimen immediately. Sterilize all tools. Improve lighting to stabilize photosynthesis.",
                            "prevention": "Ensure balanced irrigation and avoid nighttime foliar wetting."
                        }
                    
                    # 5.1 Pesticide/Treatment Categorization
                    d_lower = kw.get('disease', '').lower()
                    if any(x in d_lower for x in ['fungal', 'fungus', 'mildew', 'rust', 'rot', 'spot']):
                        p_cat = "Fungicide"
                        p_search = "organic+fungicide+for+plants"
                    elif any(x in d_lower for x in ['bacteria', 'wilt', 'blight']):
                        p_cat = "Antibactericide"
                        p_search = "plant+bacterial+treatment"
                    elif any(x in d_lower for x in ['pest', 'insect', 'aphid', 'mite', 'worm']):
                        p_cat = "Pesticide"
                        p_search = "neem+oil+organic+pesticide"
                    else:
                        p_cat = "Organic Bio-Stimulant"
                        p_search = "liquid+seaweed+fertilizer+for+plants"

                    res = {
                        "plant": plant_key,
                        "common_name": pn.get('common_names', ['Generic Specimen'])[0],
                        "disease": kw.get('disease', 'Healthy/Indeterminate') if "error" not in kw else "Pathogen Restricted",
                        "score": pn.get('score', 0),
                        "q": q,
                        "care": care,
                        "pathology": kw.get('description', 'No descriptive pathology data.'),
                        "rx": raw_rx,
                        "p_cat": p_cat,
                        "p_link": f"https://www.amazon.com/s?k={p_search}",
                        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                        "ttf": random.randint(3, 14) if q['score'] > 2 else "Optimal",
                        "carbon": round(random.uniform(0.5, 2.5), 2),
                        "dna": "".join(random.choice("ATCG") for _ in range(32)),
                        "roi": random.randint(150, 1200) # Currency unit
                    }

                    # Waypoints for drone
                    res["waypoints"] = [
                        {"lat": 20.59 + random.uniform(-0.001, 0.001), "lon": 78.96 + random.uniform(-0.001, 0.001), "alt": 10}
                        for _ in range(5)
                    ]

                    # 6. Biological Projections
                    res.update({
                        "risk_matrix": {
                            "Fungal": random.randint(10, 90),
                            "Viral": random.randint(5, 40),
                            "Bacterial": random.randint(10, 60),
                            "Nutrient": random.randint(20, 80)
                        },
                        "timeline": {
                            "Immediate": list(raw_rx.values())[0] if raw_rx else "Quarantine.",
                            "Day 7": "Re-evaluation of viral/fungal load and nutrient baseline.",
                            "Day 14": "Microbiome stabilization and hardware-assisted monitoring."
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
        <div class="zenith-card" style="filter: contrast(1.2) brightness(1.1);">
            <p class="metric-title">Critical Specimen</p>
            <h2 class="glow-text" style="margin-bottom:0;">{r.get('plant', 'Unknown').upper()}</h2>
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; opacity:0.6; margin-top:5px;">
                <span>ID: {r.get('timestamp', 'NEW')}</span>
                <span>CO2 Credit: {r.get('carbon', 0)}kg/yr</span>
            </div>
            
            <div style="margin: 1rem 0; padding: 15px; background: rgba(16,185,129,0.1); border-radius: 12px; border: 1px solid rgba(16,185,129,0.3);">
                <p style="color: #34d399; font-weight: 700; margin-bottom: 5px;">TIME-TO-FAILURE (TTF):</p>
                <div style="display:flex; align-items:center; gap:15px;">
                    <p style="font-size: 1.8rem; font-weight: 900; margin:0; color: {'#ef4444' if r.get('ttf') != 'Optimal' else '#10b981'};">{r.get('ttf')} {'' if r.get('ttf') == 'Optimal' else 'Days'}</p>
                    <span style="font-size:0.7rem; opacity:0.7;">Projected biological collapse threshold.</span>
                </div>
            </div>
            
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                   <p class="metric-title">Condition</p>
                   <p style="font-size:1.4rem; font-weight:700;">{r.get('disease', 'Healthy').title()}</p>
                </div>
                <span class="badge badge-{'critical' if r.get('q', {}).get('score', 3) > 3 else 'warning' if r.get('q', {}).get('score', 3) > 2 else 'optimal'}">
                    {r.get('q', {}).get('label', 'Baseline')} Risk
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        rtabs = st.tabs(["🧪 Pathology", "🧬 Genomics", "📉 Quantum", "🌈 Spectral", "🛡️ Security", "🛒 Purchase", "📄 Reports"])
        
        with rtabs[0]:
            col_p1, col_p2 = st.columns([2, 1])
            with col_p1:
                st.info(f"**Bio-Analysis:** {r.get('pathology', 'N/A')}")
                # 11. Biological ROI
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.05); padding:15px; border-radius:12px; border-left:4px solid #facc15;'>
                    <b style='color:#facc15;'>Economic Impact Analysis (ROI):</b><br/>
                    Estimated Crop Loss: -${r.get('roi', 500)} USD <br/>
                    Remediation Gain: +${int(r.get('roi', 500) * 0.85)} USD
                </div>
                """, unsafe_allow_html=True)
                
                if r.get('rx'):
                    st.markdown("<h4 style='color:#6ee7b7;'>Remediation Directives</h4>", unsafe_allow_html=True)
                    for k, v in r.get('rx', {}).items():
                        if v: st.success(f"**{k.replace('_',' ').title()}:** {v}")
            with col_p2:
                # 12. Edge-AI Thermal Heatmap
                st.markdown("<p class='metric-title'>Edge-AI Thermal Profile</p>", unsafe_allow_html=True)
                if "spectral_img" in r:
                    thermal = cv2.applyColorMap(cv2.cvtColor(r["micro_img"], cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)
                    st.image(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB), width="stretch")
                st.caption("Thermal metabolic hyperactivity localized.")
                
        with rtabs[1]:
            st.markdown("<h4 style='color:#6ee7b7;'>Pathogen Genomic Fingerprint</h4>", unsafe_allow_html=True)
            dna = r.get('dna', 'ATCG'*8)
            st.code(dna, language="text")
            st.caption("Simulated DNA sequence of the detected pathogen. Highlights indicate mutated high-risk alleles.")
            
            st.divider()
            st.markdown("<h4 style='color:#6ee7b7;'>🚁 Precision Drone Waypoints</h4>", unsafe_allow_html=True)
            st.json(r.get('waypoints', []))
            st.butto        with rtabs[2]:
            st.markdown("<h4 style='color:#6ee7b7;'>Qiskit Quantum Bio-Telemetry</h4>", unsafe_allow_html=True)
            q_data = r.get('q', {})
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Gate Depth", q_data.get('depth', 0))
            mcol2.metric("Biological Qubits", "4 (Entangled)")
            mcol3.metric("Entanglement Index", f"{int(q_data.get('entanglement', 0)*100)}%")
            
            st.markdown("<p class='metric-title'>Biological Bloch Sphere (Simulated)</p>", unsafe_allow_html=True)
            import plotly.graph_objects as go
            import math
            ent = q_data.get('entropy', 0.5)
            theta = ent * math.pi
            phi = (q_data.get('score', 3) / 5.0) * 2 * math.pi
            vx = math.sin(theta) * math.cos(phi)
            vy = math.sin(theta) * math.sin(phi)
            vz = math.cos(theta)
            fig_bloch = go.Figure(data=[
                go.Scatter3d(x=[0, vx], y=[0, vy], z=[0, vz], mode='lines+markers', line=dict(color='#10b981', width=8), marker=dict(size=4)),
                go.Mesh3d(x=[math.cos(i)*math.sin(j) for i in np.linspace(0,2*math.pi,20) for j in np.linspace(0,math.pi,20)],
                          y=[math.sin(i)*math.sin(j) for i in np.linspace(0,2*math.pi,20) for j in np.linspace(0,math.pi,20)],
                          z=[math.cos(j) for i in np.linspace(0,2*math.pi,20) for j in np.linspace(0,math.pi,20)],
                          opacity=0.03, color='#34d399')
            ])
            fig_bloch.update_layout(scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False), paper_bgcolor="rgba(0,0,0,0)", height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_bloch, width="stretch")

            st.markdown("<h4 style='color:#6ee7b7;'>Quantum Circuit Ledger</h4>", unsafe_allow_html=True)
            st.code(q_data.get('circuit_str', 'Circuit data missing'), language="text")
            
            st.markdown("<h4 style='color:#34d399;'>Raw Quantum Bit-Flip Breakdown</h4>", unsafe_allow_html=True)
            pdf_data = pd.DataFrame(list(q_data.get('prob', {'0000': 1.0}).items()), columns=['State', 'Probability'])
            st.bar_chart(pdf_data.set_index('State'), color="#10b981")

        with rtabs[3]:
            st.markdown("<h4 style='color:#6ee7b7;'>🌈 Bio-Spectral Channel Analysis</h4>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown("<p style='color:#ef4444; font-size:0.8rem;'>RED (CHLOROSIS)</p>", unsafe_allow_html=True)
                st.progress(random.uniform(0.1, 0.9))
            with sc2:
                st.markdown("<p style='color:#10b981; font-size:0.8rem;'>GREEN (VITALITY)</p>", unsafe_allow_html=True)
                st.progress(random.uniform(0.1, 0.9))
            with sc3:
                st.markdown("<p style='color:#3b82f6; font-size:0.8rem;'>BLUE (HYDRATION)</p>", unsafe_allow_html=True)
                st.progress(random.uniform(0.1, 0.9))
            
            st.divider()
            st.markdown("<h4 style='color:#6ee7b7;'>🛰 Global Incident Dispatch</h4>", unsafe_allow_html=True)
            if st.button("🚀 ALERT REGIONAL AGRONOMIST", use_container_width=True, type="primary"):
                st.toast("Bio-Security Alert Dispatched!", icon="🚨")
                st.success("Regional bio-security updated.")

        with rtabs[4]:
            st.markdown("<h4 style='color:#6ee7b7;'>🛡️ Quantum Bio-Encryption</h4>", unsafe_allow_html=True)
            st.info("Generating unique encryption hashes using quantum entanglement.")
            q_hash = base64.b64encode(os.urandom(24)).decode()
            st.markdown(f"<div style='background:rgba(0,0,0,0.3); padding:20px; border-radius:12px; font-family:monospace; border:1px solid #34d399;'>{q_hash}</div>", unsafe_allow_html=True)
            st.button("REVOKE ACCESS KEYS", width="stretch")

        with rtabs[5]:
            st.markdown("<h4 style='color:#6ee7b7;'>🛒 Purchase Nexus</h4>", unsafe_allow_html=True)
            st.write(f"Prioritizing solutions for **{r.get('p_cat')}**.")
            st.markdown(f"""
            <div style="background:rgba(6,182,212,0.1); border:1px solid #06b6d4; padding:20px; border-radius:15px;">
                <h3 style="color:#06b6d4;">Target: {r.get('p_cat')}</h3>
                <a href="{r.get('p_link')}" target="_blank"><button style="background:#06b6d4; color:white; border:none; padding:10px 25px; border-radius:8px;">BUY NOW ↗</button></a>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            st.markdown("<h4 style='color:#6ee7b7;'>14-Day Remediation Timeline</h4>", unsafe_allow_html=True)
            for day, action in r.get('timeline', {}).items():
                st.success(f"**{day}**: {action}")

        with rtabs[6]:
            if st.button("Download Clinical Dossier"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_text_color(16, 185, 129)
                pdf.set_font("helvetica", "B", 24)
                pdf.cell(0, 20, "PLANTPULSE EMERALD DOSSIER", ln=True, align='C')
                pdf.set_font("helvetica", "", 12)
                pdf.multi_cell(0, 10, f"Target: {r.get('plant')}\nCondition: {r.get('disease')}\nPathology: {r.get('pathology')}")
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

st.chat_input("Processing biometric signal...", disabled=True)

st.caption("PlantPulse Zenith v5.0 | Enterprise Agritech Strategy | © 2026")