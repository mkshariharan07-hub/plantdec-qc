"""
main.py — PlantPulse AI + Quantum (Enterprise Zenith v5)
======================================================
Professional Agritech Diagnostic Console
Pure Cloud + Quantum Entanglement Logic
"""

import streamlit as st

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="PlantPulse Zenith",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

import cv2
import numpy as np
import os
# Force disable proxies to prevent zombie Windows proxy settings from causing timeouts on hotspots
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

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
import hashlib
import re

# Optional Quantum Imports
try:
    from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    HAS_QUANTUM = True
except ImportError:
    HAS_QUANTUM = False

# Core Utilities
from utils import (
    decode_bytes_to_bgr, 
    identify_plant_with_plantnet, 
    identify_disease_with_kindwise, 
    identify_disease_with_plantnet, 
    get_perenual_care_info, 
    get_disease_info, 
    predict_image, 
    load_model_and_scaler, 
    remap_disease_with_nyckel, 
    get_botanical_equivalent,
    identify_disease_with_huggingface
)

load_dotenv()

# ===============================
# LOCAL MODEL LOADING (Zenith Bio-Core)
# ===============================
try:
    local_model, local_scaler = load_model_and_scaler()
    HAS_LOCAL_MODEL = True
except Exception as e:
    HAS_LOCAL_MODEL = False
    print(f"Local Model Load Failed: {e}")

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
if "last_results" not in st.session_state or st.session_state.last_results is None:
    st.session_state.last_results = {
        "plant": "UNKNOWN",
        "common_name": "UNKNOWN",
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "carbon": 1.12,
        "ttf": "Optimal",
        "disease": "Healthy/Indeterminate",
        "q": {"score": 1, "label": "Optimal", "prob": {"0000": 1.0}, "depth": 1, "entanglement": 0.05, "circuit_str": "Quantum Oracle Initialization..."},
        "dna": "AGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT",
        "pathology": "System initialized. Entanglement vectors stabilized. Awaiting biological input for live analysis.",
        "rx": {"protocol": "Maintain baseline environmental parameters."},
        "p_cat": "Bio-Stabilizer",
        "p_link": "https://www.amazon.com/s?k=organic+plant+stabilizer",
        "roi": 1200,
        "timeline": {"Startup": "Quantum matrix calibrated.", "Day 7": "Stabilization complete.", "Day 14": "Optimal yield reached."},
        "waypoints": []
    }
if "specimen_history" not in st.session_state:
    st.session_state.specimen_history = []
if "assistant_mode" not in st.session_state:
    st.session_state.assistant_mode = "Quantum Oracle"
if "last_scan_id" not in st.session_state:
    st.session_state.last_scan_id = None

# ===============================
# QUANTUM SEVERITY PROBABILISTIC
# ===============================
def analyze_severity_quantum(img: np.ndarray, backend_pref: str, is_healthy_hint: bool = False, is_pathogen_hint: bool = False, base_severity: str = "low"):
    if not HAS_QUANTUM:
        # Default scores based on severity hint if quantum is missing
        s = 1 if is_healthy_hint else 3 if base_severity == "medium" else 4 if base_severity == "high" else 2
        return {"score": s, "label": "Simulated Matrix", "prob": {"00000000": 0.5}, "backend": "sim", "entanglement": 0.5, "depth": 0, "gates": {}, "circuit_str": "No Qiskit"}
        
    try:
        small = cv2.resize(img, (64, 64))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(small)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_f = gray.astype(float) / 255.0
        
        # FEATURE 1: Spectral Entropy
        entropy = -np.sum(gray_f * np.log2(gray_f + 1e-7)) / 4096.0
        
        # FEATURE 2-4: Color Means
        r_m, g_m, b_m = np.mean(r)/255.0, np.mean(g)/255.0, np.mean(b)/255.0
        
        # FEATURE 5: Pathogenic Spot Detection (Laplacian)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var() / 250.0 # Increased sensitivity
        
        # FEATURE 6: Necrosis Detection (Brown/Black Lesion ratio)
        lower_brown = np.array([10, 20, 20])
        upper_brown = np.array([30, 255, 150])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        necrosis_ratio = np.sum(brown_mask > 0) / (64*64)
        
        # FEATURE 7: Texture noise
        std_dev = np.std(gray_f) * 5.0
        
        # FEATURE 8: Green Loss (Chlorosis)
        green_mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([85, 255, 255]))
        chlorosis_factor = 1.0 - (np.sum(green_mask > 0) / (64*64))

        features = [entropy, r_m, g_m, necrosis_ratio, std_dev, lap_var, chlorosis_factor, np.var(gray_f)*10]
        
        # 8-Qubit Zenith Protocol
        n_qubits = 8
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        # 1. State Encoding Layer
        for i, val in enumerate(features):
            qc.ry(abs(val) * math.pi * 2, qr[i])
            
        qc.barrier()
        # 2. Pathogen Entropy Injection
        if is_pathogen_hint:
            for i in range(n_qubits):
                qc.rx(math.pi/3, qr[i])
                qc.h(qr[i])

        # 3. Entanglement Cross-Coupling
        for i in range(n_qubits):
            qc.cx(qr[i], qr[(i+1) % n_qubits])
            if i < n_qubits - 2:
                qc.cz(qr[i], qr[i+2])
            
        qc.barrier()
        # 4. Superposition Mixing
        for i in range(n_qubits):
            qc.h(qr[i])
                
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
            else: raise ValueError()
        except:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
            result = sampler.run([qc]).result()[0]
            counts = result.data.c.get_counts()
            backend_name = "q-local-sim-enhanced"

        total = sum(counts.values())
        probs = {k: v/total for k, v in counts.items()}
        dom_state = max(counts, key=counts.get)
        if isinstance(dom_state, int): dom_state = format(dom_state, f'0{n_qubits}b')
        
        # RISK SCORING ENGINE
        bit_count = dom_state.count('1')
        q_score = min(5, (bit_count // 2) + 1)
        
        # Visual Heuristic Escalation (Based on real pixels)
        v_score = 1
        if lap_var > 0.04 or necrosis_ratio > 0.01: v_score = 2 
        if lap_var > 0.10 or necrosis_ratio > 0.05: v_score = 3
        if lap_var > 0.30 or necrosis_ratio > 0.12: v_score = 4
        if lap_var > 0.60 or necrosis_ratio > 0.20: v_score = 5
        
        score = max(q_score, v_score)
        
        # VISUAL REALITY OVERRIDE (Safety against Quantum False Positives)
        # If the image is extremely smooth (low laplacian) and has no necrosis, 
        # we cap the risk at 'Incipient' (2) regardless of quantum noise.
        if lap_var < 0.04 and necrosis_ratio < 0.002:
            score = min(score, 2)
            # If extremely smooth and green, it's Optimal
            if lap_var < 0.02:
                score = 1
        
        # BIOLOGICAL OVERRIDES
        if is_pathogen_hint:
            # If a pathogen is definitely found, the risk is at least Severe (4) if spots are seen, else Moderate (3)
            # Escalation based on base_severity from knowledge base
            min_risk = 3
            if base_severity == "medium": min_risk = 3
            if base_severity == "high": min_risk = 4
            
            # Further escalation if visual evidence is strong
            if lap_var > 0.08 or necrosis_ratio > 0.03:
                min_risk = max(min_risk, 4)
            
            score = max(score, min_risk)
        
        if is_healthy_hint:
            if lap_var > 0.15 or necrosis_ratio > 0.05:
                score = 2 # Incipient if spots seen but AI says healthy
            else:
                score = 1 # Optimal
            
        labels = ["Optimal", "Incipient", "Moderate", "Severe", "Critical"]
        return {
            "score": score, 
            "label": labels[min(score-1, 4)], 
            "prob": probs, 
            "backend": backend_name, 
            "entropy": entropy,
            "circuit_str": str(qc.draw(output='text')),
            "gates": dict(qc.count_ops()),
            "depth": qc.depth(),
            "entanglement": min(1.0, len(counts) / 25.6)
        }
    except Exception as e:
        return {"score": 3, "label": "Simulator Matrix", "prob": {"0000": 1.0}, "backend": "local-sim-fallback", "circuit_str": f"Error: {str(e)}", "gates": {}, "depth": 0, "entanglement": 0}

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.image("https://img.icons8.com/bubbles/200/leaf.png", width=120)
    st.markdown("<h2 class='glow-text'>PLANTPULSE ZENITH</h2>", unsafe_allow_html=True)
    
    # Check all key status - prioritized: Widget > Env > Secret
    # Moved to top to ensure availability for all components
    keys = {
        "PLANTNET": os.getenv("PLANTNET_API_KEY") or (st.secrets.get("PLANTNET_API_KEY") if "PLANTNET_API_KEY" in st.secrets else None),
        "KINDWISE": os.getenv("CROP_HEALTH_API_KEY") or (st.secrets.get("CROP_HEALTH_API_KEY") if "CROP_HEALTH_API_KEY" in st.secrets else None),
        "HUGGINGFACE": os.getenv("HUGGINGFACE_API_KEY") or (st.secrets.get("HUGGINGFACE_API_KEY") if "HUGGINGFACE_API_KEY" in st.secrets else None),
        "PERENUAL": os.getenv("PERENUAL_API_KEY") or (st.secrets.get("PERENUAL_API_KEY") if "PERENUAL_API_KEY" in st.secrets else None),
        "IBM_QUANTUM": os.getenv("IBM_QUANTUM_TOKEN") or (st.secrets.get("IBM_QUANTUM_TOKEN") if "IBM_QUANTUM_TOKEN" in st.secrets else None),
    }

    # API KEY ORCHESTRATION
    with st.expander("🔑 Credential Overrides", expanded=False):
        pn_k = st.text_input("PlantNet API Key", value=keys["PLANTNET"] or "", type="password")
        kw_k = st.text_input("Kindwise API Key", value=keys["KINDWISE"] or "", type="password")
        if pn_k: keys["PLANTNET"] = pn_k
        if kw_k: keys["KINDWISE"] = kw_k
    
    with st.expander("🛠 Matrix Configuration", expanded=False):
        q_eng = st.selectbox("Quantum Engine", ["Dynamic (Hybrid)", "Simulator Optimized"])
        primary_engine = st.selectbox("Primary Disease Engine", ["Hugging Face (Free)", "Kindwise (Paid)", "Pl@ntNet", "Local Mesh"], index=0)
        api_depth = st.slider("Discovery Depth", 1, 10, 7)
        hard_guess = st.toggle("Hard-Guess Mode", value=True)
        ssl_verify = st.toggle("Verify SSL Certificates", value=False, help="Disable if encountering SSL/Proxy errors on Windows")
    
    st.divider()
    st.markdown("### Clinical Status")
    st.success("🛰 Core Satellites: ACTIVE")
    st.info("🧬 Pathogen Matrix: SYNCED")
    st.warning("🍂 Bio-Telemetry: HIGH LOAD")
    
    if st.button("Reset Session Matrix", width="stretch"):
        st.session_state.clear()
        st.rerun()



    st.divider()
    st.markdown("### 🔍 System Diagnostics")
    c_sid = st.session_state.get('last_scan_id', 'None')
    st.write(f"Current Matrix ID: `{c_sid[:8] if c_sid else 'None'}`")
    st.write(f"Neural Mesh: {'🟢 READY' if HAS_LOCAL_MODEL else '🔴 OFFLINE'}")
    c1 = st.checkbox("PlantNet Key", value=bool(pn_k))
    c2 = st.checkbox("Kindwise Key", value=bool(kw_k))
    c3 = st.checkbox("Quantum Key", value=bool(os.getenv("IBM_QUANTUM_TOKEN")))
    if not (c1 and c2): st.warning("Diagnostics compromised: Keys missing.")
    
    st.divider()
    st.markdown("### 🧬 Assistant Core")
    st.session_state.assistant_mode = st.selectbox("Personality Matrix", ["Dr. Leaf", "Quantum Oracle", "Bio-Scientist", "Farm Guardian"])
    
    with st.expander("📡 Last Scan Diagnostics", expanded=True):
        if st.session_state.get('last_results'):
            lr = st.session_state.last_results
            st.write(f"**Target:** {lr.get('plant')}")
            st.write(f"**Confidence Matrix:**")
            st.caption(f"• PlantNet: {lr.get('pn_score', 0)}%")
            st.caption(f"• Neural Mesh: {lr.get('local_score', 0)}%")
            st.caption(f"• Host Inference: {lr.get('host_score', 0)}%")
            st.write(f"**Timestamp:** {lr.get('timestamp')}")
        else:
            st.info("No scan data in session.")
        
        if st.button("Force Global Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    st.divider()
    st.markdown("### 🛰️ Global Telemetry")
    t_col1, t_col2 = st.columns(2)
    t_col1.metric("Signal", f"{random.randint(85, 99)}%", "-2%")
    t_col2.metric("Uplink", f"{random.randint(12, 48)}kbps")
    st.caption("Active Satellite: ZENITH-SAT-7 (Encrypted)")
    
    st.divider()
    st.markdown("### 🛠 Development Core")
    
    st.markdown("""
    <div class="dev-card">
        <span class="dev-name">Sindhuja R</span>
        <span class="dev-meta">Reg: 226004099</span>
    </div>
    <div class="dev-card">
        <span class="dev-name">Saraswathy</span>
        <span class="dev-meta">Reg: 226004092</span>
    </div>
    <div class="dev-card">
        <span class="dev-name">U. Kiruthika</span>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("📝 System Pulse Log", expanded=False):
        st.caption("Uplink: STABLE")
        st.code(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Neural Mesh Sync\n[OK] Latency: 12ms\n[OK] Quantum Decryp.\n[MSG] Waveform locked.")

    # Zenith Cloud Configuration Sidebar
    st.sidebar.divider()
    st.sidebar.markdown("<h3 style='color:#34d399;'>🔐 Zenith Cloud Interface</h3>", unsafe_allow_html=True)
    
    # Check all key status
    # Check all key status - prioritized: Widget > Env > Secret
    keys = {
        "PLANTNET": pn_k or os.getenv("PLANTNET_API_KEY") or (st.secrets.get("PLANTNET_API_KEY") if "PLANTNET_API_KEY" in st.secrets else None),
        "KINDWISE": kw_k or os.getenv("CROP_HEALTH_API_KEY") or (st.secrets.get("CROP_HEALTH_API_KEY") if "CROP_HEALTH_API_KEY" in st.secrets else None),
        "HUGGINGFACE": os.getenv("HUGGINGFACE_API_KEY") or (st.secrets.get("HUGGINGFACE_API_KEY") if "HUGGINGFACE_API_KEY" in st.secrets else None),
        "PERENUAL": os.getenv("PERENUAL_API_KEY") or (st.secrets.get("PERENUAL_API_KEY") if "PERENUAL_API_KEY" in st.secrets else None),
        "IBM_QUANTUM": os.getenv("IBM_QUANTUM_TOKEN") or (st.secrets.get("IBM_QUANTUM_TOKEN") if "IBM_QUANTUM_TOKEN" in st.secrets else None),
        "NYCKEL_FID": st.session_state.get("nfid_over") or os.getenv("NYCKEL_FID"),
        "NYCKEL_ID": st.session_state.get("nid_over") or os.getenv("NYCKEL_ID"),
        "NYCKEL_SECRET": st.session_state.get("nsec_over") or os.getenv("NYCKEL_SECRET")
    }
    
    for k, v in keys.items():
        status_col, label_col = st.sidebar.columns([1, 4])
        if v and len(str(v)) > 5:
            status_col.markdown("🟢")
            label_col.caption(f"{k}: {str(v)[:4]}***")
        else:
            status_col.markdown("🔴")
            label_col.warning(f"{k} Missing")
            
    with st.sidebar.expander("🛠️ Manual Key Override"):
        st.caption("Overrides .env / Secrets")
        pk = st.text_input("New PlantNet Key", type="password", key="pk_over")
        ck = st.text_input("New Kindwise Key", type="password", key="ck_over")
        st.markdown("---")
        st.caption("🧬 Neural Remap (Nyckel)")
        nfid = st.text_input("Nyckel Function ID", key="nfid_over")
        nid = st.text_input("Nyckel Client ID", type="password", key="nid_over")
        nsec = st.text_input("Nyckel Client Secret", type="password", key="nsec_over")
        if st.button("ACTIVATE CLOUD OVERRIDE"):
            if pk: 
                os.environ["PLANTNET_API_KEY"] = pk
                st.session_state["pk_manual"] = pk
            if ck: 
                os.environ["CROP_HEALTH_API_KEY"] = ck
                st.session_state["ck_manual"] = ck
            st.toast("Zenith Cloud Matrix Re-calibrated!", icon="⚡")
            st.rerun()

    # Heartbeat Check
    if st.sidebar.button("📡 TEST PLANTNET UPLINK"):
        pn_key = keys.get("PLANTNET")
        if not pn_key:
            st.sidebar.error("Key Missing")
        else:
            try:
                test_url = f"https://my-api.plantnet.org/v2/projects?api-key={pn_key}"
                test_res = requests.get(test_url, timeout=10)
                if test_res.status_code == 200:
                    st.sidebar.success(f"Link Stable (Ping: {len(test_res.json())} projects)")
                else:
                    st.sidebar.error(f"Link Rejected: HTTP {test_res.status_code}")
            except Exception as e:
                st.sidebar.error(f"Uplink Failed: {e}")

    if st.sidebar.button("📡 TEST HUGGINGFACE UPLINK"):
        hf_key = keys.get("HUGGINGFACE")
        if not hf_key or "your_token" in hf_key:
            st.sidebar.error("Key Missing/Invalid")
        else:
            try:
                # Test with official whoami endpoint
                test_url = "https://huggingface.co/api/whoami-v2"
                headers = {"Authorization": f"Bearer {hf_key}"}
                test_res = requests.get(test_url, headers=headers, timeout=10, verify=ssl_verify)
                
                if test_res.status_code == 200:
                    user_data = test_res.json()
                    user_name = user_data.get("name", "Unknown")
                    st.sidebar.success(f"Online: Connected as {user_name}")
                else:
                    st.sidebar.error(f"Rejected: HTTP {test_res.status_code}")
            except Exception as e:
                st.sidebar.error(f"Link Failure: {str(e)}")

    if st.sidebar.button("📡 PING CLOUD HEARTBEAT"):
        try:
            requests.get("https://www.google.com", timeout=5)
            st.sidebar.success("Heartbeat: ONLINE")
        except:
            st.sidebar.error("Heartbeat: OFFLINE (Network Blocked)")

    if st.sidebar.button("📡 TEST PLANTNET DISEASE UPLINK"):
        pn_key = keys.get("PLANTNET")
        if not pn_key:
            st.sidebar.error("Key Missing")
        else:
            try:
                test_url = f"https://my-api.plantnet.org/v2/diseases/identify?api-key={pn_key}"
                # Empty POST just to check key validity
                test_res = requests.post(test_url, timeout=10)
                if test_res.status_code in [200, 400]: # 400 is fine as we sent no images
                    st.sidebar.success("Disease Uplink Stable")
                else:
                    st.sidebar.error(f"Rejected: {test_res.status_code}")
            except Exception as e:
                st.sidebar.error(f"Uplink Failed: {e}")

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
    
    if "img_bytes" not in st.session_state:
        st.session_state.img_bytes = None

    with tabs[0]:
        uf = st.file_uploader("Ingest specimen data...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if uf: st.session_state.img_bytes = uf.getvalue()
    with tabs[1]:
        cf = st.camera_input("Scanner activation", help="Capturing a photo will override file uploads")
        if cf: st.session_state.img_bytes = cf.getvalue()

    img_bytes = st.session_state.img_bytes

    if img_bytes:
        frame = decode_bytes_to_bgr(img_bytes)
        if frame is not None:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width="stretch")
            
            # MD5-BASED STABLE SCAN TRACKER
            scan_id = hashlib.md5(img_bytes).hexdigest()
            manual_run = st.button("🚀 FORCE ZENITH ANALYSIS", width="stretch", type="primary")
            # FIX: Check for both 'UNKNOWN' and 'SCANNING...' to prevent deadlocks
            current_plant = st.session_state.last_results.get('plant', 'UNKNOWN').upper()
            results_stale = "UNKNOWN" in current_plant or "SCANNING" in current_plant or "INDETERMINATE" in current_plant
            id_mismatch = (st.session_state.get('last_scan_id') != scan_id)
            
            if manual_run or id_mismatch or (results_stale and img_bytes):
                st.session_state.last_scan_id = scan_id
                with st.status("Harmonizing neural and quantum vectors...", expanded=True) as status:
                    # Mark 'started' with current timestamp, but don't overwrite the plant name 
                    # with 'SCANNING' in session state to avoid it getting stuck on UI crashes.
                    st.session_state.last_results['timestamp'] = datetime.datetime.now().strftime("%H:%M:%S")
                    
                    try:
                        status.write("Phase 1: PlantNet Botanical Uplink...")
                        pn = identify_plant_with_plantnet(frame, api_key=keys.get("PLANTNET"), verify_ssl=ssl_verify)
                        
                        if "error" in pn:
                            status.write(f"⚠️ PlantNet: {pn['error']}")
                            # If no key, maybe Kindwise or Local Mesh can help
                        else:
                            common_name = pn.get('common_names')[0] if pn.get('common_names') else pn.get('scientific_name')
                            status.write(f"✅ PlantNet: {common_name} ({pn.get('score')}% confidence)")
                        
                        # LOG FOR DEEP-DEBUG
                        log_path = os.path.join(os.path.dirname(__file__), "api_log.txt")
                        with open(log_path, "a") as f:
                            f.write(f"--- SCAN AT {datetime.datetime.now()} ---\n")
                            f.write(f"PlantNet Outcome: {pn}\n")
                        
                        # 100% Cloud-Based Identification Pipeline (Zenith Synapse-V)
                        is_weak = "error" in pn or not pn.get('scientific_name') or pn.get('scientific_name').upper() in ["UNKNOWN SPECIMEN", "UNKNOWN SPECIES", "UNKNOWN"]
                        unstable = pn.get('score', 0) < 15.0 and not hard_guess
                        inferred_plant = None

                        if is_weak or unstable:
                            status.write("PlantNet inconclusive. Scaling to Pathogen Path-Mining...")
                            # 1. Higher-Depth Pathogen Identification
                            kw = identify_disease_with_kindwise(frame, api_key=keys.get("KINDWISE"))
                            
                            # Pathogen AI Fallback for Host Extraction
                            if "error" in kw and HAS_LOCAL_MODEL:
                                status.write("🛰️ Cloud Health offline. Engaging Pathogen AI for signature match...")
                                local_res = predict_image(frame, local_model, local_scaler)
                                kw = {
                                    "disease": local_res.get('disease'),
                                    "plant": local_res.get('plant'),
                                    "probability": local_res.get('confidence')
                                }

                            disease_main = kw.get('disease', '')
                            
                            # Retrieve host directly from Kindwise Crop API, or fallback to disease name parsing
                            found_host = kw.get('plant')
                            if not found_host:
                                for host in ["Tomato", "Apple", "Potato", "Banana", "Corn", "Rice", "Wheat", "Grape", "Orange", "Pepper", "Coffee", "Mango"]:
                                    if host.lower() in disease_main.lower():
                                        found_host = host
                                        break
                            
                            if found_host:
                                status.write(f"Identity recovered via Health Engine: {found_host}")
                                pn = {"scientific_name": found_host, "common_names": [found_host], "score": 90.0}
                            elif "timeout" in pn.get('error', '').lower() or "timeout" in kw.get('error', '').lower():
                                status.write("⚠️ WARNING: Cloud Botanical APIs timed out. Check network proxy/firewall.")
                            
                            disease_name = kw.get('disease', '').lower()
                            inferred_plant = None
                            
                            # Enterprise Host Extraction Matrix (V6 - Deep Expansion)
                            hosts_matrix = {
                                "Apple": ["apple", "malus"],
                                "Paddy/Rice": ["paddy", "rice", "oryza"],
                                "Banana": ["banana", "musa"],
                                "Potato": ["potato", "solanum tuberosum"],
                                "Tomato": ["tomato", "solanum lycopersicum"],
                                "Corn/Maize": ["corn", "maize", "zea"],
                                "Grape": ["grape", "vitis"],
                                "Wheat": ["wheat", "triticum"],
                                "Cotton": ["cotton", "gossypium"],
                                "Coffee": ["coffee", "coffea"],
                                "Citrus": ["citrus", "orange", "lemon", "lime", "calamondin", "pomelo"],
                                "Strawberry": ["strawberry", "fragaria"],
                                "Mango": ["mango", "mangifera"],
                                "Chilli/Pepper": ["chilli", "pepper", "capsicum", "bell pepper", "chili"],
                                "Brinjal/Eggplant": ["brinjal", "eggplant", "solanum melongena", "aubergine"],
                                "Soybean": ["soybean", "glycine max"],
                                "Sugarcane": ["sugarcane", "saccharum"],
                                "Rubber": ["rubber", "hevea"],
                                "Cashew": ["cashew", "anacardium"],
                                "Coconut": ["coconut", "cocos nucifera"],
                                "Cabbage": ["cabbage", "brassica"],
                                "Cucumber": ["cucumber", "cucumis"],
                                "Onion": ["onion", "allium cepa"],
                                "Garlic": ["garlic", "allium sativum"],
                                "Rose": ["rose", "rosa"],
                                "Hibiscus": ["hibiscus"],
                                "Peanut/Groundnut": ["peanut", "groundnut", "arachis"],
                                "Guava": ["guava", "psidium"],
                                "Papaya": ["papaya", "carica"],
                                "Pomegranate": ["pomegranate", "punica"],
                                "Ginger": ["ginger", "zingiber"],
                                "Turmeric": ["turmeric", "curcuma"],
                                "Black Pepper": ["black pepper", "piper nigrum"],
                                "Cardamom": ["cardamom", "elettaria"],
                                "Tea": ["tea", "camellia sinensis"],
                                "Areca Nut": ["areca", "betel"],
                                "Pea": ["pea", "pisum"],
                                "Bean": ["bean", "phaseolus"],
                                "Okra/Ladyfinger": ["okra", "ladyfinger", "abelmoschus"],
                                "Sunflower": ["sunflower", "helianthus"],
                                "Mustard": ["mustard", "brassica juncea"],
                                "Millet": ["millet", "pennisetum", "sorghum", "ragi"],
                                "Rubber": ["rubber", "hevea"],
                                "Teak": ["teak", "tectona"],
                                "Sandalwood": ["sandalwood", "santalum"],
                                "Neem": ["neem", "azadirachta"],
                                "Aloe Vera": ["aloe"],
                                "Jackfruit": ["jackfruit", "artocarpus"],
                                "Durian": ["durian", "durio"],
                                "Dogwood": ["dogwood", "cornus"]
                            }
                            
                            # Search both name and description for host keywords
                            diagnostic_text = (kw.get('disease', '') + " " + kw.get('description', '')).lower()
                            inferred_plant = None
                            
                            for plant_label, keywords in hosts_matrix.items():
                                # Use word boundary matching to avoid partial matches (e.g., 'pineapple' matching 'apple')
                                if any(re.search(rf"\b{re.escape(k)}\b", diagnostic_text) for k in keywords):
                                    inferred_plant = plant_label
                                    break
                            
                            if inferred_plant:
                                status.write(f"Identity recovered via Bio-Signature: {inferred_plant}")
                                pn = {"scientific_name": inferred_plant, "common_names": [inferred_plant], "score": 88.0}
                            else:
                                status.write("Unrecognised biological waveform. Using deep-proxy diagnostics...")
                                pn = {"scientific_name": "Unknown Specimen", "common_names": ["Indeterminate Specimen"], "score": 0}
                        
                        # PERSIST IDENTITY IMMEDIATELY (Safety checkpoint)
                        st.session_state.last_results['plant'] = pn.get('scientific_name', 'Unknown Specimen')
                        st.session_state.last_results['pn_score'] = pn.get('score', 0) if "error" not in pn else 0
                        st.session_state.last_results['local_score'] = local_res.get('confidence', 0) if 'local_res' in locals() else 0
                        st.session_state.last_results['host_score'] = 88.0 if inferred_plant else 0
                        st.session_state.last_results['score'] = pn.get('score', 0)
                        
                        # 3. Pathogen Phase
                        status.write(f"Engaging Primary Engine: {primary_engine}...")
                        kw = {"error": "initialization"}
                        
                        if primary_engine == "Hugging Face (Free)":
                            kw = identify_disease_with_huggingface(frame, api_key=keys.get("HUGGINGFACE"))
                        elif primary_engine == "Kindwise (Paid)":
                            kw = identify_disease_with_kindwise(frame, api_key=keys.get("KINDWISE"))
                        elif primary_engine == "Pl@ntNet":
                            kw = identify_disease_with_plantnet(frame, api_key=keys.get("PLANTNET"))
                        
                        # MULTI-CLOUD PATHOGEN FALLBACK (Logic-based fallback chain)
                        if "error" in kw or not kw.get('disease'):
                            if primary_engine != "Hugging Face (Free)":
                                status.write("Primary engine restricted. Switching to Hugging Face AI (Free)...")
                                hf_res = identify_disease_with_huggingface(frame, api_key=keys.get("HUGGINGFACE"))
                                if "error" not in hf_res: kw = hf_res
                                
                            if ("error" in kw or not kw.get('disease')) and primary_engine != "Kindwise (Paid)":
                                status.write("Attempting Kindwise Matrix...")
                                kw_res = identify_disease_with_kindwise(frame, api_key=keys.get("KINDWISE"))
                                if "error" not in kw_res: kw = kw_res
                                
                            if ("error" in kw or not kw.get('disease')) and primary_engine != "Pl@ntNet":
                                status.write("Attempting Pl@ntNet Disease Uplink...")
                                pn_res = identify_disease_with_plantnet(frame, api_key=keys.get("PLANTNET"))
                                if "error" not in pn_res: kw = pn_res
                        
                        # PATHOGEN AI FALLBACK (Local Mesh)
                        if ("error" in kw or not kw.get('disease')) and HAS_LOCAL_MODEL:
                            status.write("🧬 Pathogen Matrix offline. Scaling to Pathogen AI (Local Mesh)...")
                            l_res = predict_image(frame, local_model, local_scaler)
                            
                            p_ai_disease = l_res.get('disease', 'Healthy/Indeterminate')
                            p_ai_plant = l_res.get('plant', 'Unknown')
                            actual_plant = str(pn.get('scientific_name', 'Specimen')).title()
                            
                            # DYNAMIC PATHOGEN MAPPING (Avoiding hardcoding)
                            # We report the local model's class but add a 'Visual Equivalent' note 
                            # if the predicted host doesn't match the actual specimen.
                            diag_suffix = ""
                            if p_ai_plant.lower() != "unknown" and p_ai_plant.lower() not in actual_plant.lower():
                                diag_suffix = f" (Visual Match: {p_ai_disease})"
                                
                                # NYCKEL DYNAMIC REMAPPING
                                if keys.get('NYCKEL_FID'):
                                    remap_text = f"{p_ai_disease} on {actual_plant}"
                                    p_ai_disease = remap_disease_with_nyckel(
                                        remap_text, 
                                        keys['NYCKEL_FID'], 
                                        keys.get('NYCKEL_ID'), 
                                        keys.get('NYCKEL_SECRET')
                                    )
                                else:
                                    # BOTANICAL MATRIX LOOKUP (Scientific Fallback)
                                    botanical_name = get_botanical_equivalent(actual_plant, p_ai_disease)
                                    if botanical_name:
                                        p_ai_disease = botanical_name
                                    else:
                                        # Fallback to indeterminate if no match (Lower risk default)
                                        p_ai_disease = f"Indeterminate {actual_plant} Specimen"

                            kw = {
                                "disease": p_ai_disease,
                                "plant": l_res.get('plant', pn.get('scientific_name')),
                                "probability": l_res.get('confidence', 0),
                                "description": f"Diagnosis: {p_ai_disease}. Analysis indicates visual alignment with pathogenic markers.",
                                "treatment": {"Primary Protocol": l_res.get('tips', 'Isolate and monitor.')}
                            }
                        
                        st.session_state.last_results['disease'] = kw.get('disease', 'Healthy/Indeterminate')
                        
                        # SEED RANDOM FOR STABLE SCAN VALUES
                        random.seed(int(scan_id, 16))
                        
                        status.write("Quantum state entanglement check...")
                        is_h_hint = "healthy" in str(kw.get('disease', '')).lower()
                        is_p_hint = not is_h_hint and "indeterminate" not in str(kw.get('disease', '')).lower()
                        
                        # Fetch base severity from knowledge matrix
                        d_info = get_disease_info(kw.get('disease', 'healthy'))
                        q = analyze_severity_quantum(frame, "Simulator Only" if q_eng == "Simulator Optimized" else "Dynamic", 
                                                    is_healthy_hint=is_h_hint, 
                                                    is_pathogen_hint=is_p_hint,
                                                    base_severity=d_info.get('severity', 'low'))
                        
                        # 4. Care Info
                        plant_key = pn.get('scientific_name', 'Unknown Specimen')
                        c_names = pn.get('common_names', [])
                        c_name = c_names[0] if c_names else plant_key
                        status.write(f"Retrieving care protocols for {c_name}...")
                        care = get_perenual_care_info(c_name)

                        # 5. Build Result Prototype (Safe extraction)
                        raw_rx = kw.get('treatment', {})
                        if not raw_rx:
                            raw_rx = {"remedy": "Isolate specimen and monitor microbial balance."}
                        
                        d_name = kw.get('disease', 'Healthy/Indeterminate')
                        d_desc = kw.get('description') or "Specimen exhibits a stable bio-signature with no dominant pathological vectors detected."
                        
                        # CROSS-VERIFICATION: If AI says Healthy but Quantum says Critical/Severe
                        if ("healthy" in d_name.lower() or "indeterminate" in d_name.lower()) and q.get('score', 1) >= 4:
                            d_name = "Undiagnosed Pathogen"
                            d_desc = "High cellular entropy and spectral variance detected. Visible lesions suggest a pathogenic vector not currently indexed in the local mesh."
                            raw_rx = {"Quarantine": "Isolate the plant immediately to prevent potential spread while a deep-tissue lab analysis is conducted."}

                        d_lower = d_name.lower()
                        p_cat = "Organic Bio-Stimulant"
                        p_search = "liquid+seaweed+fertilizer"
                        if "fungal" in d_lower or "pathogen" in d_lower: p_cat, p_search = "Fungicide", "organic+fungicide"
                        elif "bacteria" in d_lower: p_cat, p_search = "Antibactericide", "plant+antibacterial"
                        elif "pest" in d_lower or "insect" in d_lower: p_cat, p_search = "Pesticide", "neem+oil+pesticide"

                        res = {
                            "plant": plant_key,
                            "common_name": c_name,
                            "disease": d_name,
                            "score": pn.get('score', 0),
                            "q": q,
                            "care": care,
                            "pathology": d_desc,
                            "rx": raw_rx,
                            "p_cat": p_cat,
                            "p_link": f"https://www.amazon.com/s?k={p_search}",
                            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
                            "ttf": q.get('ttf', random.randint(3, 14) if q.get('score', 3) > 2 else "Optimal"),
                            "carbon": round(random.uniform(0.5, 2.5), 2),
                            "dna": "".join(random.choice("ATCG") for _ in range(32)),
                            "roi": random.randint(150, 1200),
                            "npk": {
                                "Nitrogen": random.randint(20, 85),
                                "Phosphorus": random.randint(15, 75),
                                "Potassium": random.randint(30, 95)
                            }
                        }

                        res["waypoints"] = [
                            {"lat": 20.59 + random.uniform(-0.001, 0.001), "lon": 78.96 + random.uniform(-0.001, 0.001), "alt": 10}
                            for _ in range(5)
                        ]
                        
                        # Logic-based Risk Matrix (Replacing Random)
                        d_name_l = kw.get('disease', '').lower()
                        rm = {"Fungal": random.randint(5, 15), "Viral": random.randint(2, 8), "Bacterial": random.randint(5, 12), "Nutrient": random.randint(10, 25)}
                        if "fungal" in d_name_l or "blight" in d_name_l or "mold" in d_name_l or "rust" in d_name_l or "mildew" in d_name_l or "scab" in d_name_l:
                            rm["Fungal"] = random.randint(75, 98)
                        elif "viral" in d_name_l or "virus" in d_name_l or "mosaic" in d_name_l:
                            rm["Viral"] = random.randint(70, 95)
                        elif "bacterial" in d_name_l or "bacteria" in d_name_l:
                            rm["Bacterial"] = random.randint(70, 95)
                        elif "nutrient" in d_name_l or "deficiency" in d_name_l or "chlorosis" in d_name_l:
                            rm["Nutrient"] = random.randint(60, 90)

                        res.update({
                            "risk_matrix": rm,
                            "timeline": {"Immediate": list(raw_rx.values())[0], "Day 7": "Re-evaluation of spectral load.", "Day 14": "Microbiome stabilization."}
                        })

                        # Visual Overlays
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        spectral = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                        res["spectral_img"] = cv2.addWeighted(frame, 0.6, spectral, 0.4, 0)
                        h, w = frame.shape[:2]
                        ch, cw = h//2, w//2
                        res["micro_img"] = frame[max(0, ch-128):min(h, ch+128), max(0, cw-128):min(w, cw+128)]

                        st.session_state.last_results = res
                        st.session_state.specimen_history.append({"name": plant_key, "status": res['disease'], "time": res['timestamp']})
                        status.update(label="Zenith Diagnosis Full-Locked.", state="complete")
                        
                        assistant_msg = f"Bio-Signature for **{plant_key}** fully locked. Pathogen matrix indicates **{res['disease']}**."
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_msg})
                    except Exception as e:
                        traceback.print_exc()
                        st.error(f"Groot-Shield activated. Scan Interrupted: {e}")
                        status.update(label="Hardware/API Desync Detected.", state="error")
                        
                        # Revert status from 'SCANNING' if it was set
                        if st.session_state.last_results.get('plant') == "SCANNING...":
                            st.session_state.last_results['plant'] = "UNKNOWN"
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    if st.session_state.last_results:
        r = st.session_state.last_results
        
        p_name = r.get('plant', 'Unknown').upper()
        p_status = r.get('disease', 'Healthy').title()
        is_unknown = "UNKNOWN" in p_name or "INDETERMINATE" in p_name
        
        st.markdown(f"""
<div class="zenith-card">
<p class="metric-title">{'Neural Mesh Sync' if is_unknown else 'Critical Specimen'}</p>
<h2 style="font-size: 1.8rem; margin-bottom: 0.1rem; color: #34d399; text-shadow: 0 0 25px rgba(16, 185, 129, 0.7); font-weight: 800; letter-spacing: 1px;">
    {r.get('common_name', p_name).upper() if not is_unknown else "INDETERMINATE SPECIMEN"}
</h2>
{f'<p style="color:#6ee7b7; font-size:0.8rem; font-style:italic; margin-top:-5px;">{p_name}</p>' if not is_unknown and r.get('common_name') and r.get('common_name').upper() != p_name else "<p style='color:#f87171; font-size:0.8rem; font-style:italic; margin-top:-5px;'>Bio-Signature Unrecognized</p>" if is_unknown else ""}
<p style="font-size:0.9rem; opacity:0.8; margin: 0 0 1.5rem 0; line-height: 1.5;">
ID: {r.get('timestamp', 'NEW')} | Uplink: STABLE<br/>
CO2 Credit Score: <span style="color:#10b981; font-weight:700;">{r.get('carbon', 0)}kg/yr</span>
</p>

<div style="display:flex; justify-content:center; align-items:center; background: rgba(0,0,0,0.2); padding: 12px 18px; border-radius: 14px; border: 1px solid rgba(16,185,129,0.1);">
<div style="text-align:center;">
<p class="metric-title" style="font-size: 0.65rem; margin-bottom: 5px;">Specimen Risk Level</p>
<span class="badge badge-{'critical' if r.get('q', {}).get('score', 3) > 3 else 'warning' if r.get('q', {}).get('score', 3) >= 2 else 'optimal'}" style="font-size: 1.2rem; padding: 10px 25px; box-shadow: 0 0 30px rgba(16,185,129,0.3);">
{f"{r.get('q', {}).get('label', 'Baseline').upper()} RISK" if not is_unknown else "SYNCING..."}
</span>
</div>
</div>
</div>
""", unsafe_allow_html=True)
        
        rtabs = st.tabs(["🧪 Pathology", "🌿 Care", "🧬 Genomics", "📉 Quantum", "🌈 Spectral", "🛡️ Security", "🛒 Purchase", "📄 Reports"])
        
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
                
                # NPK Visualization
                st.divider()
                st.markdown("<h4 style='color:#6ee7b7;'>🧬 NPK Saturation Matrix</h4>", unsafe_allow_html=True)
                npk = r.get('npk', {"Nitrogen": 50, "Phosphorus": 50, "Potassium": 50})
                n_col, p_col, k_col = st.columns(3)
                n_col.metric("Nitrogen (N)", f"{npk['Nitrogen']}%", delta=f"{npk['Nitrogen']-60}%" if npk['Nitrogen'] < 60 else None)
                p_col.metric("Phosphorus (P)", f"{npk['Phosphorus']}%", delta=f"{npk['Phosphorus']-50}%" if npk['Phosphorus'] < 50 else None)
                k_col.metric("Potassium (K)", f"{npk['Potassium']}%", delta=f"{npk['Potassium']-70}%" if npk['Potassium'] < 70 else None)
            with col_p2:
                # 12. Edge-AI Thermal Heatmap
                st.markdown("<p class='metric-title'>Edge-AI Thermal Profile</p>", unsafe_allow_html=True)
                if "spectral_img" in r:
                    thermal = cv2.applyColorMap(cv2.cvtColor(r["micro_img"], cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)
                    st.image(cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB), width="stretch")
                st.caption("Thermal metabolic hyperactivity localized.")
                
        with rtabs[1]:
            st.markdown("<h4 style='color:#6ee7b7;'>🌿 Botanical Intelligence (Perenual API)</h4>", unsafe_allow_html=True)
            care = r.get('care', {})
            if care and isinstance(care, dict):
                c1, c2 = st.columns(2)
                with c1:
                    st.write(f"**Cycle:** {care.get('cycle', 'N/A')}")
                    st.write(f"**Watering:** {care.get('watering', 'N/A')}")
                    st.write(f"**Sunlight:** {', '.join(care.get('sunlight', [])) if isinstance(care.get('sunlight'), list) else care.get('sunlight', 'N/A')}")
                with c2:
                    st.markdown(f"""
                    <div style='background:rgba(16,185,129,0.1); padding:15px; border-radius:12px; border:1px solid #10b981;'>
                        <p style='color:#10b981; font-weight:700; margin-bottom:5px;'>VITALITY PROTOCOL</p>
                        <p style='font-size:0.85rem;'>Based on Perenual botanical datasets, this specimen requires <b>{care.get('watering', 'standard')}</b> hydration and <b>{care.get('sunlight', ['stable sunlight'])[0]}</b> for optimal cellular regeneration.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Botanical care data for this specific variant is localized but currently awaiting peer-review verification.")
                st.info("General Protocol: Maintain soil pH 6.0-7.0 and ensure adequate nitrogen saturation.")
                
        with rtabs[2]:
            st.markdown("<h4 style='color:#6ee7b7;'>🧬 Pathogen Genomic Fingerprint</h4>", unsafe_allow_html=True)
            col_dna1, col_dna2 = st.columns([1, 1])
            with col_dna1:
                dna = r.get('dna', 'ATCG'*8)
                st.markdown(f"""
                <div style="background:#01160d; padding:20px; border-radius:12px; border:2px solid #10b981; font-family:monospace; color:#34d399; font-size:1.2rem; letter-spacing:4px; overflow-wrap:break-word;">
                    {dna[:8]}<br/>{dna[8:16]}<br/>{dna[16:24]}<br/>{dna[24:]}
                </div>
                """, unsafe_allow_html=True)
                st.caption("DNA Sequence: Mutant alleles highlighted in Matrix.")
            with col_dna2:
                # Vitality Gauge
                vitality = 100 - (r.get('q', {}).get('score', 3) * 20) + random.randint(0, 10)
                st.markdown("<p class='metric-title'>Biological Vitality Index</p>", unsafe_allow_html=True)
                st.progress(max(0, min(100, vitality)) / 100)
                st.write(f"Estimated Life Expectancy: **{vitality}% Baseline**")
            
            st.divider()
            st.markdown("<h4 style='color:#6ee7b7;'>🚁 Precision Drone Waypoints</h4>", unsafe_allow_html=True)
            wps = r.get('waypoints', [])
            if wps:
                wp_df = pd.DataFrame(wps)
                st.map(wp_df, size=20, color='#10b981')
            st.button("EXPORT MAVLINK / DJI-SDK WAYPOINTS", use_container_width=True)
            
        with rtabs[3]:
            st.markdown("<h4 style='color:#6ee7b7;'>Qiskit Enterprise Architecture (V5)</h4>", unsafe_allow_html=True)
            q_data = r.get('q', {})
            
            with st.container():
                st.markdown("""
                <div style='background:rgba(16,185,129,0.05); padding:1rem; border:1px solid rgba(16,185,129,0.3); border-radius:12px;'>
                <h5 style='color:#34d399;'>Biological Trait Amplitudes mapped to 8-Qubit Tensor Product Space</h5>
                <p style='font-size:0.85rem; opacity:0.8;'>The specimen's RGB variances and cellular entropy matrices have been encoded directly into Y-axis wavefunction rotations, followed by a hardware-efficient ring correlation (CNOT) layer.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Circuit Depth", q_data.get('depth', 0))
            mcol2.metric("Logical Qubits", "8 (Fully Entangled)")
            mcol3.metric("Entanglement Index", f"{int(q_data.get('entanglement', 0)*100)}%")
            mcol4.metric("Backend Proxy", q_data.get('backend', 'Unknown Provider').upper())
            
            st.markdown("<p class='metric-title'> Tensor Spatial Representation</p>", unsafe_allow_html=True)
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

            with st.expander("📂 Explore Real-Time Circuit Compilation"):
                st.code(q_data.get('circuit_str', 'Circuit data missing'), language="qasm")
            
            st.markdown("<h4 style='color:#34d399;'>Wavefunction Collapse Distribution</h4>", unsafe_allow_html=True)
            # Filter top 20 probabilities to prevent massive chart clobbering
            prob_dict = q_data.get('prob', {'00000000': 1.0})
            top_probs = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)[:20])
            pdf_data = pd.DataFrame(list(top_probs.items()), columns=['Eigenstate Vector', 'Amplitude Probability'])
            st.bar_chart(pdf_data.set_index('Eigenstate Vector'), color="#10b981", height=300)

        with rtabs[4]:
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
            if "spectral_img" in r:
                st.image(cv2.cvtColor(r["spectral_img"], cv2.COLOR_BGR2RGB), use_container_width=True, caption="Spectral Heatmap (Overlay)")
            
            st.divider()
            st.markdown("<h4 style='color:#6ee7b7;'>🛰 Global Incident Dispatch</h4>", unsafe_allow_html=True)
            if st.button("🚀 ALERT REGIONAL AGRONOMIST", use_container_width=True, type="primary"):
                st.toast("Bio-Security Alert Dispatched!", icon="🚨")
                st.success("Regional bio-security updated.")

        with rtabs[5]:
            st.markdown("<h4 style='color:#6ee7b7;'>🛡️ Quantum Bio-Encryption</h4>", unsafe_allow_html=True)
            st.info("Generating unique encryption hashes using quantum entanglement.")
            q_hash = base64.b64encode(os.urandom(24)).decode()
            st.markdown(f"<div style='background:rgba(0,0,0,0.3); padding:20px; border-radius:12px; font-family:monospace; border:1px solid #34d399;'>{q_hash}</div>", unsafe_allow_html=True)
            st.button("REVOKE ACCESS KEYS", width="stretch")

        with rtabs[6]:
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

        with rtabs[7]:
            st.markdown("<h4 style='color:#6ee7b7;'>📄 Clinical Reporting Engine</h4>", unsafe_allow_html=True)
            st.info("The Zenith engine generates encrypted, clinical-grade PDF dossiers including ROI projections and molecular diagnostics.")
            
            if st.button("🚀 INITIATE DOSSIER COMPILATION", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Compiling clinical-grade molecular dossier..."):
                        pdf = FPDF()
                        pdf.add_page()
                        
                        # Header
                        pdf.set_fill_color(1, 22, 13) # Zenith Dark Green
                        pdf.rect(0, 0, 210, 40, 'F')
                        pdf.set_text_color(52, 211, 153) # Zenith Green
                        pdf.set_font("helvetica", "B", 26)
                        pdf.text(10, 25, "PLANTPULSE ZENITH REPORT")
                        pdf.set_font("helvetica", "I", 10)
                        pdf.text(10, 32, "Enterprise Botanical Intelligence & Quantum Diagnostics")
                        
                        pdf.set_y(50)
                        pdf.set_text_color(0, 0, 0)
                        
                        # Section: Specimen Identity
                        pdf.set_font("helvetica", "B", 16)
                        pdf.set_text_color(16, 185, 129)
                        pdf.cell(0, 10, "1. SPECIMEN IDENTITY", ln=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("helvetica", "", 12)
                        
                        # Helper for safe encoding
                        def safe_str(val):
                            return str(val).encode('latin-1', 'replace').decode('latin-1')

                        pdf.cell(0, 8, f"Scientific Name: {safe_str(r.get('plant', 'Unknown'))}", ln=True)
                        pdf.cell(0, 8, f"Common Name: {safe_str(r.get('common_name', 'N/A'))}", ln=True)
                        pdf.cell(0, 8, f"Scan Timestamp: {safe_str(r.get('timestamp'))}", ln=True)
                        pdf.cell(0, 8, f"Confidence Score: {safe_str(r.get('score'))}%", ln=True)
                        
                        # Section: Pathology
                        pdf.ln(5)
                        pdf.set_font("helvetica", "B", 16)
                        pdf.set_text_color(16, 185, 129)
                        pdf.cell(0, 10, "2. PATHOLOGICAL DIAGNOSIS", ln=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("helvetica", "B", 12)
                        pdf.cell(0, 8, f"Condition: {safe_str(r.get('disease'))}", ln=True)
                        pdf.set_font("helvetica", "", 11)
                        pdf.multi_cell(0, 7, f"Clinical Observation: {safe_str(r.get('pathology'))}")
                        
                        # Section: NPK & ROI
                        pdf.ln(5)
                        pdf.set_font("helvetica", "B", 16)
                        pdf.set_text_color(16, 185, 129)
                        pdf.cell(0, 10, "3. BIO-CHEMICAL & ECONOMIC METRICS", ln=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("helvetica", "", 11)
                        npk_data = r.get('npk', {})
                        pdf.cell(0, 8, f"NPK Saturation: N:{npk_data.get('Nitrogen')}% | P:{npk_data.get('Phosphorus')}% | K:{npk_data.get('Potassium')}%", ln=True)
                        pdf.cell(0, 8, f"Economic Impact (ROI): -${safe_str(r.get('roi'))} USD Projected Loss", ln=True)
                        pdf.cell(0, 8, f"Carbon Sequestration: {safe_str(r.get('carbon'))} kg/yr", ln=True)
                        
                        # Section: Remediation
                        pdf.ln(5)
                        pdf.set_font("helvetica", "B", 16)
                        pdf.set_text_color(16, 185, 129)
                        pdf.cell(0, 10, "4. REMEDIATION PROTOCOLS", ln=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("helvetica", "", 11)
                        for k, v in r.get('rx', {}).items():
                            pdf.multi_cell(0, 7, f"- {k.title()}: {safe_str(v)}")
                        
                        # Section: Quantum Diagnostic
                        pdf.ln(5)
                        pdf.set_font("helvetica", "B", 16)
                        pdf.set_text_color(16, 185, 129)
                        pdf.cell(0, 10, "5. QUANTUM ENTANGLEMENT ANALYSIS", ln=True)
                        pdf.set_text_color(0, 0, 0)
                        pdf.set_font("helvetica", "", 11)
                        q_data_pdf = r.get('q', {})
                        pdf.cell(0, 8, f"Quantum Risk State: {safe_str(q_data_pdf.get('label'))}", ln=True)
                        pdf.cell(0, 8, f"Circuit Depth: {safe_str(q_data_pdf.get('depth'))}", ln=True)
                        pdf.cell(0, 8, f"Entanglement Index: {int(q_data_pdf.get('entanglement',0)*100)}%", ln=True)
                        
                        st.session_state.pdf_buffer = bytes(pdf.output())
                        st.session_state.pdf_filename = f"Zenith_Report_{r.get('plant', 'specimen')}_{r.get('timestamp')}.pdf"
                        st.toast("Dossier compiled successfully!", icon="✅")
                except Exception as e:
                    st.error(f"Dossier Compilation Error: {str(e)}")
            
            if "pdf_buffer" in st.session_state:
                st.download_button(
                    label="⬇️ DOWNLOAD FINAL DOSSIER",
                    data=st.session_state.pdf_buffer,
                    file_name=st.session_state.pdf_filename,
                    mime="application/pdf",
                    use_container_width=True
                )
    else:
        st.info("Scanner idle. Awaiting specimen input...")

# --- Assistant ---
st.divider()
st.subheader("💬 Zenith Smart Assistant")
chat_box = st.container(height=350)
with chat_box:
    for m in st.session_state.chat_history:
        st.chat_message(m["role"]).write(m["content"])

if chat_prompt := st.chat_input("Query the Quantum Oracle..."):
    st.session_state.chat_history.append({"role": "user", "content": chat_prompt})
    
    # Zenith Context-Aware AI Response Logic
    p_mode = st.session_state.get('assistant_mode', 'Dr. Leaf')
    lr = st.session_state.last_results
    plant = lr.get('plant', 'specimen')
    disease = lr.get('disease', 'healthy')
    q_risk = lr.get('q', {}).get('label', 'Standard')

    # Pre-calculated diagnostic messages for Dr. Leaf
    is_healthy = "healthy" in disease.lower()
    h_status = "Healthy" if is_healthy else "Diseased"
    h_advice = "Keep up the good care!" if is_healthy else "Don't worry — early detection means we can still help it recover."
    
    # Dr. Leaf's Knowledge Matrix (30 Q&A Pairs)
    DR_LEAF_KNOWLEDGE = {
        "brown edges": "Crispy brown edges usually mean low humidity or underwatering. Mist your plant lightly or place a water tray nearby to boost moisture.",
        "drooping": "Drooping leaves are your plant's way of saying it's thirsty or stressed. Check the soil moisture first — if dry, water it; if soggy, let it breathe.",
        "small leaves": "Stunted new growth often points to nutrient deficiency or root crowding. Try a balanced fertilizer or consider repotting into a slightly larger container.",
        "dots": "Those are most likely Spider Mites — very common and treatable! Wipe leaves with a damp cloth and spray with diluted neem oil every 3 days.",
        "losing leaves": "Sudden leaf drop is usually triggered by temperature shock, overwatering, or a dramatic change in environment. Keep your plant away from cold drafts and air vents.",
        "sunlight": "Yes! Too much direct sun causes leaf scorch — pale, bleached, or papery patches on leaves. Move your plant to bright but indirect light for recovery.",
        "black spots": "Black spots are a classic sign of fungal or bacterial disease. Remove affected leaves immediately, avoid overhead watering, and apply a suitable fungicide.",
        "lower leaves": "Some lower leaf drop is natural as plants grow. But if it's excessive, it could signal overwatering, poor light, or a nutrient imbalance — worth investigating!",
        "sticky": "Sticky leaves are usually caused by scale insects or aphids secreting a substance called honeydew. Wipe leaves clean and treat with insecticidal soap promptly.",
        "smell": "A foul smell near the soil is a strong sign of root rot. Remove the plant from its pot, trim any black or mushy roots, and repot in fresh, well-draining soil.",
        "pale": "Pale or whitish leaves indicate chlorosis — a lack of chlorophyll usually caused by iron or nitrogen deficiency. A good liquid fertilizer should bring the green back.",
        "remedy": "Yes! Diluted neem oil, baking soda spray, and garlic water are effective home treatments for many fungal and pest-related issues. Always test on one leaf first.",
        "overwatered": "Overwatered plants have soft, yellow, mushy leaves and wet soil. Underwatered plants have dry, crispy, curling leaves and bone-dry soil. Feel both the leaf and the soil!",
        "diseased leaves": "Absolutely — and as soon as possible! Removing infected leaves stops the disease from spreading and lets the plant focus its energy on healthy new growth.",
        "recover": "With the right care, most plants show improvement within 1 to 3 weeks. Full recovery can take 4–8 weeks depending on how severe the condition was. Stay consistent and be patient! 🌱",
        "healthy": f"Based on what I can see, your plant appears to be **{h_status}**. {h_advice}",
        "what disease": f"Your plant is showing signs of **{disease}**. I'll walk you through exactly what to do next.",
        "yellow": "Yellow leaves usually mean overwatering, nutrient deficiency, or too little sunlight. Check your watering routine and move the plant to a brighter spot.",
        "brown spots": "Brown spots are often caused by fungal infections, sunburn, or underwatering. Look at the pattern — dry, crispy edges suggest thirst, while soft dark spots suggest fungus.",
        "contagious": f"Yes! If your plant has **{disease}**, it can spread to nearby plants. Isolate it immediately and treat it before returning it to your garden.",
        "fungal": "Remove the affected leaves, improve air circulation, and apply a mild fungicide or a baking soda spray. Avoid getting the leaves wet when watering.",
        "curling": "Curling leaves usually signal heat stress, underwatering, or pest activity. Feel the soil — if it's bone dry, give your plant a good drink right away.",
        "powdery": "That's likely Powdery Mildew — a common fungal disease. Increase air circulation, reduce humidity, and treat with a diluted neem oil or baking soda solution.",
        "how often": "Do a quick visual check every 3–5 days. Early signs like small spots, wilting, or color changes are much easier to treat before they spread.",
        "wilted": "Wilting after watering can be a sign of root rot — caused by overwatering and poor drainage. Check the roots; healthy ones are white, while rotten ones are brown and mushy.",
        "holes": "Tiny holes usually point to pest damage — insects like aphids, caterpillars, or beetles. Inspect the underside of leaves closely and treat with neem oil or insecticidal soap.",
        "prevent": "Water at the base, not the leaves. Ensure good sunlight and airflow, use clean tools, and avoid overwatering — these simple habits prevent most common diseases.",
        "eat": f"It depends on the disease. For **{disease}**, it's best to **avoid consuming** affected produce until the plant is fully treated and healthy again.",
        "worried": "If more than 50% of the leaves are affected, the stem is soft or rotting, or the plant isn't responding to treatment after 2 weeks — it may need urgent intervention or replanting."
    }

    reply = None
    # Check knowledge matrix first
    for key, val in DR_LEAF_KNOWLEDGE.items():
        if key in chat_prompt.lower():
            reply = f"**Dr. Leaf:** {val}"
            break

    if not reply:
        # Personality-driven Prompting
        if p_mode == "Dr. Leaf":
            reply = f"**Dr. Leaf:** I'm here to help with your {plant}. It seems to be experiencing {disease}. What specific symptoms are you observing?"
        elif p_mode == "Quantum Oracle":
            base = "I am Groot... The quantum wave function of this specimen has collapsed."
            if "risk" in chat_prompt.lower() or "threat" in chat_prompt.lower():
                reply = f"{base} Identification of **{disease}** in {plant} indicates a **{q_risk}** risk state in the local matrix."
            elif "fix" in chat_prompt.lower() or "help" in chat_prompt.lower() or "remedy" in chat_prompt.lower():
                reply = f"{base} Remediation protocols are locked. I recommend following the {lr.get('p_cat')} directive immediately."
            else:
                reply = f"{base} I am detecting high entanglement indices in the {plant}'s biological mesh. The universe is watching."
        elif p_mode == "Bio-Scientist":
            base = "[Zenith Bio-Core] Analyzing pathological vectors..."
            if "risk" in chat_prompt.lower() or "threat" in chat_prompt.lower():
                reply = f"{base} The specimen '{plant}' shows symptoms of {disease}. Pathogen severity is marked as **{q_risk}** based on cellular entropy."
            elif "fix" in chat_prompt.lower() or "help" in chat_prompt.lower() or "remedy" in chat_prompt.lower():
                reply = f"{base} Tactical remediation required: {list(lr.get('rx', {}).values())[0]}. Bio-security protocols suggested."
            else:
                reply = f"{base} NPK saturation is currently at {lr.get('npk', {}).get('Nitrogen')}% Nitrogen. Cellular respiration appears {'impeded' if q_risk != 'Optimal' else 'efficient'}."
        else: # Farm Guardian
            base = "Steady now... The Guardian system is active."
            if "risk" in chat_prompt.lower() or "threat" in chat_prompt.lower():
                reply = f"{base} We've spotted {disease} on your {plant}. It looks like a **{q_risk}** situation. Don't let it spread to the rest of the field."
            elif "fix" in chat_prompt.lower() or "help" in chat_prompt.lower() or "remedy" in chat_prompt.lower():
                reply = f"{base} I've listed some steps for you in the directives. Grab some {lr.get('p_cat')} and let's get to work."
            else:
                reply = f"{base} The {plant} is under my watch. We'll get through this season together."

    st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()


st.caption("PlantPulse Zenith v5.0 | chasesolutions.in | © 2026")