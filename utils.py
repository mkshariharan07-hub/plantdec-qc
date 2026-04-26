"""
utils.py — PlantPulse Shared Utilities
=======================================
Single source of truth for:
  • Feature extraction  (used by main.py, app.py, server.py)
  • Artifact paths
  • Disease knowledge base
  • Image decoding helpers

RULE: Any change to extract_features() is made HERE only.
      All other files import from this module.
"""

import cv2
import numpy as np
import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

import requests
import base64
from typing import Optional, Dict
from dotenv import load_dotenv

# Optional Deep Learning Imports
try:
    # Attempt to import TFLite
    import tensorflow.lite as tflite
    HAS_TFLITE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        HAS_TFLITE = True
    except ImportError:
        HAS_TFLITE = False

def predict_with_tflite(img_bgr: np.ndarray, model_path: str, class_indices_path: str) -> dict:
    """Perform inference using a TFLite model and map results to class names."""
    if not HAS_TFLITE:
        return {"error": "TFLite runtime not found. Please install tensorflow or tflite-runtime."}
    
    if not os.path.exists(model_path):
        return {"error": f"Model file not found at {model_path}"}

    try:
        # Load Interpreter
        import tensorflow.lite as tflite
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Preprocess Image
        # Get required input shape (usually 224x224 or 256x256)
        input_shape = input_details[0]['shape']
        h, w = input_shape[1], input_shape[2]
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (w, h))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32)
        
        # Normalize if necessary (assuming 0-255 -> 0-1)
        if input_details[0]['dtype'] == np.float32:
            input_data = input_data / 255.0
            
        # Set Tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get Result
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        best_idx = np.argmax(output_data)
        confidence = float(output_data[best_idx])
        
        # Map Class
        label = str(best_idx)
        if os.path.exists(class_indices_path):
            import json
            with open(class_indices_path, 'r') as f:
                indices = json.load(f)
                # Reverse mapping (assuming index: name or name: index)
                # Common format is {"className": index}
                for name, idx in indices.items():
                    if str(idx) == str(best_idx):
                        label = name
                        break
        
        # Parse label (e.g., 'Apple___Scab' -> 'Scab')
        plant_name, disease_name = "Specimen", label
        if "___" in label:
            plant_name, disease_name = label.split("___")
            disease_name = disease_name.replace("_", " ").title()
            plant_name = plant_name.replace("_", " ").title()
        else:
            disease_name = label.replace("_", " ").title()

        return {
            "disease": disease_name,
            "plant": plant_name,
            "probability": round(confidence * 100, 1),
            "description": f"Diagnosis via Edge-TFLite Neural Mesh. Model Confidence: {round(confidence*100,1)}%.",
            "source": "TFLite Local"
        }
    except Exception as e:
        return {"error": f"TFLite Inference Failure: {str(e)}"}

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY")
CROP_HEALTH_API_KEY = os.getenv("CROP_HEALTH_API_KEY")
PERENUAL_API_KEY = os.getenv("PERENUAL_API_KEY")

# ── Artifact paths (one place to change if you move files) ────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "plant_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "plant_scaler.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "training_report.txt")
IMG_SIZE    = (128, 128)

# Feature-space identifiers
FEATURE_MODE_RAW  = "raw_pixels"    # old model: 128×128×3 = 49152 dims
FEATURE_MODE_HIST = "histogram"     # new model: 63 dims
RAW_PIXEL_DIM     = 128 * 128 * 3  # = 49152


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Deterministic, normalized feature vector (63 dims).

    Feature layout:
      [0:24]  — Hue histogram, 24 bins, normalized to sum=1
      [24:40] — Saturation histogram, 16 bins, normalized
      [40:56] — Value histogram, 16 bins, normalized
      [56:59] — BGR channel means  (divided by 255)
      [59:62] — BGR channel stds   (divided by 255)
      [62]    — Canny edge density (0–1)

    Args:
        img: BGR image as uint8 numpy array (any size).
    Returns:
        1-D float64 array of length 63.
    """
    # Fix distribution shift: dataset was 8x8, bottleneck all inputs to match!
    img   = cv2.resize(img, (8, 8))
    img   = cv2.resize(img, IMG_SIZE)
    hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color histograms — normalized to unit sum
    h_hist = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
    h_hist /= (h_hist.sum() + 1e-7)
    s_hist /= (s_hist.sum() + 1e-7)
    v_hist /= (v_hist.sum() + 1e-7)

    # Per-channel mean & std (scaled 0–1)
    means, stds = cv2.meanStdDev(img)
    stats = np.concatenate([means.flatten(), stds.flatten()]) / 255.0

    # Edge density
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0)) / (IMG_SIZE[0] * IMG_SIZE[1])

    return np.concatenate([h_hist, s_hist, v_hist, stats, [edge_density]])


FEATURE_DIM = len(extract_features(np.zeros((8, 8, 3), dtype=np.uint8)))  # = 63


def extract_features_raw(img: np.ndarray) -> np.ndarray:
    """
    Legacy extractor — raw pixel flatten (128×128×3 = 49152 dims).
    Used automatically when the loaded model was trained this way.
    DO NOT use for new training; use extract_features() instead.
    """
    return cv2.resize(img, IMG_SIZE).flatten().astype(np.float64)


def get_feature_mode(model) -> str:
    """
    Inspect a loaded model and return which feature extractor it was trained with.

    Returns:
        'raw_pixels'  — model.n_features_in_ == 49152  (old pipeline)
        'histogram'   — model.n_features_in_ == 63     (new pipeline)

    Raises:
        ValueError if the feature count is unrecognised.
    """
    n = model.n_features_in_
    if n == RAW_PIXEL_DIM:
        return FEATURE_MODE_RAW
    if n == FEATURE_DIM:
        return FEATURE_MODE_HIST
    raise ValueError(
        f"Unrecognised model feature count: {n}. "
        f"Expected {RAW_PIXEL_DIM} (old) or {FEATURE_DIM} (new). "
        f"Retrain with `python main.py`."
    )


def extract_for_model(img: np.ndarray, model) -> np.ndarray:
    """
    Extract features in whichever space the model was trained in.
    Removes the need for callers to know which mode is active.
    """
    mode = get_feature_mode(model)
    if mode == FEATURE_MODE_RAW:
        return extract_features_raw(img).reshape(1, -1)
    return extract_features(img).reshape(1, -1)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE DECODING
# ═══════════════════════════════════════════════════════════════════════════════
def decode_bytes_to_bgr(raw_bytes: bytes) -> Optional[np.ndarray]:
    """
    Decode raw image bytes → BGR ndarray.
    Returns None if bytes are empty or decoding fails.
    """
    if not raw_bytes:
        return None
    arr = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img  # None on failure


def decode_file_to_bgr(path: str) -> Optional[np.ndarray]:
    """Read an image file from disk → BGR ndarray."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════
DISEASE_INFO: Dict[str, Dict] = {
    "healthy": {
        "severity": "low",
        "color":    "#10b981",
        "emoji":    "🌱",
        "tips":     "No treatment needed. Maintain regular watering and sunlight.",
    },
    "early_blight": {
        "severity": "medium",
        "color":    "#f59e0b",
        "emoji":    "🟡",
        "tips":     "Remove affected leaves. Apply copper-based fungicide. Avoid overhead watering.",
    },
    "late_blight": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Isolate plant immediately. Apply mancozeb or chlorothalonil. Destroy infected tissue.",
    },
    "leaf_mold": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Improve air circulation. Apply fungicide. Reduce ambient humidity.",
    },
    "bacterial_spot": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Use copper-based bactericide. Avoid working with wet plants.",
    },
    "common_rust": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Apply triazole fungicide early. Rotate crops next season.",
    },
    "northern_leaf_blight": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Apply fungicide at first sign. Use resistant varieties next cycle.",
    },
    "gray_leaf_spot": {
        "severity": "medium",
        "color":    "#f59e0b",
        "emoji":    "🟡",
        "tips":     "Improve drainage. Apply strobilurin fungicide preventively.",
    },
    "powdery_mildew": {
        "severity": "medium",
        "color":    "#f59e0b",
        "emoji":    "🟡",
        "tips":     "Apply sulfur or potassium bicarbonate spray. Ensure good airflow.",
    },
    "target_spot": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Remove infected leaves. Apply chlorothalonil or mancozeb.",
    },
    "mosaic_virus": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "No cure — remove and destroy infected plants. Control aphid vectors.",
    },
    "yellow_leaf_curl_virus": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Remove infected plants. Use reflective mulches to deter whiteflies.",
    },
    "septoria_leaf_spot": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Remove spotted leaves. Improve air circulation. Apply copper-based fungicide in early spring.",
    },
    "anthracnose": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Prune out dead twigs. Apply fungicide (chlorothalonil) during bud break. Rake fallen leaves.",
    },
    "angular_leaf_spot": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Avoid overhead watering. Apply copper-based bactericide. Rotate crops to non-cucurbits.",
    },
    "apple_scab": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Rake and destroy fallen leaves. Apply fungicide (captan or sulfur) during green tip and petal fall stages.",
    },
    "scab": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Prune infected branches. Use resistant cultivars. Apply protective fungicides preventatively.",
    },
    "mango_malformation": {
        "severity": "critical",
        "color":    "#b91c1c",
        "emoji":    "💀",
        "tips":     "Prune infected panicles 30cm below symptoms. Disinfect tools. Destroy infected material.",
    },
    "anthracnose": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Apply copper-based fungicides. Prune dead wood. Improve air circulation.",
    },
    "guava_rust": {
        "severity": "high",
        "color":    "#ef4444",
        "emoji":    "🔴",
        "tips":     "Apply neem oil or sulfur. Remove infected leaves. Avoid overhead irrigation.",
    },
    "cercospora": {
        "severity": "medium",
        "color":    "#f97316",
        "emoji":    "🟠",
        "tips":     "Remove fallen leaves. Apply balanced NPK. Use protective fungicides if spreading.",
    }
}

# BOTANICAL CROSS-REFERENCE MATRIX
# Maps visual patterns from general models to host-specific pathogens.
CROSS_REFERENCE_MATRIX: Dict[str, Dict[str, str]] = {
    "dogwood": {
        "scab": "Dogwood Anthracnose / Septoria",
        "blight": "Dogwood Anthracnose",
        "spot": "Septoria Leaf Spot"
    },
    "cucumber": {
        "scab": "Angular Leaf Spot",
        "blight": "Downy Mildew",
        "spot": "Angular Leaf Spot"
    },
    "tomato": {
        "scab": "Septoria Leaf Spot",
        "blight": "Early/Late Blight",
        "spot": "Target Spot"
    },
    "potato": {
        "scab": "Common Scab",
        "blight": "Late Blight"
    }
}

def get_botanical_equivalent(host: str, visual_pattern: str) -> Optional[str]:
    """Look up the scientifically accurate disease name for a specific host based on visual pattern."""
    host_key = host.lower()
    pattern_key = visual_pattern.lower().replace("_", " ")
    
    # Check for host match
    for h_key, patterns in CROSS_REFERENCE_MATRIX.items():
        if h_key in host_key:
            # Check for pattern match
            for p_key, actual_name in patterns.items():
                if p_key in pattern_key:
                    return actual_name
    return None

_FALLBACK_INFO = {
    "severity": "medium",
    "color":    "#f59e0b",
    "emoji":    "⚠️",
    "tips":     "Consult an agronomist for targeted treatment advice.",
}


def get_disease_info(disease: str) -> dict:
    """Lookup disease metadata by fuzzy key match with high-severity priority."""
    key = disease.lower().replace(" ", "_")
    
    # Priority 1: Exact or direct substring match for high-severity keywords
    for high_risk in ["scab", "blight", "virus", "rust", "mold"]:
        if high_risk in key:
            for k, v in DISEASE_INFO.items():
                if high_risk in k:
                    return v
                    
    # Priority 2: General fuzzy match
    for k, v in DISEASE_INFO.items():
        if k in key or key in k:
            return v
            
    return _FALLBACK_INFO


# ═══════════════════════════════════════════════════════════════════════════════
# ARTIFACT LOADING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def load_model_and_scaler():
    """
    Load plant_model.pkl and plant_scaler.pkl from disk.
    Returns (model, scaler). scaler may be None if not found.
    Raises FileNotFoundError if model is missing.
    """
    import joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at '{MODEL_PATH}'. Run `python main.py` to train."
        )
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler


def predict_image(img_bgr: np.ndarray, model, scaler=None) -> dict:
    """
    Full prediction pipeline for a BGR image.
    Auto-detects whether the model expects raw pixels (49152) or
    histogram features (63) and extracts accordingly.

    Returns dict with keys:
        plant, disease, confidence, prediction_raw, top5,
        severity, tips, color, emoji, feature_mode
    """
    # ── Auto-detect feature space ────────────────────────────────────────────
    mode     = get_feature_mode(model)   # raises ValueError on unknown dim
    features = extract_for_model(img_bgr, model)  # shape (1, n_features)

    # Scaler only applies to histogram-trained models (raw-pixel models
    # have no associated scaler in the old pipeline)
    if scaler is not None and mode == FEATURE_MODE_HIST:
        features = scaler.transform(features)

    prediction  = model.predict(features)[0]
    conf_probs  = model.predict_proba(features)[0]
    confidence  = float(np.max(conf_probs) * 100)

    try:
        plant, disease = prediction.split("___")
    except ValueError:
        plant, disease = "Unknown", prediction

    info = get_disease_info(disease)

    top5 = sorted(
        [{"class": c, "probability": round(float(p) * 100, 2)}
         for c, p in zip(model.classes_, conf_probs)],
        key=lambda x: -x["probability"]
    )[:5]

    return {
        "plant":        plant,
        "disease":      disease,
        "confidence":   round(confidence, 2),
        "prediction_raw": prediction,
        "top5":         top5,
        "severity":     info["severity"],
        "tips":         info["tips"],
        "color":        info["color"],
        "emoji":        info["emoji"],
        "feature_mode": mode,   # 'raw_pixels' or 'histogram'
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERNAL API CONNECTORS
# ═══════════════════════════════════════════════════════════════════════════════

def identify_plant_with_plantnet(img_bgr: np.ndarray, api_key: str = None, verify_ssl: bool = False) -> dict:
    """Identify plant using PlantNet API with adaptive flora fallback."""
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("PLANTNET_API_KEY")
    
    if not api_key:
        return {"error": "PlantNet API Key missing"}
    
    try:
        # Resize for API optimization (max 1024px)
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)
            
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        files = [('images', ('image.jpg', img_encoded.tobytes(), 'image/jpeg'))]
        data = {'organs': 'auto'}
        
        # Strategy: Try 'all' first, fallback to 'weurope' if rejected
        last_raw = "None"
        for project in ["all", "weurope"]:
            url = f"https://my-api.plantnet.org/v2/identify/{project}?api-key={api_key}"
            for attempt in range(2): # Retry once for transient failures
                try:
                    # SSL Verification based on parameter (default False for Windows compatibility)
                    # Decreased timeout to 15s to prevent massive UI freezes when blocked
                    response = requests.post(url, files=files, data=data, timeout=15, verify=verify_ssl)
                    if response.status_code == 200:
                        res = response.json()
                        if res.get('results'):
                            best = res['results'][0]
                            score = round(best.get('score', 0) * 100, 1)
                            species = best.get('species', {})
                            common_names = species.get('commonNames', [])
                            
                            # Refined naming logic
                            s_name = species.get('scientificNameWithoutAuthor') or species.get('scientificName')
                            if not s_name:
                                # Fallback to genus/family if species name is missing
                                genus = species.get('genus', {}).get('scientificNameWithoutAuthor') 
                                s_name = f"{genus} sp." if genus else "Unknown Species"

                            return {
                                "scientific_name": s_name,
                                "common_names": common_names,
                                "score": score,
                                "family": species.get('family', {}).get('scientificNameWithoutAuthor'),
                                "genus": species.get('genus', {}).get('scientificNameWithoutAuthor'),
                                "raw_res": best
                            }
                    else:
                        last_raw = f"HTTP {response.status_code}: {response.text[:50]}"
                        break # Don't retry on rejection, only on timeout/failure
                except Exception as e:
                    last_raw = f"Link Failure: {str(e)}"
                    if attempt == 0: continue # Retry on first failure
            
            # If we reached here, this project failed or had no results
            if last_raw == "None":
                last_raw = "No matches found for this specimen."
                    
        return {"error": f"PlantNet: {last_raw}"}
    except Exception as e:
        return {"error": f"PlantNet Root Failure: {str(e)}"}


def identify_disease_with_kindwise(img_bgr: np.ndarray, api_key: str = None) -> dict:
    """Identify diseases using Kindwise API with multi-provider credential support."""
    if not api_key:
        import streamlit as st
        api_key = os.getenv("CROP_HEALTH_API_KEY") or (st.secrets.get("CROP_HEALTH_API_KEY") if "CROP_HEALTH_API_KEY" in st.secrets else None)

    if not api_key:
        return {"error": "Crop Health API Key missing"}

    try:
        # Resize for API optimization
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)
            
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('ascii')
        
        url = "https://crop.kindwise.com/api/v1/identification"
        headers = {
            "Api-Key": api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "images": [img_base64],
            "latitude": 49.195,
            "longitude": 16.606,
            "similar_images": True,
            "health": "all"
        }
        
        # Increased timeout to 60s for slow uplinks
        # SSL Verification bypass for Windows environment compatibility
        response = requests.post(url, headers=headers, json=payload, timeout=60, verify=False)
        
        if response.status_code != 200:
            return {"error": f"Kindwise Gateway Rejected (HTTP {response.status_code}): {response.text[:100]}"}
            
        data = response.json()
        result = data.get("result", {})
        
        disease_data = result.get("disease", {})
        suggestions = disease_data.get("suggestions", [])
        
        crop_data = result.get("crop", {})
        crop_suggs = crop_data.get("suggestions", [])
        crop_name = crop_suggs[0].get("name") if crop_suggs else None
        
        if not suggestions:
            is_healthy = result.get("is_healthy", {}).get("binary", True)
            if is_healthy:
                return {"disease": "Healthy Specimen", "plant": crop_name, "probability": 100.0, "description": "Specimen exhibits high vitality with no detectable pathogens."}
            return {"error": "Pathogen Matrix inconclusive. Diagnostic scan required.", "plant": crop_name}
            
        # Pathogen-First Logic: If 'Healthy' is suggested but other pathogens are present,
        # prefer the pathogen if it has >15% probability.
        best = suggestions[0]
        if best.get('name', '').lower() == "healthy" and len(suggestions) > 1:
            alt = suggestions[1]
            if alt.get('probability', 0) > 0.15:
                best = alt

        return {
            "disease": best.get("name"),
            "plant": crop_name,
            "probability": round(best.get("probability", 0) * 100, 1),
            "details": best.get("details", {}),
            "treatment": best.get("details", {}).get("treatment", {}),
            "description": best.get("details", {}).get("description") or "Pathogen signature localized. Consult remediation directives."
        }
    except Exception as e:
        return {"error": f"Crop Health Linkage Failure: {str(e)}"}

def get_perenual_care_info(common_name: str) -> dict:
    """Fetch additional care info using Perenual API."""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.getenv("PERENUAL_API_KEY")
    
    if not api_key:
        return {}
    
    url = f"https://perenual.com/api/species-list?key={api_key}&q={common_name}"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('data'):
            return data['data'][0]
    except:
        pass
    return {}

def identify_disease_with_plantnet(img_bgr: np.ndarray, api_key: str = None) -> dict:
    """Identify diseases using Pl@ntNet Diseases API (v2)."""
    if not api_key:
        api_key = os.getenv("PLANTNET_API_KEY")
    
    if not api_key:
        return {"error": "Pl@ntNet API Key missing"}
        
    try:
        # Resize for API optimization
        h, w = img_bgr.shape[:2]
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img_bgr = cv2.resize(img_bgr, (0, 0), fx=scale, fy=scale)
            
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        files = [('images', ('image.jpg', img_encoded.tobytes(), 'image/jpeg'))]
        data = {'organs': 'auto'}
        
        url = f"https://my-api.plantnet.org/v2/diseases/identify?api-key={api_key}"
        response = requests.post(url, files=files, data=data, timeout=20)
        
        if response.status_code == 200:
            res = response.json()
            results = res.get('results', [])
            if results:
                best = results[0]
                disease_name = best.get('disease', {}).get('scientificNameWithoutAuthor') or best.get('disease', {}).get('scientificName')
                common_names = best.get('disease', {}).get('commonNames', [])
                
                return {
                    "disease": common_names[0] if common_names else disease_name,
                    "probability": round(best.get('score', 0) * 100, 1),
                    "description": f"Diagnosis via Pl@ntNet Disease Matrix. EPPO: {best.get('disease', {}).get('eppoCode', 'N/A')}",
                    "details": best.get('disease', {})
                }
            else:
                return {"error": "Pl@ntNet: No pathological vectors identified in specimen."}
        else:
            return {"error": f"Pl@ntNet Disease API Rejected: HTTP {response.status_code}"}
    except Exception as e:
        return {"error": f"Pl@ntNet Disease Root Failure: {str(e)}"}

def remap_disease_with_nyckel(text: str, function_id: str, client_id: str = None, client_secret: str = None) -> str:
    """Remap a disease name to its botanical equivalent using Nyckel Text Classification."""
    if not function_id:
        return text
        
    try:
        # Step 1: Get OAuth Token (If credentials provided)
        auth_url = "https://www.nyckel.com/connect/token"
        token = None
        if client_id and client_secret:
            auth_res = requests.post(auth_url, data={
                'client_id': client_id,
                'client_secret': client_secret,
                'grant_type': 'client_credentials'
            }, timeout=10)
            if auth_res.status_code == 200:
                token = auth_res.json().get('access_token')

        # Step 2: Invoke Function
        invoke_url = f"https://www.nyckel.com/v1/functions/{function_id}/invoke"
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        
        # Nyckel Expects: {"data": "text to classify"}
        response = requests.post(invoke_url, headers=headers, json={"data": text}, timeout=10)
        
        if response.status_code == 200:
            res = response.json()
            # Nyckel returns labelName for the top prediction
            return res.get('labelName', text)
    except:
        pass
    return text

def identify_disease_with_huggingface(img_bgr: np.ndarray, api_key: str = None, verify_ssl: bool = False) -> dict:
    """Identify plant diseases using a free Hugging Face Inference API model with retry logic."""
    if not api_key:
        api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    if not api_key or "your_token" in api_key:
        return {"error": "Hugging Face API Key missing. Please check your .env file."}

    # Model: Sartaj/plant-disease-classification (Trained on PlantVillage)
    API_URL = "https://api-inference.huggingface.co/models/Sartaj/plant-disease-classification"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        # Encode image to bytes
        _, img_encoded = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_bytes = img_encoded.tobytes()

        # Send to Hugging Face with Retry Loop (for 'Waking Up' models)
        import time
        for attempt in range(3):
            response = requests.post(API_URL, headers=headers, data=img_bytes, timeout=30, verify=verify_ssl)
            
            if response.status_code == 200:
                results = response.json()
                if results and isinstance(results, list):
                    best = results[0]
                    label = best.get("label", "Unknown")
                    score = round(best.get("score", 0) * 100, 1)
                    try:
                        if "___" in label:
                            plant_name, disease_name = label.split("___")
                            disease_name = disease_name.replace("_", " ").title()
                            plant_name = plant_name.replace("_", " ").title()
                        else:
                            plant_name, disease_name = "Specimen", label.replace("_", " ").title()
                    except:
                        plant_name, disease_name = "Specimen", label
                    if "healthy" in disease_name.lower():
                        disease_name = "Healthy Specimen"
                    return {
                        "disease": disease_name, "plant": plant_name, "probability": score,
                        "description": f"Diagnosis via Hugging Face Neural Mesh (Sartaj-V5). Model Confidence: {score}%.",
                        "source": "Hugging Face"
                    }
            elif response.status_code == 503:
                # Model is loading - wait and retry
                time.sleep(10)
                continue
            elif response.status_code == 401:
                return {"error": "Hugging Face Token Rejected (401). Please check your Read permissions."}
            else:
                return {"error": f"Hugging Face API Error {response.status_code}: {response.text[:100]}"}
        
        return {"error": "Hugging Face Model timed out while waking up. Please try again in 30 seconds."}
    except Exception as e:
        return {"error": f"Hugging Face Linkage Failure: {str(e)}"}
