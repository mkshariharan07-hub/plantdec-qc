---
license: mit
library_name: keras
tags:
- image-classification
- plant-disease
- efficientnetv2
- tensorflow
- tflite
- plantvillage
- computer-vision
- transfer-learning
datasets:
- plant_village
metrics:
- accuracy
- f1
pipeline_tag: image-classification
base_model: google/efficientnet-v2
---

# 🌿 Plant Disease Detection — EfficientNetV2S

Automated plant disease classification trained on the
[PlantVillage benchmark](https://arxiv.org/abs/1511.08060)
(38 classes, 54,306 images).

**Live demo:** [🤗 HuggingFace Space](https://huggingface.co/spaces/animeshakr/plant-disease-detection1)  
**GitHub:** [Animesh-Kr/Plant-Disease-Prediction](https://github.com/Animesh-Kr/Plant-Disease-Prediction)

---

## Model Details

| Property | Value |
|---|---|
| Architecture | EfficientNetV2S |
| Input resolution | 384 × 384 |
| Number of classes | 38 |
| Dataset | PlantVillage (Hughes & Salathé, 2015) |
| Test accuracy | 99.57% |
| Macro F1 | 99.48% |
| Top-3 accuracy | 99.98% |
| Parameters | ~21M |
| Training framework | TensorFlow / Keras 3 |

---

## Evaluation Results

| Model | Test Accuracy | Macro F1 | Weighted F1 | Top-3 Accuracy | Mean Confidence |
|---|---|---|---|---|---|
| Baseline CNN (4-block, scratch) | 91.23% | 89.32% | 91.28% | 98.74% | 78.54% |
| **EfficientNetV2S (this model)** | **99.57%** | **99.48%** | **99.57%** | **99.98%** | **90.55%** |

## Interactive Visualisations

| Plot | Link |
|---|---|
| 3D UMAP Embeddings | [Open](https://animesh-kr.github.io/Plant-Disease-Prediction/umap_3d.html) |
| 3D Performance Surface | [Open](https://animesh-kr.github.io/Plant-Disease-Prediction/performance_surface_3d.html) |
| 3D Confusion Surface | [Open](https://animesh-kr.github.io/Plant-Disease-Prediction/confusion_3d.html) |

### Statistical Significance
- **McNemar's test:** p = 3.27 × 10⁻¹⁸² — improvement is astronomically significant
- **Expected Calibration Error (ECE):** 0.0902 — moderately well calibrated
- **MC Dropout uncertainty** (30 passes): flagged images are ~17 percentage points less accurate than unflagged, confirming the uncertainty flag is meaningful

---

## Training Pipeline

### Data
- **Dataset:** PlantVillage color subset — 54,306 images, 38 disease classes
- **Split:** Family-aware 70/15/15 train/val/test using perceptual hash deduplication
- **Deduplication:** pHash + Union-Find clustering prevents near-duplicate leakage
- **Augmentation:** RandomFlip, RandomRotation(0.08), RandomZoom(0.12), RandomTranslation(0.08), RandomContrast(0.10)

### Architecture
```
Input (384×384×3)
    ↓
Augmentation layers
    ↓
EfficientNetV2S backbone (ImageNet pretrained, include_preprocessing=True)
    ↓  ← top 40% unfrozen during fine-tune stage
GlobalAveragePooling2D
    ↓
Dropout(0.3) → Dense(512, swish) → Dropout(0.3)
    ↓
Dense(38, softmax)
```

### Training Strategy
- **Stage 1 (Warmup):** Backbone frozen, head only, lr=2e-3, 8 epochs
- **Stage 2 (Fine-tune):** Top 40% backbone unfrozen, lr=2e-5, 15 epochs
- **Loss:** CategoricalCrossEntropy + label smoothing 0.1
- **Optimizer:** AdamW (weight decay 1e-4)
- **Precision:** Mixed float16
- **Batch size:** 64 @ 384×384 (A100 80 GB)
- **Hardware:** Google Colab Pro+ A100 80 GB, 167 GB RAM

---

## Supported Classes (38 total)

| Plant | Diseases |
|---|---|
| Apple | Apple scab, Black rot, Cedar apple rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery mildew, Healthy |
| Corn | Cercospora leaf spot, Common rust, Northern leaf blight, Healthy |
| Grape | Black rot, Esca (Black Measles), Leaf blight, Healthy |
| Orange | Haunglongbing (Citrus greening) |
| Peach | Bacterial spot, Healthy |
| Pepper | Bacterial spot, Healthy |
| Potato | Early blight, Late blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery mildew |
| Strawberry | Leaf scorch, Healthy |
| Tomato | Bacterial spot, Early blight, Late blight, Leaf mold, Septoria leaf spot, Spider mites, Target spot, Yellow leaf curl virus, Mosaic virus, Healthy |

---

## Usage

### Option A — TFLite float16 (recommended, fast)

```python
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Load model
interp = tf.lite.Interpreter(model_path="model_float16_quant.tflite")
interp.allocate_tensors()
ind  = interp.get_input_details()[0]
outd = interp.get_output_details()[0]

# Load labels
with open("class_indices.json") as f:
    class_map = json.load(f)   # {"0": "Apple___Apple_scab", ...}

def predict(img_path: str, top_k: int = 5):
    img = Image.open(img_path).convert("RGB").resize((384, 384))
    arr = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    interp.set_tensor(ind["index"], arr)
    interp.invoke()
    probs = interp.get_tensor(outd["index"])[0]
    top   = np.argsort(probs)[-top_k:][::-1]
    return [(class_map[str(i)], float(probs[i])) for i in top]

results = predict("leaf.jpg")
for cls, conf in results:
    print(f"{cls}: {conf*100:.1f}%")
```

### Option B — Full Keras model

```python
import keras
import numpy as np
import json
from PIL import Image

model     = keras.models.load_model("best_model.keras")
class_map = json.load(open("class_indices.json"))

def predict(img_path: str, top_k: int = 5):
    img  = Image.open(img_path).convert("RGB").resize((384, 384))
    arr  = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    prob = model.predict(arr, verbose=0)[0]
    top  = np.argsort(prob)[-top_k:][::-1]
    return [(class_map[str(i)], float(prob[i])) for i in top]
```

---

## Files

| File | Size | Description |
|---|---|---|
| `model_float16_quant.tflite` | ~45 MB | TFLite float16 — for deployment |
| `class_indices.json` | 2 KB | Label mapping `{"0": "Apple___Apple_scab", ...}` |

---

## Limitations

- Trained on controlled laboratory photographs from PlantVillage only
- Generalisation to field photographs with occlusion, variable lighting, or soil contamination is **not validated**
- Not intended for commercial crop management decisions

---

## Citation

```bibtex
@misc{kumar2025plantdisease,
  author    = {Animesh Kumar},
  title     = {Plant Disease Detection using EfficientNetV2S},
  year      = {2025},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/animeshakr/plant-disease-efficientnetv2s}
}
```

## References

- Hughes & Salathé (2015). An open access repository of images on plant health. *arXiv:1511.08060*
- Tan & Le (2021). EfficientNetV2: Smaller Models and Faster Training. *ICML 2021*
- Gal & Ghahramani (2016). Dropout as a Bayesian Approximation. *ICML 2016*
- Dietterich (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Computation*

---

**Developed by [Animesh Kumar](https://github.com/Animesh-Kr)**