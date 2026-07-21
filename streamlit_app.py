import os
import urllib.request

import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import RTDETR

st.set_page_config(page_title="CarDD Vehicle Damage Detector",
                   page_icon="🚗", layout="wide")

# ---------------------------------------------------------------------------
# The trained RT-DETR weights (66 MB) are too big for GitHub's normal file
# storage, so they live in a GitHub *Release* and are downloaded on first run.
# ---------------------------------------------------------------------------
MODEL_URL = "https://github.com/jamuna-cyber/cardd-damage-detector/releases/download/ModelWeights/best.pt"
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only)…"):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return RTDETR(MODEL_PATH)   # trained RT-DETR weights

model = load_model()
NAMES = model.names
IMGSZ = 640

SEVERITY_W = {"dent": 2, "scratch": 1, "crack": 3,
              "glass shatter": 4, "lamp broken": 3, "tire flat": 3}

st.title("🚗 CarDD Vehicle Damage Detector — RT-DETR")
st.caption("Upload a photo of vehicle damage. Detects dents, scratches, cracks, "
           "glass shatter, broken lamps and flat tires, and estimates repair "
           "severity. MSc project demo — University of Greenwich.")

conf = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
up = st.file_uploader("Upload a car photo", type=["jpg", "jpeg", "png"])

if up is None:
    st.info("👆 Upload a photo of a damaged car to begin.")
else:
    img = Image.open(up).convert("RGB")
    with st.spinner("Analysing damage..."):
        r = model.predict(img, imgsz=IMGSZ, conf=conf,
                          verbose=False, device="cpu")[0]
    annotated = r.plot()[..., ::-1]   # BGR -> RGB

    c1, c2 = st.columns(2)
    c1.image(img, caption="Original")
    c2.image(annotated, caption="Detected damage")

    H, W = r.orig_shape
    rows, score = [], 0.0
    for b in r.boxes:
        cls = NAMES[int(b.cls)]
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area_pct = 100 * (x2 - x1) * (y2 - y1) / (W * H)
        rows.append({"Damage type": cls,
                     "Confidence": round(float(b.conf), 3),
                     "Area % of image": round(area_pct, 2)})
        score += SEVERITY_W.get(cls, 2) * min(area_pct, 25) / 25

    if not rows:
        st.warning("No damage detected above the confidence threshold — "
                   "try lowering the slider.")
    else:
        n = len(rows)
        if score < 1.5:
            st.success(f"MINOR damage — cosmetic repair, vehicle drivable. "
                       f"({n} region(s) found)")
        elif score < 4:
            st.warning(f"MODERATE damage — garage assessment recommended. "
                       f"({n} region(s) found)")
        else:
            st.error(f"SEVERE damage — professional inspection required "
                     f"before driving. ({n} region(s) found)")
        st.dataframe(pd.DataFrame(rows).sort_values("Confidence",
                     ascending=False), hide_index=True)
  
     
