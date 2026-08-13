"""
Weld Defect Detection System — Streamlit Dashboard
Run with: streamlit run app_streamlit.py
"""
import os
import sys
import time
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Weld Defect Detection System", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0B0E14; color: #E8EDF5; }
[data-testid="stSidebar"] { background-color: #13181F; border-right: 1px solid #242C3A; }
.metric-card { background: #1A2030; border: 1px solid #242C3A; border-radius: 10px; padding: 16px; text-align: center; }
.metric-val { font-size: 28px; font-weight: 700; font-family: monospace; }
.metric-label { font-size: 11px; color: #6B7A94; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
.accept-badge { background: rgba(0,224,150,0.12); border: 1px solid rgba(0,224,150,0.3); color: #00E096; padding: 12px 24px; border-radius: 8px; font-size: 20px; font-weight: 700; text-align: center; }
.review-badge { background: rgba(255,179,68,0.12); border: 1px solid rgba(255,179,68,0.3); color: #FFB344; padding: 12px 24px; border-radius: 8px; font-size: 20px; font-weight: 700; text-align: center; }
.nodefect-badge { background: rgba(107,122,148,0.12); border: 1px solid rgba(107,122,148,0.3); color: #6B7A94; padding: 12px 24px; border-radius: 8px; font-size: 20px; font-weight: 700; text-align: center; }
.rejected-badge { background: rgba(255,77,106,0.12); border: 1px solid rgba(255,77,106,0.3); color: #FF4D6A; padding: 12px 24px; border-radius: 8px; font-size: 20px; font-weight: 700; text-align: center; }
.stage-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px; border-radius: 8px; border: 1px solid #242C3A; margin-bottom: 6px; background: rgba(255,255,255,0.02); }
div[data-testid="stImage"] img { border-radius: 8px; border: 1px solid #242C3A; }
</style>
""", unsafe_allow_html=True)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = os.path.join(BASE_DIR, "evaluation", "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))

@st.cache_resource(show_spinner=False)
def load_pipeline():
    from pipeline.cascade import WeldDefectPipeline
    return WeldDefectPipeline()

if "session_counts" not in st.session_state:
    st.session_state.session_counts = {"ACCEPT": 0, "REVIEW": 0, "NO_DEFECT": 0, "REJECTED_NON_WELD": 0}
if "history" not in st.session_state: st.session_state.history = []
if "current_filename" not in st.session_state: st.session_state["current_filename"] = None

with st.sidebar:
    st.markdown("## 🔬 Weld Defect Detection")
    st.markdown("**Industrial X-ray Inspection System**")
    st.divider()

    st.markdown("### Pipeline Stages (v2 + XAI)")
    stages = [
        ("1", "EfficientNetB3", "Non-weld gate classifier"),
        ("2", "SegFormer v2", "Deep spatial segmentation"),
        ("3", "Post-processing", "Morphological ops + filter"),
        ("4", "Routing & XAI", "Confidence check & LayerCAM"),
    ]
    for num, name, detail in stages:
        st.markdown(f"""
        <div class="stage-row"><span style="font-family:monospace;font-size:11px;color:#6B7A94;width:28px;text-align:center">{num}</span>
        <div><div style="font-size:12px;font-weight:600">{name}</div><div style="font-size:10px;color:#6B7A94">{detail}</div></div></div>
        """, unsafe_allow_html=True)

    st.divider()
    counts = st.session_state.session_counts
    total  = sum(counts.values())
    st.markdown(f"**Total inspected:** {total}")
    if total > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("✅ Accept",  counts["ACCEPT"])
            st.metric("⚪ No Defect", counts["NO_DEFECT"])
        with col2:
            st.metric("⚠️ Review",  counts["REVIEW"])
            st.metric("❌ Rejected", counts["REJECTED_NON_WELD"])
    if st.button("🗑 Reset Session & Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.session_counts = {"ACCEPT": 0, "REVIEW": 0, "NO_DEFECT": 0, "REJECTED_NON_WELD": 0}
        st.session_state.history = []
        st.session_state["current_filename"] = None
        st.rerun()

st.markdown("# 🔬 Weld Defect Detection System")
st.markdown("Upload an X-ray weld image to run the streamlined 4-stage inspection pipeline.")
st.divider()

uploaded_file = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    if st.session_state["current_filename"] != uploaded_file.name:
        st.session_state["current_filename"] = None

    col_prev, col_info = st.columns([1, 2])
    with col_prev:
        st.image(Image.open(uploaded_file).convert("RGB"), caption="Uploaded image", use_container_width=True)
    with col_info:
        st.markdown(f"**File:** `{uploaded_file.name}`\n\nClick **Run Inspection** to process through the full pipeline.")

    if st.button("▶ Run Inspection", type="primary", use_container_width=True):
        tmp_path = os.path.join(OUTPUT_DIR, f"_tmp_{uploaded_file.name}")
        with open(tmp_path, "wb") as f: f.write(uploaded_file.getvalue())

        with st.spinner("Loading pipeline models..."): pipeline = load_pipeline()
        
        progress = st.progress(0, text="Starting pipeline...")
        start_time = time.time()
        
        try:
            result = pipeline.predict(tmp_path)
            pipeline._save_annotated(tmp_path, result, OUTPUT_DIR, uploaded_file.name)
            progress.progress(1.0, text="Complete ✓")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

        elapsed = round(time.time() - start_time, 2)
        if os.path.exists(tmp_path): os.remove(tmp_path)

        st.session_state.session_counts[result["status"]] += 1
        st.session_state.history.append({"File": uploaded_file.name, "Status": result["status"], "Confidence": result["confidence"], "Defect Px": result["defect_area"], "Time (s)": elapsed})
        
        st.session_state["current_filename"] = uploaded_file.name
        st.session_state["current_result"] = result
        st.session_state["current_elapsed"] = elapsed

    if st.session_state.get("current_filename") == uploaded_file.name:
        result = st.session_state["current_result"]
        st.divider()

        status_map = {"ACCEPT": ("accept-badge", "✓ ACCEPT"), "REVIEW": ("review-badge", "⚠ NEEDS REVIEW"), "NO_DEFECT": ("nodefect-badge", "○ NO DEFECT FOUND"), "REJECTED_NON_WELD": ("rejected-badge", "✕ REJECTED")}
        cls, label = status_map.get(result["status"], ("nodefect-badge", result["status"]))
        st.markdown(f'<div class="{cls}">{label}</div>', unsafe_allow_html=True)
        st.markdown("")

        m1, m2, m3 = st.columns(3)
        with m1: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#00C8FF">{result["confidence"]*100:.1f}%</div><div class="metric-label">Region Confidence</div></div>', unsafe_allow_html=True)
        with m2: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#7B5CF0">{result["defect_area"]:,}</div><div class="metric-label">Defect Pixels</div></div>', unsafe_allow_html=True)
        with m3: st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:#FFB344">{st.session_state["current_elapsed"]}s</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)

        st.markdown("\n### Output Images")
        
        # ── RAM BYPASS: Loading images directly from computer memory ──
        img_bbox    = result.get("img_bbox")
        img_overlay = result.get("img_overlay")
        img_mask    = result.get("img_mask")
        img_xai     = result.get("img_xai")

        col1, col2 = st.columns(2)
        with col1:
            if img_bbox is not None: st.image(img_bbox, caption="Bounding Boxes (per defect blob)", use_container_width=True)
            if img_mask is not None: st.image(img_mask, caption="Binary Defect Mask", use_container_width=True)
        with col2:
            if img_overlay is not None: st.image(img_overlay, caption="Defect Overlay (red = defect)", use_container_width=True)
            if img_xai is not None: st.image(img_xai, caption="GradCam++ XAI Heatmap ", use_container_width=True)

if st.session_state.history:
    st.divider()
    st.markdown("### Inspection History")
    df_hist = pd.DataFrame(st.session_state.history)
    df_hist["Confidence"] = df_hist["Confidence"].apply(lambda x: f"{x*100:.1f}%")
    def color_status(val):
        return {"ACCEPT": "background-color:#00E09620;color:#00E096", "REVIEW": "background-color:#FFB34420;color:#FFB344", "REJECTED_NON_WELD": "background-color:#FF4D6A20;color:#FF4D6A"}.get(val, "")
    st.dataframe(df_hist.style.applymap(color_status, subset=["Status"]), use_container_width=True, hide_index=True)