import json
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import streamlit as st
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# BlazePose indices
L_HIP, R_HIP   = 23, 24
L_SHO, R_SHO   = 11, 12
L_KNEE, R_KNEE = 25, 26

def load_features(npz_path):
    d = np.load(npz_path)
    ts = d["timestamps"].astype(np.float32)
    xy = d["xy_norm"].astype(np.float32)
    
    return ts, xy, d.files

def load_segments(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def make_signal(xy, bodyPart):
    if bodyPart == "pelvis":
        y = 0.5 * (xy[:, L_HIP, 1] + xy[:, R_HIP, 1])
        return y, "pelvis_y (mid-hips)"
    if bodyPart == "shoulder":
        y = 0.5 * (xy[:, L_SHO, 1] + xy[:, R_SHO, 1])
        return y, "shoulder_y (mid-shoulders)"
    if bodyPart == "knee":
        y = 0.5 * (xy[:, L_KNEE, 1] + xy[:, R_KNEE, 1])
        return y, "knee_y (mid-knees)"
    raise ValueError("Unknown signal bodyPart")

def smooth(x, window, poly):
    if window <= 0 or x.shape[0] < max(window, poly + 2) or (window % 2 == 0) or poly >= window:
        return x
    return savgol_filter(x, window_length=window, polyorder=poly, mode="interp").astype(np.float32)

def open_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, (0.0, 0, 0.0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / float(fps) if fps > 0 and total > 0 else 0.0
    return cap, (float(fps), total, float(duration))

def frame_at_time(cap, t, wmax):
    if cap is None:
        return None
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    h, w = rgb.shape[:2]
    if w > wmax:
        scale = wmax / float(w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation = cv2.INTER_AREA)
    
    return rgb

def main():
    
    st.set_page_config(page_title="Workout From Viewer", layout="wide")
    
    st.sidebar.title("Inputs")
    
    features_path = st.sidebar.text_input("Features .npz", "data/processed/test_pose_features.npz")
    reps_json_path = st.sidebar.text_input("Reps JSON", "data/processed/test_pose_reps.json")
    overlay_video_path = st.sidebar.text_input("Overlay video (.mp4/.avi)", "data/processed/test_pose.overlay.mp4")
    
    signal_bodyPart = st.sidebar.selectbox("Signal", ["shoulder", "pelvis", "knee"], index = 0)
    sg_window = st.sidebar.number_input("Savgol window (odd, 0 = off)", min_value = 0, value = 7, step = 2)
    sg_poly = st.sidebar.number_input("Savgol poly (< window)", min_value = 0, value = 2, step = 1)
    
    fpath = Path(features_path)
    jpath = Path(reps_json_path)
    vpath = Path(overlay_video_path)
    
    if not fpath.exists():
        st.error(f"Features file not found: {fpath}")
        st.stop()
    if not jpath.exists():
        st.warning(f"Reps JSON not found: {jpath} (you can still browse features)")
        
    ts, xy, keys = load_features(fpath)
    y_raw, y_desc = make_signal(xy, signal_bodyPart)
    y_s = smooth(y_raw, sg_window, sg_poly)
    
    st.sidebar.write(f"Loaded T={len(ts)} frames. Keys: {', '.join(list(keys))}")
    
    segs = []
    if jpath.exists():
        meta = load_segments(jpath)
        segs = meta.get("rep_segments", [])
        st.sidebar.success(f"Loaded {len(segs)} rep segments.")
        
    left, right = st.columns([2, 1], gap = "large")
    
    with left:
        st.subheader("Signal plot")
        fig = plt.figure(figsize = (10, 4))
        plt.plot(ts, y_raw, label = f"{y_desc} (raw)")
        if sg_window > 0:
            plt.plot(ts, y_s, label = f"{y_desc} (smoothed)")
        
        for seg in segs:
            plt.axvspan(seg["start_time"], seg["end_time"], alpha = 0.15)
        
        plt.xlabel("time (s)")
        plt.ylabel("normalized y (down is +)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width = True)
        
        if segs:
            st.markdown("### Reps")

            for seg in segs:
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
                c1.write(f"#{seg['rep_index']}")
                c2.write(f"{seg['start_time']:.2f}s -> {seg['end_time']:.2f}s")
                c3.write(f"dur: {seg['duration']:.2f}s")
                c4.write(f"range: {seg.get('range', float('nan')):.3f}")
                jump = c5.button("Jump", key=f"jump_{seg['rep_index']}")
                if jump:
                    st.session_state["jump_time"] = float(seg["start_time"])
    
    with right:
        st.subheader("Video / Frame Preview")
        
        cap, (fps, total_frames, duration) = open_video(vpath) if vpath.exists() else (None, (0.0, 0, 0.0))
        if cap is None:
            st.warning("Overlay video not found or failed to open.")
        else:
            st.caption(f"Video: {vpath.name} {duration:.2f}s @ {fps:.1f} FPS")
            
        t0 = st.session_state.get("jump_time", 0.0)
        t_sel = st.slider("Preview time (s)", min_value = 0.0, max_value = max(duration, ts[-1]), value = float(t0), step = 0.01)
        
        if cap is not None:
            img = frame_at_time(cap, t_sel, wmax= 768)
            if img is not None:
                st.image(img, caption = f"t = {t_sel:.2f}s", use_container_width = True)
            else:
                st.error("Could not read frame at that time")
        
        if vpath.exists():
            st.video(str(vpath))
            
        if "jump_time" in st.session_state:
            st.session_state.pop("jump_time", None)
            
if __name__ == "__main__":
    main()
                
        
        