import json, subprocess as sp, sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

SCRIPTS = Path("scripts")
MODELS  = Path("outputs/model")   

FAULTS = ["insufficient_depth","hip_sag","elbow_flare"]

def run(cmd: list[str]) -> str:
    proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    if proc.returncode != 0:
        st.error(proc.stdout)
        raise RuntimeError("Command failed")
    return proc.stdout

def pick_features(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = df.copy()
    # keep only columns present in training
    missing = [c for c in feat_cols if c not in X.columns]
    for m in missing: X[m] = np.nan
    return X[feat_cols]

def load_models(model_dir: Path):
    feat_cols = json.loads((model_dir/"feature_cols.json").read_text())
    models = {}
    for f in FAULTS:
        p = model_dir/f"model_{f}.joblib"
        if p.exists():
            models[f] = joblib.load(p)
    return feat_cols, models

def rule_of_thumb_text(rep_row: pd.Series, preds: dict) -> str:
    cues = []
    if preds.get("insufficient_depth",0.0) >= 0.5 or rep_row.get("pushup_depth",0.0) < 0.25:
        cues.append("Go lower: aim for shoulders to travel ~20% of torso height.")
    if preds.get("hip_sag",0.0) >= 0.5 or rep_row.get("hip_sag_at_bottom",0.0) > 0.20:
        cues.append("Brace your core: keep pelvis level with shoulders at the bottom.")
    if preds.get("elbow_flare",0.0) >= 0.5 or rep_row.get("elbow_flare_avg_norm",0.0) > 5.0:
        cues.append("Tuck elbows ~45 degrees from the torso, not straight out.")
    if not cues:
        return "Nice rep. Maintain this technique."
    return " - ".join(cues)

def llm_polish(feedback: str, conf: dict) -> str:
    """
    If you have `ollama` (e.g., `llama3.2:3b` pulled), rewrite feedback.
    Otherwise just return the original feedback.
    """
    try:
        import requests
        prompt = (
            "You are a skilled workout trainer who is giving advice on a pushup form. Rewrite the bullet feedback into 1–2 short sentences, "
            "second-person, actionable, friendly, avoid jargon. Don't say here is: just straight up to advice\n\n"
            f"Feedback bullets: {feedback}\n"
            f"Rep context (optional): {json.dumps(conf)}"
        )
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model":"llama3.2:3b","prompt":prompt,"stream":False}, timeout=8)
        if r.ok:
            return r.json().get("response", feedback).strip()
    except Exception:
        pass
    return feedback

def main():
    st.set_page_config(page_title="Workout Form Advisor — Demo", layout="wide")
    st.title("Workout Form Advisor — Push-up Demo")

    colL, colR = st.columns([1,1])
    with colL:
        up = st.file_uploader("Upload a push-up video", type=["mp4","mov","avi"])
        use_llm   = st.checkbox("Polish coaching with local LLM (Ollama)", value=False)
        if up: st.video(up)

    if not up:
        st.info("Upload a video to analyze.")
        return

    out_dir = Path("data/processed/demo"); out_dir.mkdir(parents=True, exist_ok=True)
    vid_path = out_dir / "input.mp4"
    with open(vid_path, "wb") as f: f.write(up.read())

    stem = vid_path.stem
    pose_npz  = out_dir / f"{stem}_pose.npz"
    feat_npz  = out_dir / f"{stem}_features.npz"
    reps_json = out_dir / f"{stem}_reps.json"
    rep_csv   = out_dir / f"{stem}_rep_features.csv"
    overlay   = out_dir / f"{stem}.overlay.mp4"

    st.write("**Running pose → features → reps → rep_features...**")
    run([sys.executable,"-u",str(SCRIPTS/"extract_pose.py"),
         "--input_video",str(vid_path),"--output_path",str(pose_npz),
         "--write_overlay","--savgol_window","7","--savgol_poly","2"])

    run([sys.executable,"-u",str(SCRIPTS/"compute_features.py"),
         "--pose_npz",str(pose_npz),"--output_path",str(feat_npz)])

    run([sys.executable,"-u",str(SCRIPTS/"segment_reps.py"),
         "--features_npz",str(feat_npz),"--output_json",str(reps_json),
         "--exercise","pushup","--signal","shoulder","--anchor","top",
         "--savgol_window","5","--savgol_poly","2",
         "--min_prom","0.02","--min_range","0.05",
         "--min_rep_seconds","0.45","--max_rep_seconds","6.0",
         "--plot_png",str(out_dir/f"{stem}_reps.png")])

    run([sys.executable,"-u",str(SCRIPTS/"rep_features.py"),
         "--features_npz",str(feat_npz),"--reps_json",str(reps_json),
         "--out_csv",str(rep_csv)])

    reps = pd.read_csv(rep_csv)
    reps["video_id"] = "demo"
    for col in ["insufficient_depth_rule","hip_sag_rule","elbow_flare_rule"]:
        if col not in reps.columns:
            reps[col] = 0
    st.success(f"Detected {len(reps)} reps.")

    feat_cols, models = load_models(MODELS)
    X_raw = pick_features(reps, feat_cols).to_numpy(np.float32)
    imputer = SimpleImputer(strategy = "median")
    X_raw = imputer.fit_transform(X_raw)
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)


    ml_probs = {}
    for f, clf in models.items():
        ml_probs[f] = clf.predict_proba(X)[:,1]

    # display results
    st.subheader("Per-rep faults (ML & rules)")
    show_cols = ["rep_index","duration_s","pushup_depth","hip_sag_at_bottom","elbow_flare_avg_norm"]
    st.dataframe(reps[show_cols].round(3))

    # coaching
    st.subheader("Coaching")
    for _, row in reps.iterrows():
        idx = int(row["rep_index"])
        preds = {f: float(row.get(f"_prob_ml_{f}", np.nan)) for f in FAULTS}
        bullets = rule_of_thumb_text(row, preds)
        polished = llm_polish(bullets, {"rep_index":idx,"ml_probs":preds}) if use_llm else bullets
        st.markdown(f"**Rep {idx}** — {polished}")

    # media
    with colR:
        if overlay.exists():
            st.video(str(overlay))
        st.image(str(out_dir/f"{stem}_reps.png"), caption="Signal + anchors")

if __name__ == "__main__":
    main()
