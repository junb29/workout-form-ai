import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd



PUSHUP_THRESH = {
    "min_depth": 0.20,            # shoulder_y_bottom - shoulder_y_top (down is +)
    "hip_sag": 0.05,              # pelvis_y_bottom - shoulder_y_bottom at bottom (> means sag)
    "elbow_flare_side_max": 0.80, # max(left,right) normalized by shoulder width
    "elbow_flare_avg": 0.55,      # optional backup rule on bilateral flare
}

SQUAT_THRESH = {
    "knee_valgus_max_norm": 0.12,  # max(|knee.x - ankle.x| / hip_width) at bottom
    "trunk_tilt_max_rad": 0.60,    # trunk angle vs vertical at bottom
    # Depth: if you later export knee angle at bottom as a single column, add it here.
    # For now, we can approximate depth with hip_y_bottom if you want a rule:
    "min_hip_y_bottom": None,      # e.g., 0.30 (larger y means lower hips in your coords)
}

def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based per-rep faults from rep_features.csv")
    parser.add_argument("--rep_csv", type=Path, required=True, help="Path to rep_features CSV file")
    parser.add_argument("--exercise", type=str, choices=["pushup", "squat"], required=True, help="Exercise type")
    parser.add_argument("--out_json", type=Path, required=True, help="Where to save faults JSON file")
    parser.add_argument("--out_flags_csv", type=Path, help="Optional: export boolean fault flags CSV")
    parser.add_argument("--min_conf", type=float, default=0.0,
                   help="Minimum confidence score (0-1) to include a fault in JSON (default: 0)")
    return parser.parse_args()

def pushup_rules_row(row, thresh):
    faults = []

    # Depth: require shoulder travel >= min_depth
    depth = float(row["pushup_depth"])
    if depth < thresh["min_depth"]:
        deficit = thresh["min_depth"] - depth
        score = min(1.0, deficit / max(thresh["min_depth"], 1e-6))
        faults.append({
            "name": "insufficient_depth",
            "score": score,
            "cue": "Lower your chest closer to the floor.",
            "meta": {"depth": depth, "min_required": thresh["min_depth"]}
        })

    # Hip sag: pelvis lower than shoulders at bottom
    sag = float(row["hip_sag_at_bottom"])
    if sag > thresh["hip_sag"]:
        score = min(1.0, (sag - thresh["hip_sag"]) / max(thresh["hip_sag"], 1e-6))
        faults.append({
            "name": "hip_sag",
            "score": score,
            "cue": "Brace your core to keep hips level with shoulders at the bottom.",
            "meta": {"sag": sag, "threshold": thresh["hip_sag"]}
        })

    # Elbow flare: prefer side-max so asymmetry isn't hidden
    l_flare = float(row["elbow_flare_left_norm"])
    r_flare = float(row["elbow_flare_right_norm"])
    side_max = max(l_flare, r_flare)
    if side_max > thresh["elbow_flare_side_max"]:
        side = "left" if l_flare >= r_flare else "right"
        score = min(1.0, (side_max - thresh["elbow_flare_side_max"]) / max(thresh["elbow_flare_side_max"], 1e-6))
        faults.append({
            "name": "elbow_flare",
            "score": score,
            "cue": f"Keep {side} elbow closer—aim ~45° from torso.",
            "meta": {"left": l_flare, "right": r_flare, "side_max": side_max}
        })
    else:
        # Optional: bilateral average rule (weaker)
        avg_flare = float(row["elbow_flare_avg_norm"])
        if avg_flare > thresh["elbow_flare_avg"]:
            score = min(1.0, (avg_flare - thresh["elbow_flare_avg"]) / max(thresh["elbow_flare_avg"], 1e-6))
            faults.append({
                "name": "elbow_flare_bilateral",
                "score": score,
                "cue": "Tuck elbows slightly—aim ~45° from torso.",
                "meta": {"avg": avg_flare}
            })

    return faults

def squat_rules_row(row, thresh):
    faults = []

    # Knee valgus at bottom (normalized by hip width)
    if "knee_valgus_max_norm" in row and not pd.isna(row["knee_valgus_max_norm"]):
        v_max = float(row["knee_valgus_max_norm"])
        if v_max > thresh["knee_valgus_max_norm"]:
            score = min(1.0, (v_max - thresh["knee_valgus_max_norm"]) / max(thresh["knee_valgus_max_norm"], 1e-6))
            faults.append({
                "name": "knee_valgus",
                "score": score,
                "cue": "Press knees out over mid-foot during the descent and ascent.",
                "meta": {"valgus_max_norm": v_max}
            })

    # Torso collapse at bottom (trunk tilt vs vertical)
    if "trunk_angle_at_bottom_rad" in row and not pd.isna(row["trunk_angle_at_bottom_rad"]):
        trunk_btm = float(row["trunk_angle_at_bottom_rad"])
        if trunk_btm > thresh["trunk_tilt_max_rad"]:
            score = min(1.0, (trunk_btm - thresh["trunk_tilt_max_rad"]) / max(thresh["trunk_tilt_max_rad"], 1e-6))
            faults.append({
                "name": "torso_collapse",
                "score": score,
                "cue": "Keep chest up; brace and sit between the hips.",
                "meta": {"trunk_bottom_rad": trunk_btm}
            })

    # Optional depth proxy using hip_y_bottom (bigger y => lower hips in your coords)
    if thresh.get("min_hip_y_bottom") is not None and "hip_y_bottom" in row and not pd.isna(row["hip_y_bottom"]):
        hip_y_btm = float(row["hip_y_bottom"])
        if hip_y_btm < float(thresh["min_hip_y_bottom"]):
            # If hip y at bottom is small, rep may be shallow (depending on your normalization)
            faults.append({
                "name": "insufficient_depth",
                "score": 1.0,
                "cue": "Sit deeper—aim for more hip/knee flexion while keeping heels down.",
                "meta": {"hip_y_bottom": hip_y_btm, "min_required": thresh["min_hip_y_bottom"]}
            })

    return faults

def main():
    args = parse_args()

    df = pd.read_csv(args.rep_csv)
    exercise = args.exercise

    if exercise == "pushup":
        th = PUSHUP_THRESH
    else:
        th = SQUAT_THRESH

    # Prepare JSON skeleton
    out = {
        "exercise": exercise,
        "thresholds": th,
        "num_reps": int(df.shape[0]),
        "reps": []
    }

    # Optional: collect boolean flags for CSV
    flags_rows = []

    for _, row in df.iterrows():
        rep_idx = int(row["rep_index"])
        rep_entry = {
            "rep_index": rep_idx,
            "start_frame": int(row["start_frame"]),
            "end_frame": int(row["end_frame"]),
            "start_time": float(row["start_time"]),
            "end_time": float(row["end_time"]),
            "faults": []
        }

        if exercise == "pushup":
            faults = pushup_rules_row(row, th)
        else:
            faults = squat_rules_row(row, th)

        # Filter by min confidence if requested
        faults = [f for f in faults if float(f.get("score", 1.0)) >= args.min_conf]
        rep_entry["faults"] = faults
        out["reps"].append(rep_entry)

        # Boolean flags row (for quick inspection / training)
        flag = {
            "rep_index": rep_idx,
            "start_time": rep_entry["start_time"],
            "end_time": rep_entry["end_time"],
        }
        # Initialize all known flags to 0
        for k in ["insufficient_depth", "hip_sag", "elbow_flare", "elbow_flare_bilateral",
                  "knee_valgus", "torso_collapse"]:
            flag[k] = 0
        for f in faults:
            name = f["name"]
            if name in flag:
                flag[name] = 1
        flags_rows.append(flag)

    # Write JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))
    print(f"Saved faults JSON to {args.out_json}")

    # Optional flags CSV
    if args.out_flags_csv:
        out_df = pd.DataFrame(flags_rows).sort_values("rep_index").reset_index(drop=True)
        args.out_flags_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(args.out_flags_csv, index=False)
        print(f"Saved fault flags CSV to {args.out_flags_csv}")
        
if __name__ == "__main__":
    main()