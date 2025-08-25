import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd



PUSHUP_THRESH = {
    "min_depth": 0.25, # shoulder_y_bottom - shoulder_y_top (down is +)
    "hip_sag": 0.2, # pelvis_y_bottom - shoulder_y_bottom at bottom (> means sag)
    "elbow_flare_side_max": 7.0, # max(left,right) normalized by shoulder width
}

def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based per-rep faults from rep_features.csv")
    parser.add_argument("--rep_csv", type=Path, required=True, help="Path to rep_features CSV file")
    parser.add_argument("--exercise", type=str, default="pushup", required=True, help="Exercise type")
    parser.add_argument("--out_json", type=Path, required=True, help="Where to save faults JSON file")
    parser.add_argument("--out_flags_csv", type=Path, help="Optional: export boolean fault flags CSV")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Minimum confidence score (0-1) to include a fault in JSON (default: 0)")
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
            "cue": f"Keep {side} elbow closer and aim ~45 degrees from torso.",
            "meta": {"left": l_flare, "right": r_flare, "side_max": side_max}
        })

    return faults

def main():
    args = parse_args()

    df = pd.read_csv(args.rep_csv)
    exercise = args.exercise

    th = PUSHUP_THRESH
    
    out = {
        "exercise": exercise,
        "thresholds": th,
        "num_reps": int(df.shape[0]),
        "reps": []
    }

    # Collect boolean flags
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
    
        faults = pushup_rules_row(row, th)

        # Filter by min confidence if requested
        faults = [f for f in faults if float(f.get("score", 1.0)) >= args.min_conf]
        rep_entry["faults"] = faults
        out["reps"].append(rep_entry)

        flag = {
            "rep_index": rep_idx,
        }
        # Initialize all known flags to 0
        for k in ["insufficient_depth", "hip_sag", "elbow_flare"]:
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