import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd

L_HIP, R_HIP   = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK   = 27, 28
L_SHO, R_SHO   = 11, 12
L_ELB, R_ELB   = 13, 14

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate per-frame features into per-rep features.")
    parser.add_argument("--features_npz", type=Path, required=True, help="Path to features.npz file")
    parser.add_argument("--reps_json", type=Path, required=True, help="Path to reps.json file from segment_reps.py")
    parser.add_argument("--out_csv", type=Path, required=True, help="Path to save rep_features.csv")
    return parser.parse_args()

def y_mean(xy, left, right):
    return 0.5 * (xy[:, left, 1] + xy[:, right, 1])

def get_y_signal(xy, bodyPart):
    if bodyPart == "pelvis":
        return 0.5 * (xy[:, L_HIP, 1] + xy[:, R_HIP, 1])
    if bodyPart == "shoulder":
        return 0.5 * (xy[:, L_SHO, 1] + xy[:, R_SHO, 1])
    if bodyPart == "knee":
        return 0.5 * (xy[:, L_KNEE, 1] + xy[:, R_KNEE, 1])
    raise ValueError(f"Unknown signal {bodyPart}")

def get_widths(xy):
    hip_width = np.abs(xy[:, R_HIP, 0] - xy[:, L_HIP, 0]) + 1e-6
    sho_width = np.abs(xy[:, R_SHO, 0] - xy[:, L_SHO, 0]) + 1e-6
    return {"hip": hip_width.astype(np.float32), "shoulder": sho_width.astype(np.float32)}

def parse_signal_key(signal_used):
    s = (signal_used or "").lower()
    if "shoulder" in s: 
        return "shoulder"
    if "pelvis" in s: 
        return "pelvis"
    if "knee" in s:
        return "knee"
    return "shoulder"

def main():
    args = parse_args()
    
    d = np.load(args.features_npz)
    ts = d["timestamps"].astype(np.float32)
    xy = d["xy_norm"].astype(np.float32)
    
    trunk = d["trunk_angle_rad"].astype(np.float32)
    lKnee = d["knee_angle_L_rad"].astype(np.float32)
    rKnee = d["knee_angle_R_rad"].astype(np.float32)
    lHip = d["hip_angle_L_rad"].astype(np.float32)
    rHip = d["hip_angle_R_rad"].astype(np.float32)
    
    meta = json.loads(Path(args.reps_json).read_text())
    segs = meta.get("rep_segments", [])
    if not segs:
        raise RuntimeError("No rep segments found in JSON.")

    signal_key = parse_signal_key(meta.get("signal", "shoulder"))
        
    anchor = meta.get("anchor", "top")
    
    y = get_y_signal(xy, signal_key)
    
    sho_y = y_mean(xy, L_SHO, R_SHO)
    hip_y = y_mean(xy, L_HIP, R_HIP)
    knee_y = y_mean(xy, L_KNEE, R_KNEE)
    
    widths = get_widths(xy)
    
    rows = []
    prev_end_time = None
    
    for seg in segs:
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        if end <= start or start < 0 or end > len(ts)-1:
            continue
        
        t_start, t_end = float(ts[start]), float(ts[end])
        duration = t_end - t_start
        y_window = y[start:end+1]
        y_min, y_max = float(y_window.min()), float(y_window.max())
        y_range = y_max - y_min
        
        # In image coordinates, y increases downward
        top_idx_rel = int(np.argmin(y_window))
        bottom_idx_rel = int(np.argmax(y_window))

        top_frame = start + top_idx_rel
        bottom_frame = start + bottom_idx_rel
        top_time, bottom_time = float(ts[top_frame]), float(ts[bottom_frame])
        
        trunk_at_bottom = float(trunk[bottom_frame])
        trunk_at_top = float(trunk[top_frame])
        
        lKnee_min = float(lKnee[start:end+1].min())
        lKnee_max = float(lKnee[start:end+1].max())
        rKnee_min = float(rKnee[start:end+1].min())
        rKnee_max = float(rKnee[start:end+1].max())
        lHip_min = float(lHip[start:end+1].min())
        lHip_max = float(lHip[start:end+1].max())
        rHip_min = float(rHip[start:end+1].min())
        rHip_max = float(rHip[start:end+1].max())
        
        # Ranges of motion
        lKnee_rom = lKnee_max - lKnee_min
        rKnee_rom = rKnee_max - rKnee_min
        lHip_rom = lHip_max - lHip_min
        rHip_rom = rHip_max - rHip_min
        
        # Body Symmetry
        knee_sym_at_bottom = float(abs(lKnee[bottom_frame] - rKnee[bottom_frame]))
        hip_sym_at_bottom = float(abs(lHip[bottom_frame] - rHip[bottom_frame]))
        knee_rom_diff = float(abs(lKnee_rom - rKnee_rom))
        hip_rom_diff = float(abs(lHip_rom - rHip_rom))
        
        # Pushup primitives
        shoulder_y_top = float(sho_y[top_frame])
        shoulder_y_bottom = float(sho_y[bottom_frame])
        pelvis_y_top = float(hip_y[top_frame])
        pelvis_y_bottom = float(hip_y[bottom_frame])
        pushup_depth = shoulder_y_bottom - shoulder_y_top
        hip_sag_at_bottom = pelvis_y_bottom - shoulder_y_bottom
        
        shoulder_width = float(max(widths["shoulder"][bottom_frame], 1e-6))
        l_elbow_flare = abs(xy[bottom_frame, L_ELB, 0] - xy[bottom_frame, L_SHO, 0]) / shoulder_width
        r_elbow_flare = abs(xy[bottom_frame, R_ELB, 0] - xy[bottom_frame, R_SHO, 0]) / shoulder_width
        elbow_flare_avg = 0.5 * (float(l_elbow_flare) + float(r_elbow_flare))     
        
        # Squat primitives
        squat_hip_y_bottom = float(hip_y[bottom_frame])
        hip_width = float(max(widths["hip"][bottom_frame], 1e-6))
        l_valgus = abs(xy[bottom_frame, L_KNEE, 0] - xy[bottom_frame, L_ANK, 0]) / hip_width
        r_valgus = abs(xy[bottom_frame, R_KNEE, 0] - xy[bottom_frame, R_ANK, 0]) / hip_width
        valgus_max = float(max(l_valgus, r_valgus))   
        
        # Gap between reps
        inter_rep_gap = (t_start - prev_end_time) if prev_end_time is not None else np.nan
        prev_end_time = t_end
        
        rows.append({
            "rep_index": seg["rep_index"],
            "start_frame": start, "end_frame": end,
            "start_time": t_start, "end_time": t_end, "duration_s": duration,
            "y_min": y_min, "y_max": y_max, "y_range": y_range,
            "top_frame": top_frame, "bottom_frame": bottom_frame,
            "top_time": top_time, "bottom_time": bottom_time,
            
            "signal": signal_key,
            "anchor": anchor,
            
            "trunk_angle_at_bottom_rad": trunk_at_bottom,
            "trunk_angle_at_top_rad": trunk_at_top,
            
            "lKnee_min": lKnee_min, "lKnee_max": lKnee_max, "lKnee_rom": lKnee_rom,
            "rKnee_min": rKnee_min, "rKnee_max": rKnee_max, "rKnee_rom": rKnee_rom,
            "lHip_min": lHip_min, "lHip_max": lHip_max, "lHip_rom": lHip_rom,
            "rHip_min": rHip_min, "rHip_max": rHip_max, "rHip_rom": rHip_rom,
            
            "knee_sym_at_bottom": knee_sym_at_bottom,
            "hip_sym_at_bottom": hip_sym_at_bottom,
            "knee_rom_diff": knee_rom_diff,
            "hip_rom_diff": hip_rom_diff,
            
            "shoulder_y_top": shoulder_y_top,
            "shoulder_y_bottom": shoulder_y_bottom,
            "pelvis_y_top": pelvis_y_top,
            "pelvis_y_bottom": pelvis_y_bottom,
            "pushup_depth": pushup_depth,
            "hip_sag_at_bottom": hip_sag_at_bottom,
            "elbow_flare_left_norm": float(l_elbow_flare),
            "elbow_flare_right_norm": float(r_elbow_flare),
            "elbow_flare_avg_norm": float(elbow_flare_avg),

            "hip_y_bottom": squat_hip_y_bottom,
            "knee_valgus_left_norm": float(l_valgus),
            "knee_valgus_right_norm": float(r_valgus),
            "knee_valgus_max_norm": float(valgus_max),
            
            "inter_rep_gap_s": inter_rep_gap
        })
    
    df = pd.DataFrame(rows).sort_values("rep_index").reset_index(drop=True)
    
    args.out_csv.parent.mkdir(parents = True, exist_ok = True)
    df.to_csv(args.out_csv, index = False)
    print(f"Saved per-rep features in {args.out_csv}")
    
if __name__ == "__main__":
    main()
    