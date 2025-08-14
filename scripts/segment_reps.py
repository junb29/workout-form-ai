import argparse
from pathlib import Path
import json

import numpy as np
from scipy.signal import savgol_filter, find_peaks

import matplotlib.pyplot as plt

L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_SHO, R_SHO = 11, 12

def parse_args():
    parser = argparse.ArgumentParser(description="Segment reps from per-frame features.")
    # Input and output path
    parser.add_argument("--features_npz",type=Path, required=True, help="Path to features.npz")
    parser.add_argument("--output_json", type=Path, required=True, help="Output path to save rep segments.")
    # Smoothing via Savitzky-Golay filter (odd window, poly < window)
    parser.add_argument("--savgol_window", type=int, default=7, help="Odd window length (e.g., 7). 0 disables smoothing.")
    parser.add_argument("--savgol_poly", type=int, default=2, help="Polynomial order (< window).")
    
    parser.add_argument("--exercise", choices=["sqaut", "pushup"], required=True, help="Choose preset exercise")
    parser.add_argument("--signal", choices=["pelvis", "shoulder", "knee"], default=None, help="Which y-position to use (pelvis/shoulder/knee). If omitted, preset decides")
    parser.add_argument("--anchor", choices=["top", "bottom"], default=None, help="Anchor type: top=minima, bottom=maxima. If omitted, preset decides.")
    # Selecting peaks and duration constraints
    parser.add_argument("--min_prom", type=float, default=0.05, help="Min prominence for bottoms")
    parser.add_argument("--min_range", type=float, default=0.1, help="Min y-range within a segment to count as a rep")
    parser.add_argument("--min_rep_seconds", type=float, default=0.6, help="Minimum allowed rep duration in sec.")
    parser.add_argument("--max_rep_seconds", type=float, default=8.0, help="Maximum allowed rep duration in sec")
    
    parser.add_argument("--plot_png", type=Path, default=None, help="If set, save a PNG showing velocity + detected reps.")
    
    return parser.parse_args()

def ypos(xy_norm: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    
    if name == "pelvis":
        y = 0.5 * (xy_norm[:, L_HIP, 1] + xy_norm[:, R_HIP, 1])
        return y.astype(np.float32), "pelvis_y (mid-hips)"
    if name == "shoulder":
        y = 0.5 * (xy_norm[:, L_SHO, 1] + xy_norm[:, R_SHO, 1])
        return y.astype(np.float32), "shoulder_y (mid-shoulders)"
    if name == "knee":
        y = 0.5 * (xy_norm[:, L_KNEE, 1] + xy_norm[:, R_KNEE, 1])
        return y.astype(np.float32), "knee_y (mid-knees)"
    
    raise ValueError(f"Unknown signal name: {name}")

def pick_defaults(exercise: str) -> tuple[str, str]:
    
    if exercise == "squat":
        return "pelvis", "bottom"
    
    else: # "pushup"
        return "shoulder", "top"

'''
def choose_velocity(d: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    
    if "vel_pelvis" in d.files:
        return d["vel_pelvis"].astype(np.float32), "pelvis_vel"
    candidates = []
    if "vel_lknee" in d.files and "vel_rknee" in d.files:
        candidates.append(((d["vel_lknee"] + d["vel_rknee"]) * 0.5, "mean_knee_vel"))
    if "vel_lhip" in d.files and "vel_rhip" in d.files:
        candidates.append(((d["vel_lhip"] + d["vel_rhip"]) * 0.5, "mean_hip_vel"))
    if candidates:
        arr, name = candidates[0]
        return arr.astype(np.float32), name
    raise ValueError("No suitable velocity keys found in features (.npz)")
'''

def smooth(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    
    if win <= 0 or x.shape[0] < max(win, poly + 2) or (win % 2 == 0) or poly >= win:
        return x
    return savgol_filter(x, window_length=win, polyorder=poly, mode="interp").astype(np.float32)

'''
def bottom_indices_from_velocity(v: np.ndarray, ts: np.ndarray, min_prom: float, min_gap_sec: float) -> np.ndarray:
    
    inv_v = -v
    # Average time between frames
    dt = np.maximum(np.diff(ts).mean() if ts.size > 1 else 0.04, 1e-6)
    # Minimum number of frames
    min_peak_distance = int(np.ceil(min_gap_sec / dt))
    
    peaks, _ = find_peaks(
        inv_v,
        distance = max(1, min_peak_distance),
        prominence = min_prom
    )
    
    return peaks
'''

def segment_by_anchor_position(y_pos: np.ndarray,
                               ts: np.ndarray,
                               cycle_anchor: str,
                               min_prom: float,
                               min_sec: float,
                               max_sec: float,
                               min_range: float) -> list[dict]:
    
    T = y_pos.shape[0]
    if T == 0:
        return []
    
    if cycle_anchor == "bottom":
        anchors, _ = find_peaks(y_pos, prominece = min_prom)
    else: # "top"
        anchors, _ = find_peaks(-y_pos, prominence = min_prom)
    
    # If video ends as soon as the rep is done
    def recover_tail_anchor(anchors: np.ndarray) -> np.ndarray:
        
        if anchors.size == 0:
            return anchors 
        
        last = int(anchors[-1])
        tail_start = last + 15 # Assuming fps == 30, 30 * 0.5
        if tail_start >= T - 1:
            return anchors
        
        tail = y_pos[tail_start:]
        cand_ex = int(np.argmin(tail) if cycle_anchor == "top" else np.argmax(tail))
        cand = tail_start + cand_ex
        
        dur = float(ts[cand] - ts[last])
        if not (min_sec <= dur <= max_sec):
            print('min max')
            return anchors
        seg_vals = y_pos[last: cand + 1]
        amp = float(seg_vals.max() - seg_vals.min())
        if amp < min_range:
            print("min range")
            return anchors
        
        diff = y_pos[last: cand]
        left_val = int(np.argmax(diff) if cycle_anchor == "top" else np.argmin(diff))
        if cycle_anchor == "top":
            prom_ok = (left_val - y_pos[cand] >= min_prom)
        else:
            prom_ok = (y_pos[cand] - left_val >= min_prom)
        
        if prom_ok:
            return np.concatenate([anchors, np.array([cand], dtype = int)])
        
        return anchors
    
    anchors = recover_tail_anchor(anchors)
    
    if anchors.size < 2:
        return []
    
    segs = []
    for i in range(anchors.size - 1):
        start = int(anchors[i])
        end = int(anchors[i+1])
        if end <= start:
            continue
        # Duration of a rep
        dur = ts[end] - ts[start]
        if dur < min_sec or dur > max_sec:
            continue
        
        seg_range = float(y_pos[start: end + 1].max() - y_pos[start: end + 1].min())
        if seg_range < min_range:
            continue
        
        segs.append({
            "rep_index": len(segs),
            "start_frame": start,
            "end_frame": end,
            "start_time": float(ts[start]),
            "end_time": float(ts[end]),
            'duration': float(dur),
            "anchor": cycle_anchor,
            "range": seg_range
        })
    
    return anchors, segs

def main():
    args = parse_args()
    
    d = np.load(args.features_npz)
    ts = d["timestamps"].astype(np.float32)
    xy_norm = d["xy_norm"].astype(np.float32)
    
    default_signal, default_anchor = pick_defaults(args.exercise)
    signal = args.signal or default_signal
    anchor = args.anchor or default_anchor
    
    y_pos_raw, desc = ypos(xy_norm, signal)
    y_pos = smooth(y_pos_raw, args.savgol_window, args.savgol_poly)
    
    idx, segs = segment_by_anchor_position(
        y_pos = y_pos,
        ts = ts,
        cycle_anchor = anchor,
        min_prom = args.min_prom,
        min_range = args.min_range,
        min_sec = args.min_rep_seconds,
        max_sec = args.max_rep_seconds
    )
    
    args.output_json.parent.mkdir(parents = True, exist_ok = True)
    payload = {
        "exercise": args.exercise,
        "signal": desc,
        "anchor": anchor,
        "savgol_window": args.savgol_window,
        "savgol_poly": args.savgol_poly,
        "min_prom": args.min_prom,
        "min_rep_seconds": args.min_rep_seconds,
        "max_rep_seconds": args.max_rep_seconds,
        "num_reps": len(segs),
        "rep_segments": segs
    }
    
    with open(args.output_json, "w") as f:
        json.dump(payload, f, indent = 2)
    print(f"Saved rep segments -> {args.output_json} (num_reps = {len(segs)})")
    
    if args.plot_png:
        plt.figure(figsize=(10, 4))
        if args.savgol_window > 0:
            plt.plot(ts, y_pos, label = f"{desc} (smoothed)")
        else:
            plt.plot(ts,  y_pos_raw, label = f"{desc} (raw)")
        
        plt.scatter(ts[idx], y_pos[idx], label = f"anchors ({anchor})", marker = "o")
        
        for seg in segs:
            plt.axvspan(seg["start_time"], seg["end_time"], alpha=0.15)
        plt.xlabel("time (s)")
        plt.ylabel("normalized y (down is +)")
        plt.legend()
        plt.tight_layout()
        
        out_png = args.plot_png
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"Saved plot → {out_png}")

if __name__ == "__main__":
    main()
            
        