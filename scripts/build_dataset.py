import argparse
import json
import sys                       
import subprocess as sp         
from pathlib import Path         
import pandas as pd              

def parse_args():
    parser = argparse.ArgumentParser(description="Process multiple videos and merge rep_features.")
    parser.add_argument("--exercise", default="pushup", required=True, help="Exercise type")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing raw videos (e.g., data/raw/pushup)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where to write processed data (e.g., data/processed/pushup)")
    parser.add_argument("--pattern", type=str, default="*.mp4,*.mov,*.avi", help="Comma-separated glob patterns for videos.")
    # Segmentation defaults (tweak as needed)
    parser.add_argument("--signal", default="shoulder", choices=["shoulder","pelvis","knee"])
    parser.add_argument("--anchor", default="top", choices=["top","bottom"])
    parser.add_argument("--savgol_window", type=int, default=9)
    parser.add_argument("--savgol_poly", type=int, default=2)
    parser.add_argument("--min_prom", type=float, default=0.001)
    parser.add_argument("--min_range", type=float, default=0.15)
    parser.add_argument("--min_rep_seconds", type=float, default=0.8)
    parser.add_argument("--max_rep_seconds", type=float, default=8.0)
    parser.add_argument("--run_rules", action="store_true", help="Run fault_rules.py to produce weak labels and merge them.")
    return parser.parse_args()

def run(cmd):
    """Run a command; stream stdout; raise on nonzero exit."""
    print(">>", " ".join(map(str, cmd)))
    proc = sp.run(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")
    
def process_one_video(vpath, args):
    """
    Process a single video through all steps and return artifact paths.
    """
    stem = vpath.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pose_npz = args.out_dir / f"{stem}_pose.npz"
    feat_npz = args.out_dir / f"{stem}_features.npz"
    reps_json = args.out_dir / f"{stem}_reps.json"
    plot_png = args.out_dir/ f"{stem}_reps_plot.png"
    rep_csv = args.out_dir / f"{stem}_rep_features.csv"
    overlay_mp4 = args.out_dir / f"{stem}.overlay.mp4"
    faults_json = args.out_dir / f"{stem}_faults.json"
    flags_csv = args.out_dir / f"{stem}_fault_flags.csv"

    # 1. Extract pose
    run([
        sys.executable, "-u", "scripts/extract_pose.py",
        "--input_video", str(vpath),
        "--output_path", str(pose_npz),
        "--write_overlay",
        "--savgol_window", str(args.savgol_window), "--savgol_poly", str(args.savgol_poly),
    ])

    # 2. Compute features
    run([
        sys.executable, "-u", "scripts/compute_features.py",
        "--pose_npz", str(pose_npz),
        "--output_path", str(feat_npz),
    ])

    # 3. Segment reps
    run([
        sys.executable, "-u", "scripts/segment_reps.py",
        "--features_npz", str(feat_npz),
        "--output_json", str(reps_json),
        "--exercise", args.exercise,
        "--signal", args.signal, "--anchor", args.anchor,
        "--savgol_window", str(args.savgol_window), "--savgol_poly", str(args.savgol_poly),
        "--min_prom", str(args.min_prom), "--min_range", str(args.min_range),
        "--min_rep_seconds", str(args.min_rep_seconds), "--max_rep_seconds", str(args.max_rep_seconds),
        "--plot_png", str(plot_png),
    ])

    # 4) Get per-rep features
    run([
        sys.executable, "-u", "scripts/rep_features.py",
        "--features_npz", str(feat_npz),
        "--reps_json", str(reps_json),
        "--out_csv", str(rep_csv),
    ])

    # 5) Get rules based (weak) labels
    if args.run_rules:
        run([
            sys.executable, "-u", "scripts/fault_rules.py",
            "--rep_csv", str(rep_csv),
            "--exercise", args.exercise,
            "--out_json", str(faults_json),
            "--out_flags_csv", str(flags_csv),
        ])

    return {
        "video_path": str(vpath),
        "video_id": vpath.stem,
        "rep_csv": str(rep_csv),
        "flags_csv": str(flags_csv) if args.run_rules else None,
        "overlay": str(overlay_mp4),
    }
    
def main():
    args = parse_args()

    # Collect videos
    vids = []
    for pat in args.pattern.split(","):
        vids += sorted(args.input_dir.rglob(pat.strip()))
    vids = [v for v in vids if v.is_file()]
    if not vids:
        print(f"No videos matched in {args.input_dir} with patterns {args.pattern}")
        return

    # Process videos
    entries = []
    for v in vids:
        try:
            info = process_one_video(v, args)
            entries.append(info)
        except Exception as e:
            print(f"[WARN] Failed on {v}: {e}")

    # Merge all rep_features into one CSV
    merged = []
    flags_list = []
    global_counter = 0

    for vid_info in entries:
        rep_csv = Path(vid_info["rep_csv"])
        if not rep_csv.exists():
            continue
        df = pd.read_csv(rep_csv)
        if df.empty:
            continue

        n = len(df)
        df = df.copy()
        df["video_path"] = vid_info["video_path"]
        df["video_id"] = vid_info["video_id"]

        # Keep original rep_index, but create a unique global one
        df = df.rename(columns={"rep_index": "rep_index_local"})
        df["rep_index"] = range(global_counter, global_counter + n)
        global_counter += n

        merged.append(df)

        # Weak flags (if any)
        if vid_info.get("flags_csv"):
            flag_csv = Path(vid_info["flags_csv"])
            if flag_csv.exists():
                flags = pd.read_csv(flag_csv).reset_index(drop=True)
                flags["rep_index"] = df["rep_index"].values
                flags["video_id"] = df["video_id"]
                flags_list.append(flags)

    if not merged:
        print("No per-video rep_features found; nothing to merge.")
        return

    all_reps = pd.concat(merged, ignore_index=True)
    out_rep_csv = args.out_dir / f"{args.exercise}_rep_features_all.csv"
    all_reps.to_csv(out_rep_csv, index=False)

    # Merge weak flags if present
    out_flags_csv = None
    if flags_list:
        all_flags = pd.concat(flags_list, ignore_index=True)
        # keep only rep_index + known fault columns
        keep = ["rep_index"] + [c for c in all_flags.columns if c != "rep_index"]
        out_flags_csv = args.out_dir / f"{args.exercise}_fault_flags_all.csv"
        all_flags[keep].to_csv(out_flags_csv, index=False)

    summary = {
        "exercise": args.exercise,
        "num_videos": len(entries),
        "num_reps": int(len(all_reps)),
        "outputs": {
            "rep_features_csv": str(out_rep_csv),
            "fault_flags_csv": str(out_flags_csv) if out_flags_csv else None
        }
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()