import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

import math
from scipy.signal import savgol_filter

def parse_args():
    parser = argparse.ArgumentParser(description="Extract 33-keypoint pose data from a video")
    parser.add_argument("--input_video", type=Path, required=True, help="Path to an MP4/MOV file")
    parser.add_argument("--output_path", type=Path, required=True, help="Where to save the .npz output")
    parser.add_argument("--write_overlay", action="store_true", help="Also save an overlay MP4 with skeleton drawn")
    
    # Smoothing controls. window=0 disables. Window must be odd and >=5; poly < window.
    parser.add_argument("--savgol_window", type=int, default=0, help="Odd window length, e.g., 7. 0 disables smoothing.")
    parser.add_argument("--savgol_poly",   type=int, default=2, help="Polynomial order, e.g., 2. Must be < window.")
    
    return parser.parse_args()

def savgol_smooth(seq: np.ndarray, window: int, poly: int) -> np.ndarray:
    """
    Smooths along time for x,y,z (columns 0..2). Visibility (col 3) left unchanged.
    seq: (T, 33, 4)  -> returns (T, 33, 4)
    """
    if window <= 0 or seq.shape[0] < max(window, poly + 2) or (window % 2 == 0):
        return seq
    out = seq.copy()
    for j in (0, 1, 2):  # x, y, z
        out[:, :, j] = savgol_filter(seq[:, :, j], window_length=window, polyorder=poly, axis=0, mode="interp")
    return out

def main():
    args = parse_args()
    
    cap = cv2.VideoCapture(str(args.input_video))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {total_frames} frames @ {fps:.2f} FPS")
    
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    
    pose = mp_pose.Pose(
        static_image_mode = False,
        model_complexity = 1,
        smooth_landmarks = True,
        enable_segmentation = False,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
    )
    
    def open_writer_and_get_resized(path: Path, fps_val: float, first_frame_bgr):
        """
        Returns (writer, resize_fn, out_path)
        - writer: an opened cv2.VideoWriter using avc1/mp4 fallback
        - resize_fn(frame) -> resized_frame with even width/height <= 1920 on long side
        - out_path: path to overlay video (may end .avi if we had to fallback)
        """

        # 1) Build a resizing function that keeps aspect ratio and enforces even dims
        def compute_target_size(h, w, max_long=1920):
            long_side = max(h, w)
            scale = 1.0 if long_side <= max_long else (max_long / long_side)
            new_w = int(math.floor((w * scale) / 2) * 2) or 2   # ensure even & nonzero
            new_h = int(math.floor((h * scale) / 2) * 2) or 2
            return new_w, new_h

        h0, w0 = first_frame_bgr.shape[:2]
        tw, th = compute_target_size(h0, w0)  # target even dims, <=1920 long side

        def resize_fn(img):
            return cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)

        # 2) Try avc1 in mp4, then mp4v in mp4, then MJPG in avi (last resort)
        fps_out = float(fps_val) if (fps_val and fps_val > 0) else 25.0
        trials = [
            ("mp4", "avc1", path.with_suffix(".overlay.mp4")),  # H.264 (best for QuickTime)
            ("mp4", "mp4v", path.with_suffix(".overlay.mp4")),  # Older MPEG-4
            ("avi", "MJPG", path.with_suffix(".overlay.avi")),  # Huge files, but plays everywhere
        ]

        for ext, fourcc_str, outp in trials:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            outp.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(outp), fourcc, fps_out, (tw, th))
            if writer.isOpened():
                return writer, resize_fn, outp

        raise RuntimeError("Failed to open any VideoWriter with avc1/mp4v/MJPG.")

    
    vwriter = None
    resize_fn = None
    
    keypoints = []
    timestamps = []
    frames_done = 0
    
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        if args.write_overlay and vwriter is None:
            vwriter, resize_fn, overlay_path = open_writer_and_get_resized(
                args.output_path, fps, frame)
            print(f"Overlay writer opened → {overlay_path}")
        
        frame_proc = resize_fn(frame) if resize_fn else frame
        
        rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype = np.float32)
        else:
            coords = np.zeros((33, 4), dtype = np.float32)
        
        keypoints.append(coords)
        timestamps.append(frames_done / fps)
        
        if vwriter is not None:
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image = frame_proc,
                    landmark_list = results.pose_landmarks,
                    connections = mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec = mp_styles.get_default_pose_landmarks_style()
                )
            vwriter.write(frame_proc)

            frames_done += 1
            if frames_done % 50 == 0:
                print(f"Processed {frames_done} frames...", flush=True)
    
    cap.release()
    pose.close()
    if vwriter is not None:
        vwriter.release()
    
    keypoints = np.stack(keypoints, axis = 0)
    timestamps = np.asarray(timestamps, dtype = np.float32)
    
    keypoints_sm = savgol_smooth(keypoints, args.savgol_window, args.savgol_poly)
    
    args.output_path.parent.mkdir(parents = True, exist_ok = True)
    np.savez_compressed(
        args.output_path, 
        keypoints = keypoints_sm, 
        raw_keypoints = keypoints,
        timestamps = timestamps,
        fps = float(fps)
    )
    
    print(f"Saved -> {args.output_path} | T = {keypoints.shape[0]} | sg_win = {args.savgol_window}")
    if vwriter is not None:
        print(f"Saved overlay in {overlay_path}")
    
if __name__ == "__main__":
    main()
            
        