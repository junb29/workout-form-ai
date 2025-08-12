import argparse
from pathlib import Path
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description = "Read a video and print basic info")
    parser.add_argument("--input_video", type = Path, required = True, help = "Path to an MP4/MOV file.")
    return parser.parse_args()

def main():
    args = parse_args()
    cap = cv2.VideoCapture(str(args.input_video))
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.input_video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{args.input_video} | {w}x{h} @ {fps:.2f} fps | frames: {n}")
    
    frames = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frames += 1
        
    cap.release()
    print(f"Read {frames} frames.")
    
    if __name__ == "__main__":
        main()
    
    