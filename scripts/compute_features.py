import argparse
from pathlib import Path
import numpy as np
import json

L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK   = 27, 28
L_SHO, R_SHO   = 11, 12
L_ELB, R_ELB   = 13, 14

def parse_args():
    parser = argparse.ArgumentParser(description = "Compute pelvis-centered angles and velocities from pose .npz")
    parser.add_argument("--pose_npz", type=Path, required=True, help="Path to .npz file from extract_pose.py")
    parser.add_argument("--output_path", type=Path, required=True, help="Output path to save features .npz")
    
    return parser.parse_args()

def unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis = -1, keepdims = True) + eps
    return v / n

def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_u = unit(a)
    b_u = unit(b)
    dot = np.clip((a_u * b_u).sum(axis = -1), -1.0, 1.0)
    return np.arccos(dot)

def vertical() -> np.ndarray:
    return np.array([0.0, 1.0], dtype = np.float32)

def main():
    args = parse_args()
    
    d = np.load(args.pose_npz)
    kp = d["keypoints"]
    ts = d["timestamps"]
    fps = float(d["fps"])
    
    T = kp.shape[0]
    
    xy = kp[:, :, :2]
    vis = kp[:, :, 3]
    
    mid_hip = 0.5 * (xy[:, L_HIP, :] + xy[:, R_HIP, :])
    mid_sho = 0.5 * (xy[:, L_SHO, :] + xy[:, R_SHO, :])
    
    torso_vec = mid_sho - mid_hip
    torso_len = np.linalg.norm(torso_vec, axis = -1, keepdims = True)
    torso_len = np.clip(torso_len, 1e-4, None)
    
    xy_norm = (xy - mid_hip[:,  None, :]) / torso_len[:, None, :]
    
    def knee_angle(side):
        if side == "L":
            knee, hip, ank = L_KNEE, L_HIP, L_ANK
        else:
            knee, hip, ank = R_KNEE, R_HIP, R_ANK
        
        v1 = xy_norm[:, hip, :] - xy_norm[:, knee, :] # Knee to hip
        v2 = xy_norm[:, ank, :] - xy_norm[:, knee, :] # Knee to ankle
        
        return angle(v1, v2)
    
    def hip_angle(side):
        if side == "L":
            knee, hip, sho = L_KNEE, L_HIP, L_SHO
        else:
            knee, hip, sho = R_KNEE, R_HIP, R_SHO
            
        v1 = xy_norm[:, sho, :] - xy_norm[:, hip, :]   # Hip to shoulder
        v2 = xy_norm[:, knee, :] - xy_norm[:, hip, :]  # Hip to knee
        
        return angle(v1, v2)
    
    trunk_vec = unit(mid_sho - mid_hip)
    trunk_angle = angle(trunk_vec, np.broadcast_to(vertical(), trunk_vec.shape))
    
    # Get y-velocities
    def diff1(x, ts):
        out = np.zeros_like(x)
        dt = ts[1:] - ts[:-1]
        out[1:] = (x[1:] - x[:-1]) / np.maximum(dt, 1e-6)
        out[0] = out[1]
        return out

    # dt = 1.0 / fps
    vel_pelvis = diff1(mid_hip[:, 1], ts)
    vel_lhip   = diff1(xy_norm[:, L_HIP, 1], ts)
    vel_rhip   = diff1(xy_norm[:, R_HIP, 1], ts)
    vel_lknee  = diff1(xy_norm[:, L_KNEE,1], ts)
    vel_rknee  = diff1(xy_norm[:, R_KNEE,1], ts)
    vel_lsho   = diff1(xy_norm[:, L_SHO, 1], ts)
    vel_rsho   = diff1(xy_norm[:, R_SHO, 1], ts)

    feats = {
        "timestamps": ts,
        "torso_len": torso_len.squeeze(-1),
        "trunk_angle_rad": trunk_angle,
        "knee_angle_L_rad": knee_angle("L"),
        "knee_angle_R_rad": knee_angle("R"),
        "hip_angle_L_rad": hip_angle("L"),
        "hip_angle_R_rad": hip_angle("R"),
        
        "vel_pelvis": vel_pelvis,
        "vel_lhip": vel_lhip,
        "vel_rhip": vel_rhip,
        "vel_lknee": vel_lknee,
        "vel_rknee": vel_rknee,
        "vel_lsho": vel_lsho,
        "vel_rsho": vel_rsho,
        
        "xy_norm": xy_norm,
        "vis": vis
    }
    
    args.output_path.parent.mkdir(parents = True,  exist_ok = True)
    np.savez_compressed(args.output_path, **feats)
    
    metadata = {k: (np.asarray(v).shape if not isinstance(v, float) else "scalar") for k, v in feats.items()}
    with open(args.output_path.with_suffix(".json"), "w") as f:
        json.dump(metadata, f, indent = 2)
    
    print(f"Saved features -> {args.output_path}")
    for k, v in list(metadata.items())[:5]:
        print(f"    {k}: {v}")

if __name__ == "__main__":
    main()
    