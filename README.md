# Workout Form AI

Workout Form AI is a local, end-to-end pipeline for detecting exercise form faults in bodyweight workouts (push-ups for now) from raw video. 
It combines pose estimation, rule-based heuristics, and machine learning baselines, with a Streamlit demo and a fine-tuned LLM for personalized coaching feedback.

## Features

- **Pose Extraction**: Uses MediaPipe BlazePose to extract 2D skeletons from video frames, with optional smoothing. 
- **Feature Computation**: Converts raw pose into pelvis-centered, torso-scaled coordinates, joint angles, and velocities. 
- **Repetition Segmentation**: Robust anchor-based peak detection on normalized shoulder/pelvis/knee signals, with tail recovery for final reps.
- **Rep Aggregates**: Computes per-rep features such as range of motion, symmetry, hip sag, elbow flare, and knee valgus. 
- **Fault Detection**: 
  - Heuristic rules (depth thresholds, angle differences).
  - ML models (Logistic Regression) trained on labeled rep-level data.
  - Dataset: LSTM Exercise Classification: Push Up Videos (Kaggle)
- **Evaluation**: Comparison of heuristics vs ML against manually-labeled test sets. 
- **LLM Coaching**: Fine-tuned a 3B parameter model locally (LoRA on MLX) to generate concise, supportive coaching feedback per rep.

## Pipeline

1. **Extract Pose**
   ```bash
   python -u scripts/extract_pose.py --input_video data/raw/pushup_test.mp4 --output_path data/processed/test_pose.npz --write_overlay
   ```
2. **Compute Features**
   ```bash
   python -u scripts/compute_features.py --pose_npz data/processed/test_pose.npz --output_path data/processed/test_pose_features.npz
   ```
3. **Segment Reps**
   ```bash
   python -u scripts/segment_reps.py --features_npz data/processed/test_pose_features.npz --output_json data/processed/test_pose_reps.json --exercise pushup --signal shoulder --anchor top
   ```
4. **Aggregate Rep Features**
   ```bash
   python -u scripts/rep_features.py --features_npz data/processed/test_pose_features.npz --reps_json data/processed/test_pose_reps.json --out_csv data/processed/test_pose_rep_features.csv
   ```
5. **Evaluate Heuristics vs ML**
   ```bash
   python -u scripts/eval.py --rep_csv data/processed/pushup_rep_features_all.csv --flags_csv data/processed/pushup_rule_faults.csv --splits_csv data/processed/pushup_split.csv --gold_csv data/processed/pushup_real_fault.csv --out_dir outputs/eval/pushup
   ```
6. **Streamlit Demo**
   ```bash
   streamlit run demo.py
   ```

## Results

- ML models generally outperformed heuristics in most fault detections
- F1 scores:
  - Hip sag: 0.923 (Heuristics) vs 0.909 (ML model)
  - Insufficient depth: 0.600 (Heuristics) vs 0.889 (ML model)
  - Elbow flare: 0.250 (Heuristics) vs 0.667 (ML model)
- AUPRC:
  - Hip sag: 0.857 (Heuristics) vs 0.933 (ML model)
  - Insufficient depth: 0.451 (Heuristics) vs 0.877 (ML model)
  - Elbow flare: 0.129 (Heuristics) vs 0.833 (ML model)
- Demonstrated robustness across varied camera angles and datasets of correct vs incorrect form.

## LLM Fine-tuning for Coaching

- Created a JSONL dataset mapping rep features + faults → concise coaching cues. 
- Fine-tuned **Llama-3.2-3B-Instruct** with LoRA adapters using `mlx-lm` on an Apple M3 Pro (fully local). 
- Integrated into the demo app: given detected faults, the fine-tuned model rewrites them into clear 1–2 sentence feedback. 

## Repository Layout

```
workout-form-ai/
  scripts/
    extract_pose.py
    compute_features.py
    segment_reps.py
    rep_features.py
    eval.py
    build_coach_jsonl.py
    demo.py    # Streamlit demo
  data/
    raw/         # input videos
    processed/   # pose npz, features, reps, rep_features, splits, labels
  oututs/
    eval/        # evaluation metrics
    model/       # Logistic Regression model
```

## Technical Stack

- Python 3.11, virtualenv via pyenv
- Libraries: mediapipe==0.10.14, opencv-python, numpy, scipy, scikit-learn, pandas, streamlit, mlx-lm
- Hardware: Apple MacBook Pro (M3 Pro)
- All computation fully local (no cloud GPUs required)

## Next Steps

- Expand dataset to more exercises (squats, bench press, pull ups, ...).
- Human-in-the-loop labeling workflow to bootstrap larger gold datasets.
