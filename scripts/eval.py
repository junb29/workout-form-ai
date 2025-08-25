#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
import joblib

FAULTS = ["insufficient_depth", "hip_sag", "elbow_flare"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train ML on dataset (train split) and compare to RULES (test split).")
    parser.add_argument("--rep_csv", type=Path, required=True, help="Merged rep features (has video_id, rep_index)")
    parser.add_argument("--splits_csv", type=Path, required=True, help="video_id,split in {train,test}")
    parser.add_argument("--real_csv", type=Path, required=True, help="Real labels per rep_index (train+test)")
    parser.add_argument("--rules_csv", type=Path, required=False, help="(Optional) Heuristic rule flags per rep_index")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    return parser.parse_args()

def pick_features(df: pd.DataFrame) -> list[str]:
    drop = {
        "rep_index", "rep_index_local", "rep_index_global", "duration_s", "y_min", "y_max", "y_range",
        "start_frame","end_frame","start_time","end_time","top_frame","bottom_frame","top_time","bottom_time",
        "signal","anchor","video_path","video_id","inter_rep_gap_s",
    }
    num = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    return [c for c in num if c not in drop and c not in FAULTS]

def cls_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    precision,recall,f1,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    out = {"precision":float(precision), "recall":float(recall), "f1":float(f1)}
    try: 
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
    except: 
        out["auroc"] = float("nan")
    try: 
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    except: 
        out["auprc"] = float("nan")
    return out

def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reps = pd.read_csv(args.rep_csv)
    splits = pd.read_csv(args.splits_csv)
    real = pd.read_csv(args.real_csv)

    reps = reps.merge(splits, on="video_id", how="left")
    reps = reps.merge(real, on="rep_index", how="left", suffixes=("","_real"))

    if args.rules_csv and Path(args.rules_csv).exists():
        rules = pd.read_csv(args.rules_csv)
        rule_cols_in = [c for c in rules.columns if c in FAULTS or c == "rep_index"]
        rules = rules[rule_cols_in].copy()
        rename = {c: f"{c}_rule" for c in rule_cols_in if c != "rep_index"}
        rules = rules.rename(columns=rename)
        reps = reps.merge(rules, on="rep_index", how="left")

    faults_avail = [f for f in FAULTS if f in real.columns]
    if not faults_avail:
        raise SystemExit("No fault columns found in real labels.")

    feat_cols = pick_features(reps)
    (args.model_dir / "feature_cols.json").write_text(json.dumps(feat_cols, indent=2))

    train = reps[reps["split"]=="train"].copy()
    test  = reps[reps["split"]=="test"].copy()

    Xtr = train[feat_cols].values.astype(np.float32)
    Xte = test [feat_cols].values.astype(np.float32)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s  = scaler.transform(Xtr)
    Xte_s  = scaler.transform(Xte)

    report = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_count": len(feat_cols),
        "faults": {}
    }

    lines = [
        "Rules vs ML on GOLD labels:",
        "",
        f"- Train reps: {len(train)}, Test reps: {len(test)}",
        f"- Features: {len(feat_cols)}",
        "",
        "| Fault | Rules F1 | ML F1 | Rules AUPRC | ML AUPRC |",
        "|---|---:|---:|---:|---:|"
    ]

    preds = test[["rep_index", "video_id"]].copy()

    for f in faults_avail:
        
        ytr = train[f].values.astype(int)
        yte = test[f].values.astype(int)

        # ML model
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(Xtr_s, ytr)
        joblib.dump(model, args.model_dir / f"model_{f}.joblib")
        y_prob_ml = model.predict_proba(Xte_s)[:, 1]
        m_ml = cls_metrics(yte, y_prob_ml)
        preds[f"prob_{f}_ml"] = y_prob_ml

        if f"{f}_rule" in test.columns:
            y_prob_rule = test[f"{f}_rule"].fillna(0).values.astype(float) 
        else:
            y_prob_rule = np.zeros_like(yte, dtype=float)
            
        m_rules = cls_metrics(yte, y_prob_rule)
        preds[f"pred_{f}_rule"] = (y_prob_rule >= 0.5).astype(int)

        report["faults"][f] = {"rules": m_rules, "ml": m_ml}
        lines.append(f"| {f} | {m_rules['f1']:.3f} | {m_ml['f1']:.3f} | "
                     f"{m_rules['auprc']:.3f} | {m_ml['auprc']:.3f} |")

    preds.to_csv(args.out_dir / "preds.csv", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    (args.out_dir / "metrics.md").write_text("\n".join(lines))
    print("\n".join(lines))

if __name__ == "__main__":
    main()
