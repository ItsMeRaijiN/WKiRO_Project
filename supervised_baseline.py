"""
supervised_baseline.py

Supervised ceiling for sex discriminability on the same gait cycles, using subject-wise
GroupKFold cross-validation.

Run:
    python supervised_baseline.py
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from visualize import load_or_build_cache


def cv_auc(features, y, groups, classifier_factory, scale=False, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros(len(features))
    for tr, te in gkf.split(features, y, groups=groups):
        Xtr, Xte = features[tr], features[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = classifier_factory().fit(Xtr, y[tr])
        oof[te] = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(y, oof)


def main():
    ap = argparse.ArgumentParser(description="Supervised ceiling (subject-wise CV).")
    ap.add_argument("--cache", default="cache/cycles_raw.npz")
    ap.add_argument("--dataset-root", default=r"E:\ML_datasets\Data_Run_Walk")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    X, y, subj, _ = load_or_build_cache(Path(args.cache), args.dataset_root, args.num_workers)
    y = y.astype(int)
    N = len(X)
    Xflat = X.reshape(N, -1)
    rom = X.max(axis=1) - X.min(axis=1)

    print(f"Cycles: {N}, features: {Xflat.shape[1]}, "
          f"F={int((y == 0).sum())} M={int((y == 1).sum())}, subjects={len(set(subj))}")
    print(f"\nSupervised ceiling ({args.n_splits}-fold GroupKFold):")
    print(f"  LogisticRegression (flattened): ROC-AUC = "
          f"{cv_auc(Xflat, y, subj, lambda: LogisticRegression(max_iter=2000, C=0.1), scale=True, n_splits=args.n_splits):.4f}")
    print(f"  RandomForest (flattened):       ROC-AUC = "
          f"{cv_auc(Xflat, y, subj, lambda: RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1), n_splits=args.n_splits):.4f}")
    print(f"  LogisticRegression (ROM only):  ROC-AUC = "
          f"{cv_auc(rom, y, subj, lambda: LogisticRegression(max_iter=2000), scale=True, n_splits=args.n_splits):.4f}")


if __name__ == "__main__":
    main()
