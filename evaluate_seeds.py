"""
evaluate_seeds.py
=================
Multi-seed evaluation of the gait CAE anomaly detector, for both training directions.
Reports mean +/- std of ROC-AUC (cycle), average precision, and ROC-AUC (subject) over
several subject-wise splits. Reproduces the multi-seed numbers cited in REPORT.md.

Uses the raw-cycles cache `cache/cycles_raw.npz` (built automatically from the dataset
if missing).

Run:
    python evaluate_seeds.py --seeds 10
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset_torch import MocapDataset
from model import Conv1dCAE
from train import train_epoch, validate
from evaluate import compute_scores
from visualize import load_or_build_cache, build_split


def run_one(X, y, subj, normal_label, latent, epochs, seed, device):
    """Train one AE on a subject-wise split and return (cycle ROC, AP, subject ROC)."""
    itr, iva, ite = build_split(X, y, subj, normal_label, seed)
    mean = X[itr].mean((0, 1), keepdims=True)
    std = X[itr].std((0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    nz = lambda A: (A - mean) / std

    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader = DataLoader(MocapDataset(nz(X[itr])), batch_size=32, shuffle=True)
    val_loader = DataLoader(MocapDataset(nz(X[iva])), batch_size=64)
    anom = (y[ite] != normal_label).astype(np.float32)
    test_loader = DataLoader(MocapDataset(nz(X[ite]), anom), batch_size=64)

    model = Conv1dCAE(X.shape[2], latent_dim=latent, seq_len=X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_state = 1e9, None
    for _ in range(epochs):
        train_epoch(model, train_loader, opt, device, grad_clip=1.0)
        vl = validate(model, val_loader, device)
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    scores, labels = compute_scores(model, test_loader, device)
    roc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)

    ts = np.asarray(subj)[ite]
    s_scores = [scores[ts == s].mean() for s in sorted(set(ts))]
    s_labels = [int(anom[ts == s][0]) for s in sorted(set(ts))]
    subj_roc = roc_auc_score(s_labels, s_scores)
    return roc, ap, subj_roc


def main():
    ap = argparse.ArgumentParser(description="Multi-seed evaluation (both training directions).")
    ap.add_argument("--cache", default="cache/cycles_raw.npz")
    ap.add_argument("--dataset-root", default=r"E:\ML_datasets\Data_Run_Walk")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    X, y, subj, _ = load_or_build_cache(Path(args.cache), args.dataset_root, args.num_workers)
    y = y.astype(int); subj = np.asarray(subj)
    seeds = list(range(args.seeds))

    print(f"\nLatent dim={args.latent_dim}, epochs={args.epochs}, seeds={args.seeds}\n")
    print(f"{'normal class':<14} {'ROC-AUC (cycle)':>18} {'AP':>16} {'ROC-AUC (subject)':>20}")
    for nl, name in [(1, "M (anomaly=F)"), (0, "F (anomaly=M)")]:
        R = np.array([run_one(X, y, subj, nl, args.latent_dim, args.epochs, s, device) for s in seeds])
        print(f"{name:<14} "
              f"{R[:,0].mean():>9.3f} +/- {R[:,0].std():.3f} "
              f"{R[:,1].mean():>7.3f} +/- {R[:,1].std():.3f} "
              f"{R[:,2].mean():>9.3f} +/- {R[:,2].std():.3f}")


if __name__ == "__main__":
    main()
