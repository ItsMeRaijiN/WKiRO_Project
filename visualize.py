"""
Generate figures

Run:
    python visualize.py
"""

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_reader import load_dataset
from dataset_torch import MocapDataset
from model import Conv1dCAE
from train import train_epoch, validate
from evaluate import compute_scores


def load_or_build_cache(cache_path: Path, dataset_root: str, num_workers: int):
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        return d["X"].astype(np.float32), d["y"].astype(int), d["subj"], d["channels"].tolist()

    print(f"Cache {cache_path} not found - building from dataset {dataset_root} ...")
    ds = load_dataset(dataset_root, verbose=True, num_workers=num_workers)
    X = ds["X"].astype(np.float32)
    y = ds["y"].astype(np.int8)
    subj = np.array([m["subject_id"] for m in ds["meta"]])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, subj=subj,
                        channels=np.array(ds["channels"], dtype=object))
    print(f"Saved cache: {cache_path}")
    return X, y.astype(int), subj, ds["channels"]


def build_split(X, y, subj, normal_label, seed):
    ns = sorted(set(subj[y == normal_label]))
    rng = np.random.default_rng(seed); rng.shuffle(ns)
    n = len(ns); ntr = int(n * 0.7); nva = max(1, int(n * 0.15))
    trs = set(ns[:ntr]); vas = set(ns[ntr:ntr + nva]); used = trs | vas
    itr = [i for i in range(len(X)) if y[i] == normal_label and subj[i] in trs]
    iva = [i for i in range(len(X)) if y[i] == normal_label and subj[i] in vas]
    ite = [i for i in range(len(X)) if not (y[i] == normal_label and subj[i] in used)]
    return itr, iva, ite


def main():
    ap = argparse.ArgumentParser(description="Figures for the gait experiment.")
    ap.add_argument("--cache", default="cache/cycles_raw.npz")
    ap.add_argument("--dataset-root", default=r"E:\ML_datasets\Data_Run_Walk")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--normal-label", type=int, default=1, help="1=M (default), 0=F")
    ap.add_argument("--latent-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    figdir = Path(args.figures_dir); figdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    X, y, subj, channels = load_or_build_cache(Path(args.cache), args.dataset_root, args.num_workers)
    T, C = X.shape[1], X.shape[2]

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    itr, iva, ite = build_split(X, y, subj, args.normal_label, args.seed)
    mean = X[itr].mean((0, 1), keepdims=True); std = X[itr].std((0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    nz = lambda A: (A - mean) / std

    train_loader = DataLoader(MocapDataset(nz(X[itr])), batch_size=32, shuffle=True)
    val_loader = DataLoader(MocapDataset(nz(X[iva])), batch_size=64)
    anom = (y[ite] != args.normal_label).astype(np.float32)
    test_loader = DataLoader(MocapDataset(nz(X[ite]), anom), batch_size=64)

    model = Conv1dCAE(C, latent_dim=args.latent_dim, seq_len=T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    hist_tr, hist_va = [], []
    best, best_state = 1e9, None
    for ep in range(args.epochs):
        tl = train_epoch(model, train_loader, opt, device, grad_clip=1.0)
        vl = validate(model, val_loader, device)
        hist_tr.append(tl); hist_va.append(vl)
        if vl < best:
            best = vl; best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)

    scores, labels = compute_scores(model, test_loader, device)
    roc = roc_auc_score(labels, scores)
    gender_te = np.array(["M" if yy == 1 else "F" for yy in y[ite]])

    #Training curve
    plt.figure(figsize=(7, 4))
    plt.plot(hist_tr, label="train")
    plt.plot(hist_va, label="val")
    plt.xlabel("Epoch"); plt.ylabel("Loss (MSE + 0.5 * derivative MSE)")
    plt.title("Training curve (trained on males)")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "gait_training_curve.png", dpi=130); plt.close()

    # Error distribution normal v anomaly
    err_m = scores[gender_te == "M"]; err_f = scores[gender_te == "F"]
    plt.figure(figsize=(7, 4))
    plt.hist(err_m, bins=40, alpha=0.6, density=True, label="Males (normal)")
    plt.hist(err_f, bins=40, alpha=0.6, density=True, label="Females (anomaly)")
    plt.xlabel("Reconstruction error"); plt.ylabel("Density")
    plt.title(f"Reconstruction-error distribution (ROC-AUC={roc:.3f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "gait_error_dist.png", dpi=130); plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC curve (cycle level)")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "gait_roc.png", dpi=130); plt.close()

    #Example reconstructions
    show_channels = ["LKneeAngles_X", "LHipAngles_X", "LAnkleAngles_X"]
    ch_idx = [channels.index(c) for c in show_channels if c in channels]
    i_norm = next(i for i in range(len(ite)) if gender_te[i] == "M")
    i_anom = next(i for i in range(len(ite)) if gender_te[i] == "F")
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(nz(X[ite][[i_norm, i_anom]]), dtype=torch.float32, device=device)
        rec = model(xb).cpu().numpy()
    rec_deg = rec * std + mean
    orig_deg = X[ite][[i_norm, i_anom]]

    fig, axes = plt.subplots(2, len(ch_idx), figsize=(4 * len(ch_idx), 6), squeeze=False)
    rows = [("Male (normal)", 0), ("Female (anomaly)", 1)]
    for r, (title, ri) in enumerate(rows):
        for c, ci in enumerate(ch_idx):
            ax = axes[r][c]
            ax.plot(orig_deg[ri, :, ci], label="original", lw=2)
            ax.plot(rec_deg[ri, :, ci], label="reconstruction", ls="--", lw=2)
            ax.set_title(f"{title}\n{channels[ci]}")
            ax.set_xlabel("% of cycle"); ax.set_ylabel("angle [deg]")
            if r == 0 and c == 0:
                ax.legend()
    fig.suptitle("Original vs reconstruction (AE trained on males)")
    plt.tight_layout()
    plt.savefig(figdir / "gait_reconstructions.png", dpi=130); plt.close()

    print(f"ROC-AUC (cycle) = {roc:.4f}")
    print(f"Saved figures to: {figdir}/")
    for f in ["gait_training_curve.png", "gait_error_dist.png", "gait_roc.png", "gait_reconstructions.png"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
