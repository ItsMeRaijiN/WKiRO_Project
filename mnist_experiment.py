"""
mnist_experiment.py

Metrics: ROC-AUC, average precision + threshold-based metrics. Also saves figures.

Run:
    python mnist_experiment.py --epochs 25 --latent-dim 8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             accuracy_score, precision_score, recall_score, f1_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class ConvAE2d(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GELU(),   # 28 -> 14
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GELU(),  # 14 -> 7
        )
        self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, latent_dim))
        self.from_latent = nn.Sequential(nn.Linear(latent_dim, 32 * 7 * 7), nn.GELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.GELU(),  # 7 -> 14
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid(),  # 14 -> 28
        )
    def forward(self, x):
        z = self.to_latent(self.encoder(x))
        h = self.from_latent(z).view(-1, 32, 7, 7)
        return self.decoder(h)


def load_mnist(root):
    tf = transforms.ToTensor()
    train = datasets.MNIST(root, train=True, download=True, transform=tf)
    test = datasets.MNIST(root, train=False, download=True, transform=tf)
    Xtr = train.data.float().unsqueeze(1) / 255.0
    ytr = train.targets.clone()
    Xte = test.data.float().unsqueeze(1) / 255.0
    yte = test.targets.clone()
    return Xtr, ytr, Xte, yte

@torch.no_grad()
def recon_error(model, X, device, batch=512):
    model.eval()
    errs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(device)
        xh = model(xb)
        errs.append(torch.mean((xb - xh) ** 2, dim=(1, 2, 3)).cpu())
    return torch.cat(errs).numpy()


def main():
    ap = argparse.ArgumentParser(description="MNIST: CAE as an anomaly detector (9 = anomaly).")
    ap.add_argument("--data-root", default=r"E:\ML_datasets\mnist")
    ap.add_argument("--output-dir", default="artifacts_mnist")
    ap.add_argument("--figures-dir", default="figures")
    ap.add_argument("--anomaly-digit", type=int, default=9)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figures_dir); figdir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte = load_mnist(args.data_root)
    a = args.anomaly_digit
    normal_mask = ytr != a
    Xn = Xtr[normal_mask]
    perm = torch.randperm(len(Xn))
    n_val = 5000
    Xval = Xn[perm[:n_val]]; Xtrain = Xn[perm[n_val:]]
    print(f"Train: {len(Xtrain)} (digits 0-8), val: {len(Xval)}, test: {len(Xte)} "
          f"(incl. {(yte == a).sum().item()} anomalies '{a}')")

    train_loader = DataLoader(TensorDataset(Xtrain), batch_size=256, shuffle=True)

    model = ConvAE2d(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    history = []
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb)
            loss.backward(); opt.step()
            tot += loss.item()
        tr = tot / len(train_loader)
        va = float(np.mean(recon_error(model, Xval, device)))
        history.append((tr, va))
        print(f"Epoch {ep:02d} | train={tr:.6f} | val={va:.6f}")

    scores = recon_error(model, Xte, device)
    labels = (yte.numpy() == a).astype(int)
    roc = roc_auc_score(labels, scores)
    apr = average_precision_score(labels, scores)

    thr = float(np.percentile(scores[labels == 0], 95))
    pred = (scores >= thr).astype(int)
    metrics = {
        "anomaly_digit": a, "latent_dim": args.latent_dim,
        "roc_auc": float(roc), "average_precision": float(apr),
        "threshold_p95": thr,
        "accuracy": float(accuracy_score(labels, pred)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "recall": float(recall_score(labels, pred, zero_division=0)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "normal_mean_error": float(scores[labels == 0].mean()),
        "anomaly_mean_error": float(scores[labels == 1].mean()),
    }
    print("\nMNIST EVALUATION")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    with (out / "metrics.json").open("w", encoding="utf-8") as h:
        json.dump(metrics, h, indent=2)

    #figures
    plt.figure(figsize=(7, 4))
    plt.hist(scores[labels == 0], bins=60, alpha=0.6, density=True, label="Normal (0-8)")
    plt.hist(scores[labels == 1], bins=60, alpha=0.6, density=True, label=f"Anomaly ({a})")
    plt.axvline(thr, color="k", ls="--", lw=1, label="Threshold (95th pct)")
    plt.xlabel("Reconstruction error (MSE)"); plt.ylabel("Density")
    plt.title(f"MNIST: error distribution (ROC-AUC={roc:.3f}, AP={apr:.3f})")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "mnist_error_hist.png", dpi=130); plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC={roc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("MNIST: ROC curve")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "mnist_roc.png", dpi=130); plt.close()

    # Example reconstructions
    model.eval()
    idx_norm = np.where(labels == 0)[0][:8]
    idx_anom = np.where(labels == 1)[0][:8]
    sel = np.concatenate([idx_norm, idx_anom])
    with torch.no_grad():
        rec = model(Xte[sel].to(device)).cpu().numpy()
    fig, axes = plt.subplots(2, 16, figsize=(16, 2.4))
    for j, i in enumerate(sel):
        axes[0, j].imshow(Xte[i, 0], cmap="gray"); axes[0, j].axis("off")
        axes[1, j].imshow(rec[j, 0], cmap="gray"); axes[1, j].axis("off")
    axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("recon.")
    fig.suptitle(f"MNIST reconstructions - left half: normal (0-8), right: anomaly ({a})")
    plt.tight_layout()
    plt.savefig(figdir / "mnist_reconstructions.png", dpi=130); plt.close()

    print(f"\nSaved figures to: {figdir}/  and metrics to {out/'metrics.json'}")


if __name__ == "__main__":
    main()
