# pgp_test.py
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy.stats import pearsonr


@torch.no_grad()
def run_tests_with_wandb(
    model,
    gene_tensor,
    peak_tensor,
    labels,
    umap_neighbors=15,
    thr_min=0.0,
    thr_max=3.3,
    thr_step=0.3,
):
    model.eval()

    # --------------------------------------------------
    # z1 latent (from real peaks)
    # --------------------------------------------------
    z1 = model.enc1_mu(peak_tensor).cpu().numpy()

    # --------------------------------------------------
    # z2 latent (from predicted genes)
    # --------------------------------------------------
    gene_hat = model.peak_to_gene(peak_tensor)
    z2 = model.enc2_mu(gene_hat).cpu().numpy()

    results = {}

    # --------------------------------------------------
    # Helper: latent eval + UMAP
    # --------------------------------------------------
    def eval_latent(Z, name):
        reducer = umap.UMAP(
            n_neighbors=umap_neighbors,
            min_dist=0.1,
            random_state=0
        )
        Z_2d = reducer.fit_transform(Z)

        plt.figure(figsize=(6, 5))
        for lab in np.unique(labels):
            idx = labels == lab
            plt.scatter(Z_2d[idx, 0], Z_2d[idx, 1], s=8, alpha=0.9)
        plt.title(f"UMAP {name}")
        plt.tight_layout()

        wandb.log({f"UMAP/{name}": wandb.Image(plt)})
        plt.close()

        clf = LogisticRegression(max_iter=3000)
        clf.fit(Z, labels)
        preds = clf.predict(Z)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted", zero_division=0
        )
        acc = accuracy_score(labels, preds)

        ari = adjusted_rand_score(labels, preds)
        ami = adjusted_mutual_info_score(labels, preds)

        wandb.log({
            f"test/{name}_accuracy": acc,
            f"test/{name}_precision": precision,
            f"test/{name}_recall": recall,
            f"test/{name}_f1": f1,
            f"test/{name}_ARI": ari,
            f"test/{name}_AMI": ami,
        })

        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "ARI": float(ari),
            "AMI": float(ami),
        }

    # --------------------------------------------------
    # Evaluate z1 and z2
    # --------------------------------------------------
    results["z1"] = eval_latent(z1, "z1")
    results["z2"] = eval_latent(z2, "z2")

    # --------------------------------------------------
    # Peak reconstruction (binary) with threshold sweep
    # Using cycle output: Peak -> Gene_hat -> Peak_hat
    # --------------------------------------------------
    peak_real = peak_tensor.cpu().numpy()
    peak_hat_cycle = model.cycle_peak(peak_tensor).cpu().numpy()

    peak_real_bin = (peak_real > 0).astype(int)

    threshold_values = np.arange(thr_min, thr_max, thr_step)

    best_threshold = None
    best_p, best_r, best_f1 = 0.0, 0.0, -1.0

    for thr in threshold_values:
        peak_hat_bin = (peak_hat_cycle > thr).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            peak_real_bin.flatten(),
            peak_hat_bin.flatten(),
            average="binary",
            zero_division=0
        )

        if f1 > best_f1:
            best_f1 = f1
            best_p = p
            best_r = r
            best_threshold = thr

    wandb.log({
        "test/peak_best_threshold": float(best_threshold),
        "test/peak_precision": float(best_p),
        "test/peak_recall": float(best_r),
        "test/peak_f1": float(best_f1),
    })

    results["peaks"] = {
        "best_threshold": float(best_threshold),
        "precision": float(best_p),
        "recall": float(best_r),
        "f1": float(best_f1),
    }

    # --------------------------------------------------
    # Gene prediction quality (regression)
    # Using intermediate output: Peak -> Gene_hat
    # --------------------------------------------------
    gene_hat = gene_hat.cpu().numpy()
    gene_real = gene_tensor.cpu().numpy()

    corrs = []
    for i in range(gene_real.shape[0]):
        r, _ = pearsonr(gene_real[i], gene_hat[i])
        if not np.isnan(r):
            corrs.append(r)

    mean_corr = float(np.mean(corrs)) if len(corrs) > 0 else float("nan")

    wandb.log({
        "test/gene_mean_pearson": mean_corr
    })

    results["genes"] = {
        "mean_pearson_correlation": mean_corr
    }

    return results