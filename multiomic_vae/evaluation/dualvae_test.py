import numpy as np
import torch
import matplotlib.pyplot as plt
import umap
import wandb

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    adjusted_rand_score,
    adjusted_mutual_info_score
)
from scipy.stats import pearsonr


@torch.no_grad()
def run_tests_with_wandb(
    model,
    gene_tensor,
    peak_tensor,
    labels,
    peak_pred_threshold=None,
    umap_neighbors=15,
):
    model.eval()
    labels = np.asarray(labels)

    # --------------------------------------------------
    # Latents
    # IMPORTANT:
    # For evaluation we use posterior means (deterministic),
    # not sampled latents.
    # --------------------------------------------------
    z_gene = model.get_z_gene(gene_tensor, sample=False).cpu().numpy()
    z_peak = model.get_z_peak(peak_tensor, sample=False).cpu().numpy()

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
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ARI": ari,
            "AMI": ami,
        }

    # --------------------------------------------------
    # Evaluate z_gene and z_peak
    # --------------------------------------------------
    results["z_gene"] = eval_latent(z_gene, "z_gene")
    results["z_peak"] = eval_latent(z_peak, "z_peak")

    # --------------------------------------------------
    # Latent alignment quality
    # --------------------------------------------------
    latent_l2 = float(np.mean(np.linalg.norm(z_gene - z_peak, axis=1)))

    wandb.log({
        "test/latent_mean_l2": latent_l2
    })

    results["latent_alignment"] = {
        "mean_l2_distance": latent_l2
    }

    # --------------------------------------------------
    # Peak reconstruction (binary threshold sweep)
    #
    # IMPORTANT:
    # In DualVAE this is peak reconstruction quality,
    # not gene->peak translation like GPG.
    # --------------------------------------------------
    peak_real = peak_tensor.cpu().numpy()
    peak_hat = model.reconstruct_peak(peak_tensor).cpu().numpy()

    peak_real_bin = (peak_real > 0).astype(int)

    if peak_pred_threshold is None:
        threshold_values = np.arange(0.0, 3.3, 0.3)
    else:
        threshold_values = np.array([peak_pred_threshold], dtype=float)

    best_threshold = None
    best_p, best_r, best_f1 = 0.0, 0.0, -1.0

    for thr in threshold_values:
        peak_hat_bin = (peak_hat > thr).astype(int)

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
    # Gene reconstruction
    # --------------------------------------------------
    gene_hat = model.reconstruct_gene(gene_tensor).cpu().numpy()
    gene_real = gene_tensor.cpu().numpy()

    corrs = []
    for i in range(gene_real.shape[0]):
        try:
            r, _ = pearsonr(gene_real[i], gene_hat[i])
            if not np.isnan(r):
                corrs.append(r)
        except Exception:
            pass

    mean_corr = float(np.mean(corrs)) if len(corrs) > 0 else float("nan")

    wandb.log({
        "test/gene_mean_pearson": mean_corr
    })

    results["genes"] = {
        "mean_pearson_correlation": mean_corr
    }

    return results