
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from multiomic_vae.config.gpg_config import config as default_gpg_config
from multiomic_vae.models.gpg_model import GPGVAE
from multiomic_vae.utils.train_data_loader import load_npz_as_df


# --------------------------------------------------
# Reproducibility helper
# --------------------------------------------------
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Path + data loading helpers
# --------------------------------------------------
def load_pbmc_data(
    project_root: Path | str,
    dataset_name: str = "pbmc_10k",
) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """
    Load the preprocessed gene and peak matrices used by the GPG notebook.

    Expected files:
    - processed_data/<dataset_name>/cell_gene_preprocessed.npz
    - processed_data/<dataset_name>/cell_peak_preprocessed.npz
    """
    project_root = Path(project_root)
    data_path = project_root / "processed_data" / dataset_name

    gene_path = data_path / "cell_gene_preprocessed.npz"
    peak_path = data_path / "cell_peak_preprocessed.npz"

    gene = load_npz_as_df(gene_path)
    peak = load_npz_as_df(peak_path)

    return gene, peak, data_path


def dataframe_to_tensor(
    df: pd.DataFrame,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    device = torch.device(device) if device is not None else torch.device("cpu")
    return torch.tensor(df.values, dtype=torch.float32, device=device)


# --------------------------------------------------
# Model helpers
# --------------------------------------------------
def build_gpg_model(
    gene_df: pd.DataFrame,
    peak_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    device: torch.device | str | None = None,
) -> tuple[GPGVAE, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Instantiate the GPG model using dataframe shapes, without training it.
    """
    cfg = dict(default_gpg_config)
    if config is not None:
        cfg.update(config)

    device = torch.device(device) if device is not None else torch.device("cpu")

    gene_x = dataframe_to_tensor(gene_df, device=device)
    peak_x = dataframe_to_tensor(peak_df, device=device)

    model = GPGVAE(
        gene_x=gene_x,
        peak_x=peak_x,
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        activation=cfg["activation"],
    ).to(device)

    return model, gene_x, peak_x, cfg


def beta_schedule(epoch: int, beta_min: float, beta_max: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return beta_max
    if epoch >= warmup_epochs:
        return beta_max
    return beta_min + (beta_max - beta_min) * (epoch / warmup_epochs)


def train_gpg_without_wandb(
    gene_df,
    peak_df,
    config=None,
    device=None,
    return_history=True,
    verbose=True,
    print_every=1,
):
    import time
    import pandas as pd
    import torch

    if verbose:
        print("Building model...")

    model, gene_x, peak_x, cfg = build_gpg_model(
        gene_df=gene_df,
        peak_df=peak_df,
        config=config,
        device=device,
    )

    if verbose:
        print("Model built.")
        print("Device:", next(model.parameters()).device)
        print("gene_x shape:", tuple(gene_x.shape))
        print("peak_x shape:", tuple(peak_x.shape))
        print("Config:", cfg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
    )

    history_rows = []

    model.train()
    for epoch in range(cfg["epochs"]):
        t0 = time.time()

        beta = beta_schedule(
            epoch=epoch,
            beta_min=cfg["beta_min"],
            beta_max=cfg["beta_max"],
            warmup_epochs=cfg["beta_warmup_epochs"],
        )

        optimizer.zero_grad()

        if verbose and epoch == 0:
            print("Starting forward pass...")

        gene_hat, peak_hat, mu1, logvar1, mu2, logvar2 = model(gene_x)

        if verbose and epoch == 0:
            print("Forward pass done.")
            print("gene_hat shape:", tuple(gene_hat.shape))
            print("peak_hat shape:", tuple(peak_hat.shape))

        recon_gene = cfg["recon_loss"](gene_hat, gene_x)
        recon_peak = cfg["recon_loss"](peak_hat, peak_x)

        kl1 = -0.5 * torch.mean(1 + logvar1 - mu1**2 - logvar1.exp())
        kl2 = -0.5 * torch.mean(1 + logvar2 - mu2**2 - logvar2.exp())

        loss = recon_gene + recon_peak + beta * (kl1 + kl2)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"Invalid loss at epoch {epoch}: "
                f"loss={loss.item()}, "
                f"recon_gene={recon_gene.item()}, "
                f"recon_peak={recon_peak.item()}, "
                f"kl1={kl1.item()}, kl2={kl2.item()}"
            )

        loss.backward()
        optimizer.step()

        history_rows.append({
            "epoch": epoch,
            "beta": beta,
            "loss_total": loss.item(),
            "loss_gene_recon": recon_gene.item(),
            "loss_peak_recon": recon_peak.item(),
            "loss_kl1": kl1.item(),
            "loss_kl2": kl2.item(),
        })

        if verbose and ((epoch + 1) % print_every == 0):
            print(
                f"[Epoch {epoch+1}/{cfg['epochs']}] "
                f"loss={loss.item():.6f} | "
                f"gene={recon_gene.item():.6f} | "
                f"peak={recon_peak.item():.6f} | "
                f"kl1={kl1.item():.6f} | "
                f"kl2={kl2.item():.6f} | "
                f"beta={beta:.6f} | "
                f"time={time.time()-t0:.2f}s"
            )

    model.eval()
    history = pd.DataFrame(history_rows)

    return model, gene_x, peak_x, history


# --------------------------------------------------
# Explainability helpers
# --------------------------------------------------
def get_gene_index(gene_df: pd.DataFrame, gene_name: str) -> int:
    if gene_name not in gene_df.columns:
        raise KeyError(f"Gene '{gene_name}' was not found in gene_df.columns.")
    return int(gene_df.columns.get_loc(gene_name))


def get_cell_vector(gene_df, cell_name, device=None):
    import numpy as np
    import torch

    if cell_name not in gene_df.index:
        raise KeyError(f"Cell '{cell_name}' was not found in gene_df.index.")

    device = torch.device(device) if device is not None else torch.device("cpu")

    # Keep as 1-row DataFrame, then densify only that row
    row_df = gene_df.loc[[cell_name]]

    # If DataFrame uses pandas sparse dtype, convert only this row to dense
    try:
        row_df = row_df.sparse.to_dense()
    except Exception:
        pass

    values = row_df.iloc[0].to_numpy(dtype=np.float32)

    return torch.tensor(values, dtype=torch.float32, device=device)


def perturb_gene(
    gene_vector: torch.Tensor | np.ndarray,
    gene_index: int,
    new_value: float,
) -> torch.Tensor:
    vector = _to_tensor_1d(gene_vector).clone()
    vector[gene_index] = float(new_value)
    return vector


def predict_peaks(
    model: torch.nn.Module,
    gene_vector: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """
    Run one forward pass and return the predicted peak vector.

    Notes:
    - The GPG model forward() returns:
      gene_hat, peak_hat, mu1, logvar1, mu2, logvar2
    - Because the model uses the VAE reparameterization trick, repeated calls
      can return different predictions even in eval mode.
    """
    model.eval()

    x = _to_tensor_1d(gene_vector)
    device = next(model.parameters()).device
    x = x.to(device).unsqueeze(0)

    with torch.no_grad():
        _, peak_hat, *_ = model(x)

    return peak_hat.squeeze(0).detach().cpu()


def classify_peak(
    probability: float,
    threshold: float = 0.3,
    grey_zone: tuple[float, float] = (0.29, 0.31),
) -> int:
    """
    Classify a peak score into:
    - 1  -> OPEN
    - 0  -> CLOSED
    - -1 -> UNCERTAIN
    """
    grey_lower, grey_upper = grey_zone

    if not (grey_lower <= threshold <= grey_upper):
        raise ValueError("The threshold should lie inside the grey zone bounds.")

    if probability > grey_upper:
        return 1
    if probability < grey_lower:
        return 0
    return -1


def detect_peak_changes(
    baseline_peaks: torch.Tensor | np.ndarray,
    perturbed_peaks: torch.Tensor | np.ndarray,
    threshold: float = 0.3,
    grey_zone: tuple[float, float] = (0.29, 0.31),
    min_abs_change: float = 0.05,
) -> dict[str, np.ndarray]:
    """
    Compare baseline and perturbed predictions and flag only robust binary changes.

    A peak is considered changed only if:
    1. baseline and perturbed states are both outside the grey zone
    2. the binary state changes (OPEN <-> CLOSED)
    3. |perturbed - baseline| >= min_abs_change
    """
    baseline = _to_numpy_1d(baseline_peaks)
    perturbed = _to_numpy_1d(perturbed_peaks)

    baseline_state = classify_peaks_vectorized(
        baseline,
        threshold=threshold,
        grey_zone=grey_zone,
    )
    perturbed_state = classify_peaks_vectorized(
        perturbed,
        threshold=threshold,
        grey_zone=grey_zone,
    )

    valid_mask = (baseline_state != -1) & (perturbed_state != -1)
    state_changed = baseline_state != perturbed_state
    delta_ok = np.abs(perturbed - baseline) >= float(min_abs_change)
    changed_mask = valid_mask & state_changed & delta_ok

    closed_to_open = changed_mask & (baseline_state == 0) & (perturbed_state == 1)
    open_to_closed = changed_mask & (baseline_state == 1) & (perturbed_state == 0)

    transitions = np.full(baseline.shape, "", dtype=object)
    transitions[closed_to_open] = "Closed->Open"
    transitions[open_to_closed] = "Open->Closed"

    return {
        "baseline_scores": baseline,
        "perturbed_scores": perturbed,
        "baseline_state": baseline_state,
        "perturbed_state": perturbed_state,
        "valid_mask": valid_mask,
        "changed_mask": changed_mask,
        "closed_to_open_mask": closed_to_open,
        "open_to_closed_mask": open_to_closed,
        "transitions": transitions,
        "absolute_delta": np.abs(perturbed - baseline),
    }


def run_monte_carlo_perturbation(
    model: torch.nn.Module,
    gene_vector: torch.Tensor | np.ndarray,
    gene_index: int,
    new_value: float,
    N: int,
    peak_names: Optional[Sequence[str]] = None,
    baseline_peaks: torch.Tensor | np.ndarray | None = None,
    threshold: float = 0.3,
    grey_zone: tuple[float, float] = (0.29, 0.31),
    min_abs_change: float = 0.05,
) -> dict[str, Any]:
    """
    Keep the baseline prediction fixed, perturb one gene, then repeatedly sample
    perturbed predictions and count how often each peak changes state.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer.")

    original_gene_vector = _to_tensor_1d(gene_vector).detach().cpu()
    perturbed_gene_vector = perturb_gene(original_gene_vector, gene_index, new_value)

    if baseline_peaks is None:
        baseline_peaks = predict_peaks(model, original_gene_vector)

    baseline_np = _to_numpy_1d(baseline_peaks)
    num_peaks = baseline_np.shape[0]

    if peak_names is None:
        peak_names = [f"Peak_{i}" for i in range(num_peaks)]
    else:
        peak_names = list(peak_names)

    if len(peak_names) != num_peaks:
        raise ValueError("Length of peak_names must match the number of peaks.")

    changed_counts = np.zeros(num_peaks, dtype=np.int64)
    closed_to_open_counts = np.zeros(num_peaks, dtype=np.int64)
    open_to_closed_counts = np.zeros(num_peaks, dtype=np.int64)

    for _ in range(N):
        perturbed_peaks = predict_peaks(model, perturbed_gene_vector)

        change_info = detect_peak_changes(
            baseline_peaks=baseline_np,
            perturbed_peaks=perturbed_peaks,
            threshold=threshold,
            grey_zone=grey_zone,
            min_abs_change=min_abs_change,
        )

        changed_counts += change_info["changed_mask"].astype(np.int64)
        closed_to_open_counts += change_info["closed_to_open_mask"].astype(np.int64)
        open_to_closed_counts += change_info["open_to_closed_mask"].astype(np.int64)

    results = pd.DataFrame(
        {
            "Peak_ID": peak_names,
            "Change_Count": changed_counts,
            "Change_Frequency": changed_counts / float(N),
            "Closed_to_Open_Count": closed_to_open_counts,
            "Closed_to_Open_Frequency": closed_to_open_counts / float(N),
            "Open_to_Closed_Count": open_to_closed_counts,
            "Open_to_Closed_Frequency": open_to_closed_counts / float(N),
            "Baseline_Prediction": baseline_np,
        }
    ).sort_values(
        by=["Change_Frequency", "Change_Count", "Peak_ID"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return {
        "baseline_peaks": baseline_np,
        "original_gene_vector": original_gene_vector.cpu().numpy(),
        "perturbed_gene_vector": perturbed_gene_vector.cpu().numpy(),
        "results": results,
        "settings": {
            "N": int(N),
            "threshold": float(threshold),
            "grey_zone": tuple(float(x) for x in grey_zone),
            "min_abs_change": float(min_abs_change),
            "gene_index": int(gene_index),
            "new_value": float(new_value),
        },
    }


def plot_top_sensitive_peaks(
    results_df: pd.DataFrame,
    top_k: int = 20,
    frequency_column: str = "Change_Frequency",
    title: str = "Most sensitive peaks",
) -> pd.DataFrame:
    """
    Plot the top-k peaks ranked by change frequency.
    """
    top_df = results_df.nlargest(top_k, frequency_column).copy()

    plt.figure(figsize=(12, max(4, 0.35 * len(top_df))))
    plt.barh(top_df["Peak_ID"].astype(str), top_df[frequency_column].values)
    plt.xlabel("Change frequency")
    plt.ylabel("Peak ID")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return top_df


# --------------------------------------------------
# Internal vector helpers
# --------------------------------------------------
def classify_peaks_vectorized(
    values: np.ndarray,
    threshold: float = 0.3,
    grey_zone: tuple[float, float] = (0.29, 0.31),
) -> np.ndarray:
    grey_lower, grey_upper = grey_zone

    if not (grey_lower <= threshold <= grey_upper):
        raise ValueError("The threshold should lie inside the grey zone bounds.")

    states = np.full(values.shape, -1, dtype=np.int8)
    states[values > grey_upper] = 1
    states[values < grey_lower] = 0
    return states


def _to_numpy_1d(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 1:
        raise ValueError(f"Expected a 1D vector, but received shape {x.shape}.")
    return x


def _to_tensor_1d(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        tensor = x.detach().clone()
    else:
        tensor = torch.tensor(np.asarray(x), dtype=torch.float32)

    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    if tensor.ndim != 1:
        raise ValueError(f"Expected a 1D vector, but received shape {tuple(tensor.shape)}.")

    return tensor.to(torch.float32)
