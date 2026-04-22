config = {
    "project_name": "DualVAE Model PBMC10k - new",

    # architecture
    "gene_hidden_dims": [2048],
    "peak_hidden_dims": [2048],
    "latent_dim": 256,

    # activations
    "gene_activation": "tanh",
    "peak_activation": "tanh",

    # alignment
    "alignment_type": "l2",      # "none", "l2", "kl", "fusion"
    "fusion_hidden_dims": [256], # used only if alignment_type == "fusion"
    "align_weight": 1.0,

    # reconstruction losses
    "gene_recon_loss": "mse",
    "peak_recon_loss": "weighted_mse",
    "peak_w_zero": 0.05,

    # optimization
    "lr": 1e-4,
    "beta_min": 0.0,
    "beta_max": 0.1,
    "beta_warmup_epochs": 20,
    "epochs": 100,
}