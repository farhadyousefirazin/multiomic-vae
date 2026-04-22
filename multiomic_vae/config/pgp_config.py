# pgp_config.py
import torch.nn as nn
import torch.nn.functional as F

config = {
    "recon_loss": F.mse_loss,
    "hidden_dims": [1024],
    "latent_dim": 256,
    "activation": nn.Tanh,
    "lr": 2e-4,
    "beta_min": 0.0,
    "beta_max": 0.1,
    "beta_warmup_epochs": 20,
    "epochs": 30,
}