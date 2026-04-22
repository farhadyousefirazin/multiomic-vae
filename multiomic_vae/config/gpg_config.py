import torch.nn as nn
import torch.nn.functional as F

config = {
    "recon_loss": F.mse_loss,
    "hidden_dims": [1024],
    "latent_dim": 256,
    "activation": nn.Tanh,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "batch_size": 512,
    "beta_min": 0.0,
    "beta_max": 0.1,
    "beta_warmup_epochs": 20,
    "epochs": 30,
}
