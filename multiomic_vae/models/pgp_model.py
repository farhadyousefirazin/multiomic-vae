# pgp_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------
# Reparameterization trick
# --------------------------------------------------
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# --------------------------------------------------
# Simple MLP builder
# (keeps your original behavior: activation after EVERY Linear,
# including the last layer)
# --------------------------------------------------
def make_mlp(dims, activation):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(activation())
    return nn.Sequential(*layers)


# --------------------------------------------------
# Connected VAE (Reverse of GPG)
#
# Peak -> Encoder1 -> z1 -> Decoder1 -> Gene_hat
# Gene_hat -> Encoder2 -> z2 -> Decoder2 -> Peak_hat
#
# IMPORTANT:
# - Outputs are REAL-VALUED predictions (not probabilities)
# - gene_dim and peak_dim are inferred from data
# - Model behavior is FIXED
# --------------------------------------------------
class PGPVAE(nn.Module):
    def __init__(
        self,
        gene_x,             # torch.Tensor (cells × genes)
        peak_x,             # torch.Tensor (cells × peaks)
        hidden_dims,        # e.g. [256, 128]
        latent_dim,         # int
        activation=nn.ReLU  # activation CLASS
    ):
        super().__init__()

        gene_dim = gene_x.shape[1]
        peak_dim = peak_x.shape[1]

        # ----- Encoder 1: Peak -> z1 -----
        self.enc1 = make_mlp([peak_dim] + hidden_dims, activation)
        self.mu1 = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar1 = nn.Linear(hidden_dims[-1], latent_dim)

        # ----- Decoder 1: z1 -> Gene_hat -----
        self.dec1 = make_mlp(
            [latent_dim] + hidden_dims[::-1] + [gene_dim],
            activation
        )

        # ----- Encoder 2: Gene_hat -> z2 -----
        self.enc2 = make_mlp([gene_dim] + hidden_dims, activation)
        self.mu2 = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar2 = nn.Linear(hidden_dims[-1], latent_dim)

        # ----- Decoder 2: z2 -> Peak_hat -----
        self.dec2 = make_mlp(
            [latent_dim] + hidden_dims[::-1] + [peak_dim],
            activation
        )

    # -----------------------------
    # Small helper APIs (so your test code stays clean)
    # -----------------------------
    def enc1_mu(self, peak_x):
        h1 = self.enc1(peak_x)
        return self.mu1(h1)

    def enc2_mu(self, gene_x):
        h2 = self.enc2(gene_x)
        return self.mu2(h2)

    def peak_to_gene(self, peak_x):
        h1 = self.enc1(peak_x)
        mu1 = self.mu1(h1)
        logvar1 = self.logvar1(h1)
        z1 = reparameterize(mu1, logvar1)
        gene_hat = self.dec1(z1)
        return gene_hat

    def gene_to_peak(self, gene_x):
        h2 = self.enc2(gene_x)
        mu2 = self.mu2(h2)
        logvar2 = self.logvar2(h2)
        z2 = reparameterize(mu2, logvar2)
        peak_hat = self.dec2(z2)
        return peak_hat

    def cycle_peak(self, peak_x):
        gene_hat = self.peak_to_gene(peak_x)
        peak_hat = self.gene_to_peak(gene_hat)
        return peak_hat

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, peak_x):
        # Peak -> z1
        h1 = self.enc1(peak_x)
        mu1 = self.mu1(h1)
        logvar1 = self.logvar1(h1)
        z1 = reparameterize(mu1, logvar1)

        # z1 -> Gene_hat (REAL VALUES)
        gene_hat = self.dec1(z1)

        # Gene_hat -> z2
        h2 = self.enc2(gene_hat)
        mu2 = self.mu2(h2)
        logvar2 = self.logvar2(h2)
        z2 = reparameterize(mu2, logvar2)

        # z2 -> Peak_hat
        peak_hat = self.dec2(z2)

        # Return order matches your GPG training style:
        # (gene_hat first, then peak_hat)
        return gene_hat, peak_hat, mu1, logvar1, mu2, logvar2


# --------------------------------------------------
# Loss function
#
# - recon_loss_fn is user-chosen (REGRESSION ONLY)
# - beta is provided externally (can change per epoch)
# --------------------------------------------------
def loss_fn(
    gene_hat,
    gene_true,
    peak_hat,
    peak_true,
    mu1,
    logvar1,
    mu2,
    logvar2,
    recon_loss_fn=F.mse_loss,
    beta=1.0
):
    recon_gene = recon_loss_fn(gene_hat, gene_true)
    recon_peak = recon_loss_fn(peak_hat, peak_true)

    kl1 = -0.5 * torch.mean(1 + logvar1 - mu1**2 - logvar1.exp())
    kl2 = -0.5 * torch.mean(1 + logvar2 - mu2**2 - logvar2.exp())

    return recon_gene + recon_peak + beta * (kl1 + kl2)