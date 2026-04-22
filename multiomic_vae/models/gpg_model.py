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
# --------------------------------------------------
def make_mlp(dims, activation, final_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))

        is_last_layer = (i == len(dims) - 2)
        if (not is_last_layer) or final_activation:
            layers.append(activation())

    return nn.Sequential(*layers)


# --------------------------------------------------
# Connected VAE
#
# Gene -> Encoder1 -> z1 -> Decoder1 -> Peak_hat
# Peak_hat -> Encoder2 -> z2 -> Decoder2 -> Gene_hat
# --------------------------------------------------
class GPGVAE(nn.Module):
    def __init__(
        self,
        gene_x,
        peak_x,
        hidden_dims,
        latent_dim,
        activation=nn.SiLU
    ):
        super().__init__()

        gene_dim = gene_x.shape[1]
        peak_dim = peak_x.shape[1]

        # ----- Encoder 1: Gene -> z1 -----
        self.enc1 = make_mlp([gene_dim] + hidden_dims, activation)
        self.mu1 = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar1 = nn.Linear(hidden_dims[-1], latent_dim)

        # ----- Decoder 1: z1 -> Peak_hat -----
        self.dec1 = make_mlp(
            [latent_dim] + hidden_dims[::-1] + [peak_dim],
            activation,
            final_activation=False
        )

        # ----- Encoder 2: Peak_hat -> z2 -----
        self.enc2 = make_mlp([peak_dim] + hidden_dims, activation)
        self.mu2 = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar2 = nn.Linear(hidden_dims[-1], latent_dim)

        # ----- Decoder 2: z2 -> Gene_hat -----
        self.dec2 = make_mlp(
            [latent_dim] + hidden_dims[::-1] + [gene_dim],
            activation,
            final_activation=False
        )

    def forward(self, gene_x):
        h1 = self.enc1(gene_x)
        mu1 = self.mu1(h1)
        logvar1 = self.logvar1(h1)
        z1 = reparameterize(mu1, logvar1)

        peak_hat = self.dec1(z1)

        h2 = self.enc2(peak_hat)
        mu2 = self.mu2(h2)
        logvar2 = self.logvar2(h2)
        z2 = reparameterize(mu2, logvar2)

        gene_hat = self.dec2(z2)

        return gene_hat, peak_hat, mu1, logvar1, mu2, logvar2


# --------------------------------------------------
# Loss function
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