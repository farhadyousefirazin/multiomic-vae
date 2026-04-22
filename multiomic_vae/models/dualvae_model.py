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
# Activation factory
# --------------------------------------------------
def get_activation(activation):
    """
    Supports either:
    - string: "relu", "tanh", "gelu", "silu", "mish"
    - nn.Module class: nn.ReLU, nn.Tanh, ...
    """
    if isinstance(activation, str):
        name = activation.lower()
        mapping = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish,
        }
        if name not in mapping:
            raise ValueError(f"Unknown activation: {activation}")
        return mapping[name]

    if isinstance(activation, type) and issubclass(activation, nn.Module):
        return activation

    raise ValueError(f"Unsupported activation: {activation}")


# --------------------------------------------------
# Simple MLP builder
# - no activation after final layer
# --------------------------------------------------
def make_mlp(dims, activation):
    if len(dims) < 2:
        return nn.Identity()

    act_cls = get_activation(activation)
    layers = []

    for i in range(len(dims) - 2):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(act_cls())

    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# --------------------------------------------------
# Single-modality VAE branch
# --------------------------------------------------
class SingleVAE(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims,
        latent_dim,
        activation="relu",
    ):
        super().__init__()

        hidden_dims = hidden_dims or []

        # Encoder
        if len(hidden_dims) == 0:
            self.encoder = nn.Identity()
            last_hidden_dim = input_dim
        else:
            self.encoder = make_mlp([input_dim] + hidden_dims, activation)
            last_hidden_dim = hidden_dims[-1]

        self.mu = nn.Linear(last_hidden_dim, latent_dim)
        self.logvar = nn.Linear(last_hidden_dim, latent_dim)

        # Decoder
        self.decoder = make_mlp(
            [latent_dim] + hidden_dims[::-1] + [input_dim],
            activation
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = reparameterize(mu, logvar)
        return mu, logvar, z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar, z = self.encode(x)
        recon = self.decode(z)
        return recon, mu, logvar, z


# --------------------------------------------------
# Dual VAE
#
# Gene  -> gene VAE -> gene_hat
# Peak  -> peak VAE -> peak_hat
#
# Plus latent alignment between gene and peak branches
# --------------------------------------------------
class DualVAE(nn.Module):
    def __init__(
        self,
        gene_x,                     # torch.Tensor (cells x genes)
        peak_x,                     # torch.Tensor (cells x peaks)
        gene_hidden_dims,           # e.g. [2048]
        peak_hidden_dims,           # e.g. [2048]
        latent_dim,                 # int
        gene_activation="relu",
        peak_activation="relu",
        alignment_type="l2",        # "none", "l2", "kl", "fusion"
        fusion_hidden_dims=None,
    ):
        super().__init__()

        gene_dim = gene_x.shape[1]
        peak_dim = peak_x.shape[1]

        self.alignment_type = alignment_type.lower()
        self.latent_dim = latent_dim

        self.gene_vae = SingleVAE(
            input_dim=gene_dim,
            hidden_dims=gene_hidden_dims,
            latent_dim=latent_dim,
            activation=gene_activation,
        )

        self.peak_vae = SingleVAE(
            input_dim=peak_dim,
            hidden_dims=peak_hidden_dims,
            latent_dim=latent_dim,
            activation=peak_activation,
        )

        self.fusion_mlp = None
        if self.alignment_type == "fusion":
            fusion_hidden_dims = fusion_hidden_dims or [256]
            fusion_activation = gene_activation

            self.fusion_mlp = make_mlp(
                [2 * latent_dim] + fusion_hidden_dims + [latent_dim],
                fusion_activation
            )

    def forward(self, gene_x, peak_x):
        gene_hat, mu_gene, logvar_gene, z_gene = self.gene_vae(gene_x)
        peak_hat, mu_peak, logvar_peak, z_peak = self.peak_vae(peak_x)

        return (
            gene_hat, peak_hat,
            mu_gene, logvar_gene, z_gene,
            mu_peak, logvar_peak, z_peak,
        )

    # ----------------------------------------------
    # Convenient helpers for testing
    # ----------------------------------------------
    def get_z_gene(self, gene_x, sample=False):
        mu_gene, logvar_gene, z_gene = self.gene_vae.encode(gene_x)
        return z_gene if sample else mu_gene

    def get_z_peak(self, peak_x, sample=False):
        mu_peak, logvar_peak, z_peak = self.peak_vae.encode(peak_x)
        return z_peak if sample else mu_peak

    def reconstruct_gene(self, gene_x):
        gene_hat, _, _, _ = self.gene_vae(gene_x)
        return gene_hat

    def reconstruct_peak(self, peak_x):
        peak_hat, _, _, _ = self.peak_vae(peak_x)
        return peak_hat

    # ----------------------------------------------
    # Alignment loss
    # ----------------------------------------------
    def alignment_loss(self, mu_gene, logvar_gene, mu_peak, logvar_peak):
        if self.alignment_type == "none":
            return torch.tensor(0.0, device=mu_gene.device)

        if self.alignment_type == "l2":
            return F.mse_loss(mu_gene, mu_peak)

        if self.alignment_type == "kl":
            return 0.5 * (
                self._kl_between_gaussians(mu_gene, logvar_gene, mu_peak, logvar_peak) +
                self._kl_between_gaussians(mu_peak, logvar_peak, mu_gene, logvar_gene)
            )

        if self.alignment_type == "fusion":
            if self.fusion_mlp is None:
                raise RuntimeError("fusion_mlp is None but alignment_type='fusion'.")

            z_shared = self.fusion_mlp(torch.cat([mu_gene, mu_peak], dim=1))
            return (
                F.mse_loss(z_shared, mu_gene) +
                F.mse_loss(z_shared, mu_peak)
            )

        raise ValueError(f"Unknown alignment type: {self.alignment_type}")

    def _kl_between_gaussians(self, mu1, logvar1, mu2, logvar2):
        var1 = logvar1.exp()
        var2 = logvar2.exp()

        return 0.5 * (
            (logvar2 - logvar1) +
            (var1 + (mu1 - mu2) ** 2) / (var2 + 1e-8) -
            1
        ).sum(dim=1).mean()


# --------------------------------------------------
# Loss helpers
# --------------------------------------------------
def kl_standard_normal(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def recon_mse(x, recon_x):
    return F.mse_loss(recon_x, x)


def recon_weighted_mse(x, recon_x, w_zero=0.05):
    w = torch.where(
        x > 0,
        torch.ones_like(x),
        torch.full_like(x, w_zero)
    )
    return (w * (recon_x - x) ** 2).mean()


def recon_nb_simplified(x, recon_x):
    return torch.mean(torch.exp(recon_x) - x * recon_x)


def recon_zinb_simplified(x, recon_x):
    return F.mse_loss(torch.sigmoid(recon_x), x)


def recon_loss_by_name(x, recon_x, name, w_zero=0.05):
    name = name.lower()

    if name == "mse":
        return recon_mse(x, recon_x)
    if name in ("wmse", "weighted_mse"):
        return recon_weighted_mse(x, recon_x, w_zero=w_zero)
    if name == "nb":
        return recon_nb_simplified(x, recon_x)
    if name == "zinb":
        return recon_zinb_simplified(x, recon_x)

    raise ValueError(f"Unknown reconstruction loss: {name}")


# --------------------------------------------------
# Total loss
# --------------------------------------------------
def loss_fn(
    model,
    gene_hat,
    gene_true,
    peak_hat,
    peak_true,
    mu_gene,
    logvar_gene,
    mu_peak,
    logvar_peak,
    gene_recon_loss="mse",
    peak_recon_loss="mse",
    peak_w_zero=0.05,
    beta=1.0,
    align_weight=1.0,
):
    recon_gene = recon_loss_by_name(
        gene_true, gene_hat, gene_recon_loss
    )
    recon_peak = recon_loss_by_name(
        peak_true, peak_hat, peak_recon_loss, w_zero=peak_w_zero
    )

    kl_gene = kl_standard_normal(mu_gene, logvar_gene)
    kl_peak = kl_standard_normal(mu_peak, logvar_peak)

    align_loss = model.alignment_loss(
        mu_gene, logvar_gene, mu_peak, logvar_peak
    )

    total = (
        recon_gene +
        recon_peak +
        beta * (kl_gene + kl_peak) +
        align_weight * align_loss
    )

    return total, recon_gene, recon_peak, kl_gene, kl_peak, align_loss