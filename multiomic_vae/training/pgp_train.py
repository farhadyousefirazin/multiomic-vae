# pgp_train.py
import torch
import wandb

from multiomic_vae.models.pgp_model import PGPVAE
from multiomic_vae.config.pgp_config import config


def beta_schedule(epoch, beta_min, beta_max, warmup_epochs):
    if epoch >= warmup_epochs:
        return beta_max
    return beta_min + (beta_max - beta_min) * (epoch / warmup_epochs)


def pgp_train(gene, peak):
    wandb.init(
        project="PGP Model PBMC 10k - new",
        config={k: str(v) for k, v in config.items()}
    )

    gene_x = torch.tensor(gene.values, dtype=torch.float32)
    peak_x = torch.tensor(peak.values, dtype=torch.float32)

    model = PGPVAE(
        gene_x=gene_x,
        peak_x=peak_x,
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        activation=config["activation"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    for epoch in range(config["epochs"]):
        beta = beta_schedule(
            epoch,
            config["beta_min"],
            config["beta_max"],
            config["beta_warmup_epochs"]
        )

        optimizer.zero_grad()

        # IMPORTANT: PGP forward takes peak_x as input
        gene_hat, peak_hat, mu1, logvar1, mu2, logvar2 = model(peak_x)

        recon_gene = config["recon_loss"](gene_hat, gene_x)
        recon_peak = config["recon_loss"](peak_hat, peak_x)

        kl1 = -0.5 * torch.mean(1 + logvar1 - mu1**2 - logvar1.exp())
        kl2 = -0.5 * torch.mean(1 + logvar2 - mu2**2 - logvar2.exp())

        loss = recon_gene + recon_peak + beta * (kl1 + kl2)

        loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "beta": beta,
            "loss/total": loss.item(),
            "loss/gene_recon": recon_gene.item(),
            "loss/peak_recon": recon_peak.item(),
            "loss/kl1": kl1.item(),
            "loss/kl2": kl2.item(),
        })

    return model, gene_x, peak_x