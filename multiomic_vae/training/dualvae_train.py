import torch
import wandb

from multiomic_vae.models.dualvae_model import DualVAE, loss_fn
from multiomic_vae.config.dualvae_config import config


def beta_schedule(epoch, beta_min, beta_max, warmup_epochs):
    if epoch >= warmup_epochs:
        return beta_max
    return beta_min + (beta_max - beta_min) * (epoch / warmup_epochs)


def dualvae_train(gene, peak):
    wandb.init(
        project=config["project_name"],
        config={k: str(v) for k, v in config.items()}
    )

    gene_x = torch.tensor(gene.values, dtype=torch.float32)
    peak_x = torch.tensor(peak.values, dtype=torch.float32)

    model = DualVAE(
        gene_x=gene_x,
        peak_x=peak_x,
        gene_hidden_dims=config["gene_hidden_dims"],
        peak_hidden_dims=config["peak_hidden_dims"],
        latent_dim=config["latent_dim"],
        gene_activation=config["gene_activation"],
        peak_activation=config["peak_activation"],
        alignment_type=config["alignment_type"],
        fusion_hidden_dims=config["fusion_hidden_dims"],
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"]
    )

    for epoch in range(config["epochs"]):
        model.train()

        beta = beta_schedule(
            epoch,
            config["beta_min"],
            config["beta_max"],
            config["beta_warmup_epochs"]
        )

        optimizer.zero_grad()

        (
            gene_hat, peak_hat,
            mu_gene, logvar_gene, z_gene,
            mu_peak, logvar_peak, z_peak
        ) = model(gene_x, peak_x)

        total_loss, recon_gene, recon_peak, kl_gene, kl_peak, align_loss = loss_fn(
            model=model,
            gene_hat=gene_hat,
            gene_true=gene_x,
            peak_hat=peak_hat,
            peak_true=peak_x,
            mu_gene=mu_gene,
            logvar_gene=logvar_gene,
            mu_peak=mu_peak,
            logvar_peak=logvar_peak,
            gene_recon_loss=config["gene_recon_loss"],
            peak_recon_loss=config["peak_recon_loss"],
            peak_w_zero=config["peak_w_zero"],
            beta=beta,
            align_weight=config["align_weight"],
        )

        total_loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "beta": beta,
            "loss/total": total_loss.item(),
            "loss/gene_recon": recon_gene.item(),
            "loss/peak_recon": recon_peak.item(),
            "loss/kl_gene": kl_gene.item(),
            "loss/kl_peak": kl_peak.item(),
            "loss/alignment": align_loss.item(),
        })

    return model, gene_x, peak_x