import torch
import wandb
from torch.utils.data import TensorDataset, DataLoader

from multiomic_vae.models.gpg_model import GPGVAE
from multiomic_vae.config.gpg_config import config


def beta_schedule(epoch, beta_min, beta_max, warmup_epochs):
    if epoch >= warmup_epochs:
        return beta_max
    return beta_min + (beta_max - beta_min) * (epoch / warmup_epochs)


def gpg_train(gene, peak):
    wandb.init(
        project="GPG Model PBMC 10k - newnew-to delete",
        config={k: str(v) for k, v in config.items()}
    )

    gene_x = torch.tensor(gene.values, dtype=torch.float32)
    peak_x = torch.tensor(peak.values, dtype=torch.float32)

    dataset = TensorDataset(gene_x, peak_x)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False
    )

    model = GPGVAE(
        gene_x=gene_x,
        peak_x=peak_x,
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        activation=config["activation"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    model.train()

    for epoch in range(config["epochs"]):
        beta = beta_schedule(
            epoch,
            config["beta_min"],
            config["beta_max"],
            config["beta_warmup_epochs"]
        )

        epoch_loss = 0.0
        epoch_gene_recon = 0.0
        epoch_peak_recon = 0.0
        epoch_kl1 = 0.0
        epoch_kl2 = 0.0
        n_batches = 0

        for gene_batch, peak_batch in dataloader:
            optimizer.zero_grad()

            gene_hat, peak_hat, mu1, logvar1, mu2, logvar2 = model(gene_batch)

            recon_gene = config["recon_loss"](gene_hat, gene_batch)
            recon_peak = config["recon_loss"](peak_hat, peak_batch)

            kl1 = -0.5 * torch.mean(1 + logvar1 - mu1**2 - logvar1.exp())
            kl2 = -0.5 * torch.mean(1 + logvar2 - mu2**2 - logvar2.exp())

            loss = recon_gene + recon_peak + beta * (kl1 + kl2)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_gene_recon += recon_gene.item()
            epoch_peak_recon += recon_peak.item()
            epoch_kl1 += kl1.item()
            epoch_kl2 += kl2.item()
            n_batches += 1

        wandb.log({
            "epoch": epoch,
            "beta": beta,
            "loss/total": epoch_loss / n_batches,
            "loss/gene_recon": epoch_gene_recon / n_batches,
            "loss/peak_recon": epoch_peak_recon / n_batches,
            "loss/kl1": epoch_kl1 / n_batches,
            "loss/kl2": epoch_kl2 / n_batches,
        })

    return model, gene_x, peak_x