import torch
import numpy as np
import umap
import matplotlib.pyplot as plt
import torch.nn.functional as F
from _misc import *


def zinb_loss(y_true, y_pred, pi, r, eps=1e-10):
    y_true = y_true.float()
    y_pred = y_pred.float()

    # Negative binomial part
    nb_case = (
        -torch.lgamma(r + eps)
        + torch.lgamma(y_true + r + eps)
        - torch.lgamma(y_true + 1.0)
        + r * torch.log(pi + eps)
        + y_true * torch.log(1.0 - (pi + eps))
    )

    # Zero-inflated part
    zero_nb = torch.pow(pi, r)
    zero_case = torch.where(
        y_true < eps,
        -torch.log(
            zero_nb + (1.0 - zero_nb) * torch.exp(-r * torch.log(1.0 - pi + eps))
        ),
        torch.tensor(0.0, device=y_true.device),
    )

    return -torch.mean(zero_case + nb_case)


def train_GMVAE(
    x_train,
    labels_train,
    model,
    epoch,
    dataloader,
    optimizer,
    proportion_tensor,
    kl_weight,
    mapping_dict,
    color_map,
    max_epochs,
    device="cuda",
):
    model.train()
    total_loss = 0
    model = model.to(device)

    for idx, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # forward
        reconstructed, mus, logvars, pis, zs = model(data, labels)

        proportion_tensor_reshaped = proportion_tensor.to(pis.device)
        fraction_loss = F.mse_loss(pis.mean(0), proportion_tensor_reshaped)
        loss_recon = F.mse_loss(reconstructed, data)
        # loss_recon = F.cross_entropy(reconstructed, data, reduction='sum')

        zinb_loss_val = zinb_loss(
            data, reconstructed, model.module.prob_extra_zero, model.module.over_disp
        )

        loss_kl = 0.5 * torch.sum(-1 - logvars + mus.pow(2) + logvars.exp()) / 1e10
        loss_kl = loss_kl * kl_weight
        loss_kl = 1 if loss_kl > 1 else loss_kl

        loss = loss_recon + loss_kl + fraction_loss + zinb_loss_val
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        pis = pis.mean(0)
        print(pis)
        print(
            f"Epoch: {epoch+1} KL Loss: {loss_kl:.4f} Recon Loss: {loss_recon:.4f} Total Loss: {total_loss:.4f} Fraction Loss: {fraction_loss:.4f} ZINB Loss: {zinb_loss_val:.4f}"
        )

    if (epoch + 1) % 100 == 0:
        ############################  reconstruction with train data
        x_recon, zs = model.module.decode_with_labels(mus, logvars, labels_train)
        x_recon = x_recon.cpu().detach().numpy()
        export_mtx(x_recon, "x_recon")
        plot_umap(zs.cpu().detach().numpy(), labels_train.cpu().detach().numpy(), color_map, mapping_dict, "UMAP of Latent Z", "umap_latent")

        # Save reconstructed.
        # torch.save(reconstructed, "_pt/x_recon_training.pt")
        # export_mtx(reconstructed.cpu().detach().numpy(), "x_recon_training")

        # save for reconstruction testing
        torch.save(mus, "_pt/GMVAE_mus.pt")
        torch.save(logvars, "_pt/GMVAE_logvars.pt")
        torch.save(pis, "_pt/GMVAE_pis.pt")

        mus = mus.mean(0)
        logvars = logvars.mean(0)
        pis = pis.mean(0)

        # Save the mean, logvar, and pi.
        torch.save(mus, "_pt/GMVAE_mus_mean.pt")
        torch.save(logvars, "_pt/GMVAE_logvars_mean.pt")
        torch.save(pis, "_pt/GMVAE_pis_mean.pt")
        print("GMVAE mu & var & pi saved.")

        model.eval()

        z = zs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        reconstructed = reconstructed.cpu().detach().numpy()

        torch.save(model.state_dict(), "_pt/GMVAE_model.pt")
        print("GMVAE Model saved.")

    return total_loss, mus, logvars, pis
