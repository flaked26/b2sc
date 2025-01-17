import torch
import torch.nn as nn
from models import GaussianMixtureVAE
from _misc import *


def test_recon(x_train, labels_train, mus, logvars, model, color_map, mapping_dict):
    x_recon, zs = model.module.decode_with_labels(mus, logvars, labels_train)
    x_recon = x_recon.cpu().detach().numpy()
    export_mtx(x_recon, "x_recon")
    plot_umap(
        zs.cpu().detach().numpy(),
        labels_train.cpu().detach().numpy(),
        color_map,
        mapping_dict,
        "UMAP of Latent Z",
        "umap_latent",
    )
