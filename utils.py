import scanpy as sc
import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse

from _misc import *

# Set random seed.
np.random.seed(4)
# Set torch seed.
torch.manual_seed(4)


def get_celltype2int_dict():
    mapping_dict = {
        "Naive B cells": 0,
        "Non-classical monocytes": 1,
        "Classical Monocytes": 2,
        "Natural killer  cells": 3,
        "CD8+ NKT-like cells": 4,
        "Memory CD4+ T cells": 5,
        "Naive CD8+ T cells": 6,
        "Platelets": 7,
        "Pre-B cells": 8,
        "Plasmacytoid Dendritic cells": 9,
        "Effector CD4+ T cells": 10,
        "Macrophages": 11,
        "Myeloid Dendritic cells": 12,
        "Effector CD8+ T cells": 13,
        "Plasma B cells": 14,
        "Memory B cells": 15,
        "Naive CD4+ T cells": 16,
        "Progenitor cells": 17,
        "γδ-T cells": 18,
        "Eosinophils": 19,
        "Neutrophils": 20,
        "Basophils": 21,
        "Mast cells": 22,
        "Intermediate monocytes": 23,
        "Megakaryocyte": 24,
        "Endothelial": 25,
        "Erythroid-like and erythroid precursor cells": 26,
        "HSC/MPP cells": 27,
        "Granulocytes": 28,
        "ISG expressing immune cells": 29,
        "Cancer cells": 30,
        "Memory CD8+ T cells": 31,
        "Pro-B cells": 32,
        "Immature B cells": 33,
    }
    return mapping_dict


def get_celltype2strint_dict():
    mapping_dict = get_celltype2int_dict()
    return {key: str(value) for key, value in mapping_dict.items()}


def get_colormap():
    color_map = {
        "Naive B cells": "red",
        "Non-classical monocytes": "black",
        "Classical Monocytes": "orange",
        "Natural killer  cells": "cyan",
        "CD8+ NKT-like cells": "pink",
        "Memory CD4+ T cells": "magenta",
        "Naive CD8+ T cells": "blue",
        "Platelets": "yellow",
        "Pre-B cells": "cornflowerblue",
        "Plasmacytoid Dendritic cells": "lime",
        "Effector CD4+ T cells": "grey",
        "Macrophages": "tan",
        "Myeloid Dendritic cells": "green",
        "Effector CD8+ T cells": "brown",
        "Plasma B cells": "purple",
        "Memory B cells": "darkred",
        "Naive CD4+ T cells": "darkblue",
        "Progenitor cells": "darkgreen",
        "γδ-T cells": "darkcyan",
        "Eosinophils": "darkolivegreen",
        "Neutrophils": "darkorchid",
        "Basophils": "darkred",
        "Mast cells": "darkseagreen",
        "Intermediate monocytes": "darkslateblue",
        "Megakaryocyte": "darkslategrey",
        "Endothelial": "darkturquoise",
        "Erythroid-like and erythroid precursor cells": "darkviolet",
        "HSC/MPP cells": "deeppink",
        "Granulocytes": "deepskyblue",
        "ISG expressing immune cells": "dimgray",
        "Cancer cells": "dodgerblue",
        "Memory CD8+ T cells": "darkkhaki",
        "Pro-B cells": "darkorange",
        "Immature B cells": "darkgoldenrod",
        # 'CD4+ NKT-like cells': 'darkmagenta',
    }
    return color_map


def load_data(data_dir, barcode_path, _retrain):
    adata = sc.read_10x_mtx(data_dir, var_names="gene_symbols", cache=False)

    # Read barcodes_with_labels tsv file.
    barcodes_with_labels = pd.read_csv(barcode_path, sep=",", header=None).iloc[1:]
    barcodes_with_labels.columns = ["barcodes", "labels"]

    # Remove rows with 'unknown' or NaN labels
    barcodes_with_labels = barcodes_with_labels[
        barcodes_with_labels["labels"] != "Unknown"
    ]

    # Cleaned labels after filtering
    labels = barcodes_with_labels["labels"].values
    filtered_barcodes = barcodes_with_labels["barcodes"].values
    adata = adata[adata.obs.index.isin(filtered_barcodes)]
    adata.obs["barcodes"] = adata.obs.index
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs = adata.obs.merge(barcodes_with_labels, on="barcodes", how="left")
    adata.X = adata.X.toarray()
    adata.obs.index = adata.obs.index.astype(str)
    mapping_dict = get_celltype2strint_dict()
    adata.obs["labels"] = adata.obs["labels"].replace(mapping_dict)
    adata.obs["labels"] = adata.obs["labels"].astype("category")

    # Assuming adata is your DataFrame and labels is adata.obs['labels']
    labels = adata.obs["labels"]

    # Create a custom mapping from unique strings to integers
    label_to_int = {}
    max_int_label = -1

    for label in labels.unique():
        if label.isdigit():  # Check if the label is numeric
            int_val = int(label)
            label_to_int[label] = int_val
            max_int_label = max(max_int_label, int_val)
        else:
            # For non-numeric labels, assign a unique integer
            # starting from the next integer after the highest numeric label
            if label not in label_to_int:
                max_int_label += 1
                label_to_int[label] = max_int_label

    # Map the labels in your DataFrame to integers
    int_labels = labels.map(label_to_int)

    # Convert the Series of integers to a LongTensor
    labels = torch.LongTensor(int_labels.values)
    X_tensor = torch.Tensor(adata.X)

    # randomly sample 80% of the data.
    rand_index = np.random.choice(
        X_tensor.shape[0], int(0.8 * X_tensor.shape[0]), replace=False
    )
    X_train = X_tensor[rand_index]
    labels_train = labels[rand_index]
    dataset_train = TensorDataset(X_tensor, labels)
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of labels_train: {labels_train.shape}")

    # Select those not in the random index to be the test set.
    mask = np.isin(np.arange(X_tensor.shape[0]), rand_index, invert=True)
    X_test = X_tensor[mask]
    labels_test = labels[mask]
    dataset_test = TensorDataset(X_test, labels_test)
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of labels_test: {labels_test.shape}")

    if _retrain:
        d = get_celltype2int_dict()

        export_mtx(adata.X, "x_raw")
        save_labels(labels.cpu().detach().numpy(), d, "labels_raw")

        export_mtx(X_train.cpu().detach().numpy(), "x_train")
        save_labels(labels_train.cpu().detach().numpy(), d, "labels_train")

        export_mtx(X_test.cpu().detach().numpy(), "x_test")
        save_labels(labels_test.cpu().detach().numpy(), d, "labels_test")

    # Cell type fractions
    cell_type_fractions = np.unique(adata.obs["labels"].values, return_counts=True)[
        1
    ] / len(adata.obs["labels"].values)

    return (
        dataset_train,
        X_train,
        labels_train,
        dataset_test,
        X_test,
        labels_test,
        cell_type_fractions,
    )


def configure(data_dir, barcode_path, _retrain):
    (
        dataset_train,
        X_train,
        labels_train,
        dataset_test,
        X_test,
        labels_test,
        cell_type_fractions,
    ) = load_data(data_dir=data_dir, barcode_path=barcode_path, _retrain=_retrain)

    num_cells = X_train.shape[0]
    num_genes = X_train.shape[1]

    parser = argparse.ArgumentParser(description="Process neural network parameters.")

    args = parser.parse_args()
    args.num_cells = num_cells
    # args.learning_rate = 1e-3
    args.hidden_dim = 600
    args.latent_dim = 300
    args.train_GMVAE_epochs = 100
    args.bulk_encoder_epochs = 1000
    # args.dropout = 0.05
    args.batch_size = num_cells
    args.input_dim = num_genes

    args.X_train = X_train
    args.labels_train = labels_train
    args.X_test = X_test
    args.labels_test = labels_test

    ###### dataloader
    dataloader = DataLoader(
        dataset_train, batch_size=num_cells // 5, shuffle=True, drop_last=True
    )
    args.dataloader = dataloader

    testloader = DataLoader(
        dataset_test, batch_size=X_test.shape[0], shuffle=True, drop_last=True
    )
    args.testloader = testloader

    args.mapping_dict = get_celltype2int_dict()
    args.color_map = get_colormap()
    args.K = 34

    unique_cell_types = np.unique(labels_train)
    # Create a dictionary mapping from cell type to its fraction
    cell_type_to_fraction = {
        cell_type: fraction
        for cell_type, fraction in zip(unique_cell_types, cell_type_fractions)
    }
    cell_type_fractions_ = []
    for i in range(args.K):
        # Append the fraction if the cell type is present, else append 0
        cell_type_fractions_.append(cell_type_to_fraction.get(i, 0))
    args.cell_type_fractions = torch.FloatTensor(np.array(cell_type_fractions_))
    print("args.cell_type_fractions\n", args.cell_type_fractions)

    label_map = {str(v): k for k, v in args.mapping_dict.items()}
    args.label_map = label_map

    print("Configuration is complete.")
    return args


def get_saved_GMM_params(mus_path, vars_path):
    gmm_mus_celltypes = torch.load(mus_path).squeeze().T
    gmm_vars_celltypes = torch.load(vars_path).squeeze().T
    return gmm_mus_celltypes, gmm_vars_celltypes
