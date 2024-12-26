import torch
import numpy as np
import scipy.io
import pandas as pd
from scipy.stats import pearsonr
from utils import *
from _misc import *

read_mtx = True


def load_raw_data(data_dir, barcode_path):
    adata = sc.read_10x_mtx(data_dir, var_names="gene_symbols", cache=True)
    barcodes_with_labels = pd.read_csv(barcode_path, sep=",", header=None).iloc[1:]
    barcodes_with_labels.columns = ["barcodes", "labels"]
    barcodes_with_labels = barcodes_with_labels[
        (barcodes_with_labels["labels"] != "Unknown")
    ]
    filtered_barcodes = barcodes_with_labels["barcodes"].values
    adata = adata[adata.obs.index.isin(filtered_barcodes)]
    adata.obs["barcodes"] = adata.obs.index
    adata.obs = adata.obs.reset_index(drop=True)
    adata.obs = adata.obs.merge(barcodes_with_labels, on="barcodes", how="left")
    return adata.X.toarray()


# return ndarray
def _load(filename):
    global read_mtx
    if read_mtx:
        mtx = np.asarray(scipy.io.mmread(f"_data/{filename}.mtx").todense())
    else:
        mtx = torch.load(f"_pt/{filename}.pt").cpu().detach().numpy()
        export_mtx(mtx, f"{filename}")
    return mtx


def qqplots(raw, x_train, x_recon, x_test, gen_test):
    qqplot(raw, "raw", "Raw", "Raw PBMC Data")
    qqplot(x_train, "x_train", "Training Data", "Training Data")
    qqplot(x_recon, "x_recon", "Reconstructed", "Reconstructed Training Data")
    qqplot(x_test, "x_test", "Test Data", "Test Data")
    qqplot(gen_test, "gen_test", "Generated", "Generated Data with Test Labels")


def _cossim(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)


def cossim(x_train, x_recon, x_test, gen_test):
    print("Cosine Similarity")
    print("x_train & x_recon:", _cossim(x_train, x_recon))
    print("x_test & gen_test:", _cossim(x_test, gen_test))


def _pearson(A, B):
    assert A.shape == B.shape, f"Make sure A.shape({A.shape}) == B.shape({B.shape})"
    num_columns = A.shape[1]
    pearson_correlations = []
    for i in range(num_columns):
        if np.std(A[:, i]) == 0 or np.std(B[:, i]) == 0:
            continue
        # corr, _ = pearsonr(A[:, i], B[:, i])
        corr = np.corrcoef(A[:, i], B[:, i])[0, 1]
        pearson_correlations.append(corr)
    return np.mean(pearson_correlations)


def pearson(x_train, x_recon, x_test, gen_test):
    print("Average Pearson Correlation Coefficient")
    print("x_train & x_recon:", _pearson(x_train, x_recon))
    print("x_test & gen_test:", _pearson(x_test, gen_test))


def _sel(A):
    flat_A = A.flatten()
    top_100_indices = np.argsort(flat_A)[-100:]
    rows, cols = np.unravel_index(top_100_indices, A.shape)
    top_100_columns = np.unique(cols)
    return top_100_columns


def _corr(A, B, method):
    assert A.shape == B.shape, f"Make sure A.shape({A.shape}) == B.shape({B.shape})"
    if method == "frobenius":
        return np.linalg.norm(A - B, "fro")
    elif method == "mean_difference":
        return np.mean(np.abs(A - B))


def corr(x_train, x_recon, x_test, gen_test):
    # 100 most highly variable genes
    cols = _sel(x_train)
    x_train = x_train[:, cols]
    x_recon = x_recon[:, cols]

    cols = _sel(x_test)
    x_test = x_test[:, cols]
    gen_test = gen_test[:, cols]

    print("Correlation Discrepancy")
    print("x_train & x_recon:", _corr(x_train, x_recon, "mean_difference"))
    print("x_test & gen_test:", _corr(x_test, gen_test, "mean_difference"))


def _stdscale(arr):
    return (arr - np.mean(arr, axis=0)) / np.std(arr, axis=0)


def _stdscale_alt(arr, arr2):
    return (arr - np.mean(arr2, axis=0)) / np.std(arr2, axis=0)


def _stdscale2(arr):
    return (arr - np.mean(arr)) / np.std(arr)


def _flatten(arr):
    return np.sort(arr.flatten())


def _nz(arr):
    return np.sort(arr[arr != 0])


def validate():
    col = get_colormap()
    d = get_celltype2int_dict()

    x_raw = load_raw_data("data/", "data/barcode_to_celltype.csv")
    labels_raw = get_labels("labels_raw", d)

    x_train = _load("x_train")
    labels_train = get_labels("labels_train", d)

    x_recon = _load("x_recon")
    labels_recon = labels_train

    x_test = _load("x_test")
    labels_test = get_labels("labels_test", d)

    gen_test = _load("gen_test")
    labels_gen_test = get_labels("labels_gen_test", d)

    plot_umap(x_raw, labels_raw, col, d, "UMAP of Raw Data", "umap_raw")
    plot_umap(x_train, labels_train, col, d, "UMAP of Training Data", "umap_x_train")
    plot_umap(
        x_recon, labels_recon, col, d, "UMAP of Reconstructed Data", "umap_x_recon"
    )
    plot_umap(x_test, labels_test, col, d, "UMAP of Testing Data", "umap_x_train")
    plot_umap(
        gen_test, labels_gen_test, col, d, "UMAP of Data Generated", "umap_gen_test"
    )

    # pearson(x_train, x_recon, x_test, gen_test)
    # corr(x_train, x_recon, x_test, gen_test)

    train, recon, test, gen = (
        _stdscale2(x_train),
        _stdscale2(x_recon),
        _stdscale2(x_test),
        _stdscale2(gen_test),
    )
    pearson(train, recon, test, gen)
    corr(train, recon, test, gen)

    train, recon, test, gen = (
        _stdscale(_flatten(x_train)),
        _stdscale(_flatten(x_recon)),
        _stdscale(_flatten(x_test)),
        _stdscale(_flatten(gen_test)),
    )
    cossim(train, recon, test, gen)

    train, recon, test, gen = (
        _stdscale(_nz(x_train)),
        _stdscale_alt(_nz(x_recon), _nz(x_train)),
        _stdscale(_nz(x_test)),
        _stdscale_alt(_nz(gen_test), _nz(x_test)),
    )
    qqplots(_stdscale(_nz(x_raw)), train, recon, test, gen)


if __name__ == "__main__":
    validate()
