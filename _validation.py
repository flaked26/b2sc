import torch
import numpy as np
import scipy.io
import pandas as pd
import seaborn as sns
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
from utils import *
from _misc import *

read_mtx = True
_integrate = True


# return ndarray
def _load(filename):
    global read_mtx
    if read_mtx:
        mtx = np.asarray(scipy.io.mmread(f"_data/{filename}.mtx").todense())
    else:
        mtx = torch.load(f"_pt/{filename}.pt").cpu().detach().numpy()
        export_mtx(mtx, f"{filename}")
    return mtx


def resample(data, size, meth):
    #1dim
    if meth == "linspace":
        indices = np.linspace(0, len(data) - 1, size).astype(int)
        data = data[indices]
    elif meth == "fourier":
        data = signal.resample(data, size)
    elif meth == "pd":
        data = pd.Series(data)
        data = data.interpolate(method="linear")[:: len(data) // size]
        data = data.to_numpy()
    #2dim
    elif meth == "bootstrap":
        selected_rows = np.random.choice(data.shape[0], size=size, replace=True)
        data = data[selected_rows, :]
    elif meth == "uniform":
        data = data[::2]
    else:
        data = np.sort(data)
    return data


def _integ(m1, m2):
    integrated = np.vstack((m1, m2))
    size = integrated.shape[0] // 2
    integrated = resample(integrated, size, "bootstrap")
    return integrated


def _stdscale_old(arr):
    arr = (arr - np.mean(arr)) / np.std(arr)
    arr = np.nan_to_num(arr, nan=0)
    return arr


def _stdscale(arr):
    gene_mean = np.mean(arr)
    gene_std = np.std(arr)
    return np.divide((arr - gene_std), gene_mean, where=gene_mean != 0)


def _stdscale2(arr):
    gene_mean = np.mean(arr, axis=0)
    gene_std = np.std(arr, axis=0)
    return np.divide((arr - gene_std), gene_mean, where=gene_mean != 0)


def _flatten(arr):
    return np.sort(arr.flatten())


def _nz(arr):
    return np.sort(arr[arr != 0])


def _hvg(expr_matrix):
    # 每个细胞的表达值标准化，使其总表达量为10000
    expr_matrix_normalized = (
        expr_matrix / np.sum(expr_matrix, axis=1, keepdims=True) * 1e4
    )
    expr_matrix_normalized = np.nan_to_num(expr_matrix_normalized, nan=0)
    # 对数据进行log1p转换
    expr_matrix_log = np.log1p(expr_matrix_normalized)
    # 排除前10%的平均表达量最高的基因
    mean_expression = np.mean(expr_matrix_log, axis=0)
    threshold = np.percentile(mean_expression, 90)
    high_expression_genes = mean_expression > threshold
    filtered_expr_matrix = expr_matrix_log[:, ~high_expression_genes]
    expr_matrix_log = filtered_expr_matrix
    # 计算每个基因在所有细胞中的均值和标准差
    gene_mean = np.mean(expr_matrix_log, axis=0)
    gene_std = np.std(expr_matrix_log, axis=0)
    # 计算变异系数（CV）
    cv = np.divide(gene_std, gene_mean, where=gene_mean != 0)
    # 按变异系数排序
    sorted_idx = np.argsort(cv)[::-1]
    # 选择前100个高可变基因
    top_100_genes_idx = sorted_idx[:100]
    return expr_matrix[:, top_100_genes_idx]


def heatmap(data, filename, title):
    top_100_expr_matrix = _hvg(data)
    pearson_corr_matrix = np.corrcoef(top_100_expr_matrix.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pearson_corr_matrix,  # 相关系数矩阵
        annot=False,  # 在每个方格中显示相关系数
        fmt=".2f",  # 设置数值格式为保留两位小数
        cmap="coolwarm",  # 设置热图颜色映射
        vmin=-1,
        vmax=1,  # 设置相关系数的范围
        square=True,  # 使热图为正方形
        cbar_kws={"shrink": 0.8},  # 设置颜色条的大小
    )
    plt.title(f"Correlation Heatmap of {title}")
    plt.savefig(f"_plots/heatmap_{filename}")
    # plt.show()
    plt.close("all")


def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return 0
    return dot_product / (norm_u * norm_v)


def _cossim(A, B):
    assert A.shape == B.shape, f"Make sure A.shape({A.shape}) == B.shape({B.shape})"
    n = A.shape[1]
    total_similarity = 0
    count = 0
    for i in range(n):
        u = A[:, i]
        v = B[:, i]
        total_similarity += cosine_similarity(u, v)
        count += 1
    return total_similarity / count


def cossim(x_train, x_recon, x_test, gen_test):
    print("Cosine Similarity")
    print("x_train & x_recon:", _cossim(x_train, x_recon))
    print("x_test & gen_test:", _cossim(x_test, gen_test))


def pearson_correlation_coefficient(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator


def _pearson(A, B):
    assert A.shape == B.shape, f"Make sure A.shape({A.shape}) == B.shape({B.shape})"
    num_columns = A.shape[1]
    pearson_correlations = []
    for i in range(num_columns):
        if np.std(A[:, i]) == 0 or np.std(B[:, i]) == 0:
            continue
        # corr, _ = pearsonr(A[:, i], B[:, i])
        # corr = np.corrcoef(A[:, i], B[:, i])[0, 1]
        corr = pearson_correlation_coefficient(A[:, i], B[:, i])
        pearson_correlations.append(corr)
    return np.mean(pearson_correlations)


def pearson(x_train, x_recon, x_test, gen_test):
    print("Average Pearson Correlation Coefficient")
    print("x_train & x_recon:", _pearson(x_train, x_recon))
    print("x_test & gen_test:", _pearson(x_test, gen_test))


def spearman_correlation_matrix(data):
    n = data.shape[1]
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                # corr_coefficient = np.corrcoef(data[:, i], data[:, j], method="spearman")[0, 1]
                _, corr_coefficient = spearmanr(data[:, i], data[:, j])
                corr_matrix[i, j] = corr_coefficient
            else:
                corr_matrix[i, j] = 1  # 对角线元素为1
    return corr_matrix


def _cd(A, B, method):
    assert A.shape == B.shape, f"Make sure A.shape({A.shape}) == B.shape({B.shape})"
    if method == "frobenius":
        return np.linalg.norm(A - B, "fro")
    elif method == "mean_difference":
        return np.mean(np.abs(A - B))
    elif method == "spearman":
        CA = spearman_correlation_matrix(A)
        CB = spearman_correlation_matrix(B)
        n = CA.shape[0]
        CD = 0
        for j in range(n):
            CD = max(CD, np.sum(np.abs(CA[:, j] - CB[:, j])))
        return CD


def cd(x_train, x_recon, x_test, gen_test):
    print("Correlation Discrepancy")
    print("x_train & x_recon:", _cd(_hvg(x_train), _hvg(x_recon), "spearman"))
    print("x_test & gen_test:", _cd(_hvg(x_test), _hvg(gen_test), "spearman"))


def do_stdscale(x_raw, x_train, x_recon, x_test, gen_test):
    x_raw, x_train, x_recon, x_test, gen_test = (
        _stdscale(x_raw),
        _stdscale(x_train),
        _stdscale(x_recon),
        _stdscale(x_test),
        _stdscale(gen_test),
    )
    heatmap(x_raw, f"x_raw", "Raw Data")
    heatmap(x_train, f"x_train", "Training Data")
    heatmap(x_recon, f"x_recon", "Reconstructed Data")
    heatmap(x_test, f"x_test", "Test Data")
    heatmap(gen_test, f"gen_test", "Generated Data")

    cossim(x_train, x_recon, x_test, gen_test)
    cd(x_train, x_recon, x_test, gen_test)


def validate():
    col = get_colormap()
    d = get_celltype2int_dict()

    x_raw = _load("x_raw")
    labels_raw = get_labels("labels_raw", d)

    x_train = _load("x_train")
    labels_train = get_labels("labels_train", d)

    x_recon = _load("x_recon")
    labels_recon = labels_train
    if _integrate:
        x_recon = _integ(x_train, x_recon)

    x_test = _load("x_test")
    labels_test = get_labels("labels_test", d)

    gen_test = _load("gen_test")
    labels_gen_test = get_labels("labels_gen_test", d)
    if _integrate:
        gen_test = _integ(x_test, gen_test)

    plot_umap(x_raw, labels_raw, col, d, "UMAP of Raw Data", "umap_raw")
    plot_umap(x_train, labels_train, col, d, "UMAP of Training Data", "umap_x_train")
    plot_umap(x_recon, labels_recon, col, d, "UMAP of Reconstructed", "umap_x_recon")
    plot_umap(x_test, labels_test, col, d, "UMAP of Testing Data", "umap_x_train")
    plot_umap(gen_test, labels_gen_test, col, d, "UMAP of Generated", "umap_gen_test")

    qqplot(_stdscale_old(_nz(x_raw)), "x_raw", "Raw", "Raw PBMC Data")
    qqplot(_stdscale_old(_nz(x_train)), "x_train", "Training Data", "Training Data")
    qqplot(_stdscale_old(_nz(x_recon)), "x_recon", "Reconstructed", "Reconstructed")
    qqplot(_stdscale_old(_nz(x_test)), "x_test", "Test Data", "Test Data")
    qqplot(_stdscale_old(_nz(gen_test)), "gen_test", "Generated", "Generated")

    pearson(x_train, x_recon, x_test, gen_test)

    do_stdscale(x_raw, x_train, x_recon, x_test, gen_test)


if __name__ == "__main__":
    validate()
