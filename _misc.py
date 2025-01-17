import umap
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.sparse import csr_matrix
import scipy.stats as stats
from scipy import signal
import pandas as pd

np.set_printoptions(suppress=True)


def export_mtx(x, filename):
    scipy.io.mmwrite(f"_data/{filename}.mtx", csr_matrix(x))


def save_labels(labels, d, filename):
    # convert celltype numbers to strings
    celltype_d = {v: k for k, v in d.items()}
    vec_func = np.vectorize(lambda x: celltype_d[x] if x in celltype_d.keys() else x)
    celltypes = vec_func(labels)
    np.savetxt(f"_data/{filename}.csv", celltypes, fmt="%s")


def get_labels(filename, d):
    data = np.genfromtxt(
        f"_data/{filename}.csv", dtype=str, delimiter=",", skip_header=0
    )
    return np.vectorize(d.get)(data)


def plot_umap(x, labels, color_map, mapping_dict, title, filename):
    # Plot UMAP with get_colormap().
    label_map = {v: k for k, v in mapping_dict.items()}
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(x)
    plt.figure(figsize=(12, 10))
    # import pdb; pdb.set_trace()
    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=[color_map[label_map[label.item()]] for label in labels],
        s=5,
    )
    # Remove ticks
    plt.xticks([])
    plt.yticks([])
    # Name the axes.
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.title(f"{title}")
    plt.savefig(f"_plots/{filename}.png")
    plt.close()


def qqplot(arr, filename, legend, title):
    stats.probplot(arr, dist="norm", plot=plt)
    plt.legend([legend, "Normal"])
    plt.title(title)
    plt.savefig(f"_plots/qq_{filename}.png")
    # plt.show()
    plt.close()
