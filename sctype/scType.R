# load libraries
lapply(c("dplyr", "Seurat", "HGNChelper"), library, character.only = T)

# Load the PBMC dataset
pbmc.data <- Read10X(data.dir = "data/")
# Initialize the Seurat object with the raw (non-normalized data).
gene_expression_matrix <- pbmc.data
pbmc <- CreateSeuratObject(counts = gene_expression_matrix, project = "pbmc3k", min.cells = 3, min.features = 200)

# normalize data
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# make some filtering based on QC metrics visualizations, see Seurat tutorial: https://satijalab.org/seurat/articles/pbmc3k_tutorial.html
# pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)

pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", scale.factor = 10000)

pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)

# scale and run PCA
pbmc <- ScaleData(pbmc, features = rownames(pbmc))

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# Check number of PC components (we selected 10 PCs for downstream analysis, based on Elbow plot)
ElbowPlot(pbmc)

# cluster and visualize
pbmc <- FindNeighbors(pbmc, dims = 1:10)
pbmc <- FindClusters(pbmc, resolution = 0.8)

pbmc <- RunUMAP(pbmc, dims = 1:10)

DimPlot(pbmc, reduction = "umap")

saveRDS(pbmc, file = "sctype/pbmc_tutorial.rds")

# load gene set preparation function
source("sctype/source/gene_sets_prepare.R")
# load cell type annotation function
source("sctype/source/sctype_score_.R")
# DB file
db_ <- "sctype/source/ScTypeDB_full.xlsx"

# e.g. Immune system,Pancreas,Liver,Eye,Kidney,Brain,Lung,Adrenal,Heart,Intestine,Muscle,Placenta,Spleen,Stomach,Thymus
tissue <- "Immune system"

# prepare gene sets
gs_list <- gene_sets_prepare(db_, tissue)

str(pbmc[["RNA"]])

# check Seurat object version (scRNA-seq matrix extracted differently in Seurat v4/v5)
seurat_package_v5 <- isFALSE("counts" %in% names(attributes(pbmc[["RNA"]])))
print(sprintf("Seurat object %s is used", ifelse(seurat_package_v5, "v5", "v4")))

# extract scaled scRNA-seq matrix
scRNAseqData_scaled <- if (seurat_package_v5) as.matrix(pbmc[["RNA"]]$scale.data) else as.matrix(pbmc[["RNA"]]@scale.data)

# get cell-type by cell matrix
es.max <- sctype_score(
  scRNAseqData = scRNAseqData_scaled, scaled = TRUE,
  gs = gs_list$gs_positive, gs2 = gs_list$gs_negative
)
# NOTE: scRNAseqData parameter should correspond to your input scRNA-seq matrix.
# In case Seurat is used, it is either pbmc[["RNA"]]@scale.data (default), pbmc[["SCT"]]@scale.data, in case sctransform is used for normalization,
# or pbmc[["integrated"]]@scale.data, in case a joint analysis of multiple single-cell datasets is performed.

# merge by cluster
cL_resutls <- do.call("rbind", lapply(unique(pbmc@meta.data$seurat_clusters), function(cl) {
  es.max.cl <- sort(rowSums(es.max[, rownames(pbmc@meta.data[pbmc@meta.data$seurat_clusters == cl, ])]), decreasing = !0)
  head(data.frame(cluster = cl, type = names(es.max.cl), scores = es.max.cl, ncells = sum(pbmc@meta.data$seurat_clusters == cl)), 10)
}))
sctype_scores <- cL_resutls %>%
  group_by(cluster) %>%
  top_n(n = 1, wt = scores)

# set low-confident (low ScType score) clusters to "unknown"
sctype_scores$type[as.numeric(as.character(sctype_scores$scores)) < sctype_scores$ncells / 4] <- "Unknown"
print(sctype_scores[, 1:3])
pbmc@meta.data$customclassif <- ""
for (j in unique(sctype_scores$cluster)) {
  cl_type <- sctype_scores[sctype_scores$cluster == j, ]
  pbmc@meta.data$customclassif[pbmc@meta.data$seurat_clusters == j] <- as.character(cl_type$type[1])
}
DimPlot(pbmc, reduction = "umap", label = TRUE, repel = TRUE, group.by = "customclassif")
barcodes <- read.table("data/barcodes.tsv", header = TRUE, sep = "\t")

# assuming `pbmc` is your Seurat object
seurat_barcodes <- rownames(pbmc)

# Replace `Barcode` with the column name in `barcodes.tsv` containing the barcodes
all(barcodes$barcodes %in% seurat_barcodes)
cell_types <- pbmc@meta.data$customclassif
barcodes_df <- data.frame(Barcode = rownames(pbmc))
cell_types_df <- data.frame(Barcode = rownames(pbmc@meta.data), CellType = pbmc@meta.data$customclassif)

# Assuming `pbmc` is your Seurat object and `customclassif` is the column with cell type annotations
mapping <- data.frame(
  Barcode = rownames(pbmc@meta.data),
  CellType = pbmc@meta.data$customclassif
)

# Checking the first few rows of the mapping to ensure it's correct
head(mapping)

# Save as a CSV file
write.table(mapping, file = "data/barcode_to_celltype.csv", sep = ",", row.names = FALSE, quote = FALSE)
# Or save as a TSV file
write.table(mapping, file = "data/barcode_to_celltype.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
