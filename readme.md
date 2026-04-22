# PREPROCESSING

---

## 1. data/preprocess_rna.py

RNA preprocessing utilities.

### `load_cell_gene(path)`

Load 10x `.h5` RNA file.  
Input: path to 10x `.h5`  
Output: AnnData (cell × gene)

### `annotate_qc(adata)`

Annotate mitochondrial, ribosomal, hemoglobin genes.  
Input: AnnData  
Output: AnnData

### `compute_qc_metrics(adata)`

Compute per-cell QC metrics.  
Input: AnnData  
Output: AnnData

### `filter_cells_by_qc(adata, ...)`

Filter cells based on QC percentage thresholds.  
Input: AnnData, thresholds  
Output: filtered AnnData

### `filter_cells_genes(adata, min_genes, min_cells)`

Filter low-quality cells and rarely expressed genes.  
Input: AnnData, thresholds  
Output: filtered AnnData

### `detect_doublets(adata)`

Detect doublets using Scrublet.  
Input: AnnData  
Output: AnnData

### `normalize_log1p(adata)`

Normalize counts and apply log(1+x).  
Input: AnnData  
Output: AnnData

### `select_hvgs(adata, n_top_genes)`

Compute highly variable genes.  
Input: AnnData, number of genes  
Output: AnnData

### `run_pca(adata, n_comps)`

Run PCA.  
Input: AnnData, number of components  
Output: AnnData

### `run_neighbors_umap(adata, n_neighbors, n_pcs)`

Build neighbor graph and compute UMAP.  
Input: AnnData, parameters  
Output: AnnData

### `run_leiden(adata)`

Run Leiden clustering.  
Input: AnnData  
Output: AnnData

### `save_leiden_labels(adata, output_path)`

Save Leiden labels to CSV.  
Input: AnnData, path  
Output: CSV file

---

## 2. utils/cell_peak_utils.py

Raw ATAC → cell × peak construction.

### `load_fragments(path)`

Load ATAC fragments file.  
Input: fragments `.tsv`  
Output: DataFrame

### `load_peaks(path)`

Load peak `.bed` file.  
Input: peak file  
Output: DataFrame

### `filter_fragments(frags)`

Filter fragments by chromosome, length, and cell quality.  
Input: fragments DataFrame  
Output: filtered DataFrame

### `filter_peaks(peaks)`

Filter peaks to valid chromosomes.  
Input: peaks DataFrame  
Output: filtered DataFrame

### `assign_peak_ids(peaks)`

Generate peak identifiers (`chr_start_end`).  
Input: peaks DataFrame  
Output: DataFrame

### `map_fragments_to_peaks(frags, peaks)`

Map fragments to peaks by genomic overlap.  
Input: fragments, peaks  
Output: (Barcode, PeakID) DataFrame

### `build_peak_matrix(mapped)`

Construct cell × peak matrix.  
Input: mapped overlaps  
Output: DataFrame (cell × peak)

---

## 3. data/preprocessing_peak.py

Peak-level filtering and analysis.

### `load_cell_peak_df(h5_path)`

Extract cell × peak matrix from 10x multiome `.h5`.  
Input: `.h5` path  
Output: sparse DataFrame

### `filter_valid_chromosomes_df(cell_peak_df)`

Keep peaks on chr1–22, chrX, chrY.  
Input: cell × peak DataFrame  
Output: filtered DataFrame

### `ensure_sparse(X)`

Convert DataFrame to sparse matrix if needed.  
Input: DataFrame or sparse matrix  
Output: sparse matrix

### `compute_peak_activity(X_atac)`

Compute per-peak activity and fraction.  
Input: cell × peak matrix  
Output: peak_counts, peak_frac

### `plot_peak_activity(peak_counts, n_cells)`

Visualize peak activity distribution.  
Input: counts, number of cells  
Output: plot

### `print_peak_threshold_summary(peak_frac)`

Print number of peaks under different activity thresholds.  
Input: peak fractions  
Output: console summary

### `filter_peaks_by_fraction(X_atac, peak_frac, min_frac)`

Filter peaks by minimum activity fraction.  
Input: matrix, fractions, threshold  
Output: filtered matrix, mask

### `compute_sparsity(X)`

Compute sparsity (fraction of zeros).  
Input: matrix  
Output: float

### `log1p_peak_matrix(X)`

Apply log(1+x) to peak matrix.  
Input: matrix  
Output: transformed matrix

---

## 4. data/harmonization.py

Cross-modality alignment.

### `align_cells_between_modalities(df_peak, df_rna)`

Keep only shared cells between RNA and ATAC.  
Input: two DataFrames  
Output: aligned DataFrames

---

## 5. utils/io_utils.py

Matrix saving utilities.

### `save_sparse_matrix(df, save_path)`

Save DataFrame as compressed sparse `.npz` while preserving identifiers.  
Input: DataFrame, path  
Output: `.npz` file

# **GPGVAE: Multi-Omic Variational Autoencoder for PBMC 3k Dataset**

This project implements a **Gene-Peak-Gene Variational Autoencoder (GPGVAE)** model for analyzing multi-omic data from the **PBMC 3k** dataset. The model integrates **gene expression** data with **chromatin accessibility** (peak) data using a shared latent space, providing a method to reconstruct both gene and peak data from the learned representation.

## **Project Overview**

The **GPGVAE** model is designed to learn a joint latent representation of both gene and peak data. By integrating these two data types, it enables better understanding of how gene expression and chromatin accessibility are linked. The model consists of two encoders: one for gene data and one for peak data, and two decoders that reconstruct gene expression and chromatin accessibility values from the shared latent space.

### **Key Components**

- **`gpg_model.py`**:
  - Contains the architecture for the **GPGVAE** model.
  - Implements the **reparameterization trick** to sample from a latent space during training.
  - Includes two encoders (for gene and peak data) and two decoders (to reconstruct peaks and genes).
- **`gpg_train.py`**:
  - Defines the training loop for the **GPGVAE** model.
  - Includes the configuration for optimization and loss functions.
  - Integrates **Weights & Biases (wandb)** for experiment tracking, providing real-time insights into training progress and model performance over time.

- **`gpg_test.py`**:
  - Handles the evaluation of the trained model.
  - Evaluates the model's latent space using **UMAP** (Uniform Manifold Approximation and Projection) for visualization.
  - Assesses **peak prediction performance** using precision, recall, and F1 score, and measures **gene reconstruction performance** with Pearson correlation.

- **`train_data_loader.py`**:
  - A utility for loading preprocessed **gene** and **peak** data from `.npz` files and converting them into pandas DataFrames for further processing.

- **`gpg_config.py`**:
  - Configuration file that stores hyperparameters for the model.
  - Parameters include the latent space dimension, hidden layer sizes, learning rate, and the beta schedule for balancing reconstruction loss and KL divergence during training.

## **Usage Guide**

### **1. Data Requirements**

Before running the model, ensure you have the following preprocessed data available:

- **Gene Expression Data**: Preprocessed gene expression data in `.npz` format.
- **Chromatin Accessibility (Peak) Data**: Preprocessed peak data in `.npz` format.
- **Leiden Cluster Labels**: A CSV file containing cell cluster labels (`rna_leiden_labels.csv`).

The data files must be properly aligned, and cell IDs should be consistent across all files.

### **2. Running the Model**

To train and evaluate the model, follow these steps:

- **Training**: Run the `gpg_train.py` script. This will initialize the model, load the data, and train the model for the specified number of epochs. During training, key metrics such as loss and performance on gene and peak reconstruction will be logged using **wandb**.

- **Evaluation**: After training, the `gpg_test.py` script evaluates the model by performing:
  - **Latent Space Visualization**: UMAP projections of the latent variables (`z1` for genes and `z2` for peaks).
  - **Peak Prediction**: Evaluates the model’s ability to predict peaks from gene data using metrics like precision, recall, and F1 score.
  - **Gene Reconstruction**: Assesses how accurately the model reconstructs gene expression values using **Pearson correlation**.

### **3. Experiment Tracking with wandb**

Throughout training and evaluation, **wandb** logs the following:

- **Loss metrics**: Overall loss, reconstruction loss for genes and peaks, and KL divergence.
- **Performance metrics**: Precision, recall, F1 score for peak prediction, and Pearson correlation for gene reconstruction.
- **Hyperparameters**: Learning rate, beta schedule, and model configurations.

This allows for easy comparison of different experiments and facilitates hyperparameter tuning.

### **4. Model Configuration**

All important hyperparameters for the model are stored in the `gpg_config.py` file. Key parameters include:

- **`latent_dim`**: The dimension of the shared latent space.
- **`hidden_dims`**: The number of hidden units in the model's encoders and decoders.
- **`activation`**: The activation function used in the MLP layers (default is **ReLU**).
- **`lr`**: Learning rate for the optimizer.
- **`epochs`**: Number of epochs for training.
- **`beta_min`** and **`beta_max`**: Beta values for controlling the balance between reconstruction loss and KL divergence during training.

You can adjust these values based on the specific needs of your dataset or experiment.

### **5. Results and Visualizations**

The evaluation script will generate several key visualizations and metrics:

- **UMAP projections** of the latent space, which help in understanding how well the model has learned to cluster the data.
- **Peak prediction performance** using binary classification metrics (precision, recall, F1 score).
- **Gene reconstruction** using Pearson correlation between predicted and true gene values.

These results will be logged in **wandb**, where you can monitor training and evaluation progress over time.
