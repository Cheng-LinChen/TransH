import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from openTSNE import TSNE
import os

def plot_pca_embeddings(embedding_file, output_dir="embedding_pca_plots"):
    """
    Load embeddings and visualize them using PCA (2D).
    Includes Variance Analysis and Scree Plot for quality assessment.
    
    embedding_file format:
    index,name,dim_0,dim_1,...dim_n
    """
    # Load data
    df = pd.read_csv(embedding_file)
    # Keep entity names for optional labeling
    names = df['entity_idx'].astype(str).to_list()

    # Select only numeric embedding columns
    embedding_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = df[embedding_cols].to_numpy(dtype=float)


    n_entities, dim = embeddings.shape
    print(f"[PCA] Loaded {n_entities} embeddings with dimension {dim}")

    os.makedirs(output_dir, exist_ok=True)

    # PCA projection (for the 2D plot)
    pca_2d = PCA(n_components=2, random_state=42)
    emb_pca_2d = pca_2d.fit_transform(embeddings)

    # Embedding Quality Analysis (Variance)
    explained_variance = pca_2d.explained_variance_ratio_
    cumulative_variance = explained_variance.sum()
    
    print("-" * 40)
    print(f"[PCA] PC1 captures: {explained_variance[0]*100:.2f}% variance")
    print(f"[PCA] PC2 captures: {explained_variance[1]*100:.2f}% variance")
    print(f"[PCA] Total variance retained by 2D plot: {cumulative_variance*100:.2f}%")
    print(f"Goal: Try to keep this value above 50% for good visualization fidelity.")
    print("-" * 40)

    # Scatter Plot
    plt.figure(figsize=(8,6))
    plt.scatter(emb_pca_2d[:, 0], emb_pca_2d[:, 1], s=8)
    
    # Add cumulative variance to the title for quick assessment
    plt.title(f"PCA Projection (Total Variance Retained: {cumulative_variance*100:.2f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "pca_plot_2d.png")
    plt.savefig(save_path, dpi=200)
    print(f"[PCA] 2D Scatter Plot saved → {save_path}")

    # Scree Plot (for general embedding quality)
    # Re-run PCA with all components to find the optimal dimension (k)
    pca_full = PCA(random_state=42)
    pca_full.fit(embeddings)
    
    # Calculate Cumulative Explained Variance
    cumulative_variance_full = np.cumsum(pca_full.explained_variance_ratio_)
    
    plt.figure(figsize=(9, 6))
    plt.plot(range(1, len(cumulative_variance_full) + 1), cumulative_variance_full, marker='o', linestyle='--')
    
    # Draw lines for 80% and 90% retained variance
    plt.axhline(y=0.80, color='r', linestyle='-')
    plt.text(1, 0.80, '80% Retention', color='r', fontsize=10, va='bottom')
    
    plt.title("Scree Plot: Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components ($k$)")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.grid(True)
    plt.xticks(np.arange(1, min(dim, 20) + 1, max(1, min(dim, 20) // 10))) # Show up to 20 components
    plt.tight_layout()
    
    save_path_scree = os.path.join(output_dir, "pca_scree_plot.png")
    plt.savefig(save_path_scree, dpi=200)
    print(f"[PCA] Scree Plot saved → {save_path_scree}")

    return emb_pca_2d


def plot_tsne_embeddings(embedding_file, output_dir="embedding_tsne_plots", perplexity=5):
    """
    Load embeddings and visualize them using t-SNE (2D) via openTSNE.
    
    embedding_file format:
    index,name,dim_0,dim_1,...dim_n
    """

    # Load embeddings
    df = pd.read_csv(embedding_file)
    names = df['entity_idx'].astype(str).to_list()
    embedding_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings = df[embedding_cols].to_numpy(dtype=float)


    n_entities, dim = embeddings.shape
    print(f"[t-SNE] Loaded {n_entities} embeddings with dimension {dim}")

    os.makedirs(output_dir, exist_ok=True)

    # Adjust perplexity if too large
    perplexity = min(perplexity, n_entities - 1)
    if n_entities < 50:
        perplexity = max(5, n_entities // 2)

    print("cp1: initializing t-SNE")

    # Use openTSNE (fast for large datasets)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        initialization="pca",
        n_jobs=-1,         
        random_state=42
    )
    print("cp2: fitting t-SNE")

    # Fit and transform embeddings
    emb_tsne = tsne.fit(embeddings)

    print(f"[t-SNE] Completed t-SNE with perplexity={perplexity}")

    # Plot
    plt.figure(figsize=(8,6))
    plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], s=8)
    plt.title(f"t-SNE Projection (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, "tsne_plot.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[t-SNE] Plot saved → {save_path}")
    return emb_tsne


def save_metrics(results, save_dir, filename="evaluation_results.txt"):
    """Extract metrics from PyKEEN results and save to a text file."""
    metrics = {
        "Mean Rank (MR)": results.get_metric('mean_rank'),
        "Mean Reciprocal Rank (MRR)": results.get_metric('mrr'),
        "Hits@10": results.get_metric('hits@10'),
        "Hits@3": results.get_metric('hits@3'),
        "Hits@1": results.get_metric('hits@1'),
    }

    # Display result
    print("\nEvaluation Results on Test Set:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save to file
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "w") as f:
        f.write("Evaluation Results on Test Set:\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"\nMetrics saved to: {filepath}")


# ---------- MAIN SCRIPT TO RUN BOTH VISUALIZATIONS ----------

EMBEDDING_FILE = "TransE_entity_embeddings_256.csv"   
OUTPUT_PCA_DIR = "pca_output"
OUTPUT_TSNE_DIR = "tsne_output"

if __name__ == "__main__":

    print("===== Running PCA Visualization =====")
    pca_result = plot_pca_embeddings(
        embedding_file=EMBEDDING_FILE,
        output_dir=OUTPUT_PCA_DIR
    )
    print("PCA Finished\n")

    # print("===== Running t-SNE Visualization =====")
    # tsne_result = plot_tsne_embeddings(
    #     embedding_file=EMBEDDING_FILE,
    #     output_dir=OUTPUT_TSNE_DIR,
    #     perplexity=5    # choose any value 5-50; tune this
    # )
    # print("t-SNE Finished\n")

    print("All visualizations complete!")

