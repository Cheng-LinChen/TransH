import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Using sklearn's TSNE as a widely available alternative to openTSNE
from sklearn.manifold import TSNE 
import os

# --- The directory where your embedding file is and where plots will be saved ---
INPUT_DIR = "TransH_dim=64_epochs=3" 
# Assuming your embedding file is named 'embedding.csv' inside INPUT_DIR
EMBEDDING_FILE = os.path.join(INPUT_DIR, "node_embeddings_dim=64.csv")
TYPE_FILE = "unique_entities.csv" # The file containing entity types

def plot_pca_embeddings_colored(embedding_file, output_dir=INPUT_DIR):
    """
    Load metadata-enriched embeddings and visualize using PCA.
    """
    # 1. Load the single combined CSV
    df = pd.read_csv(embedding_file)
    
    # 2. Extract types
    if 'type' not in df.columns:
        raise KeyError(f"Column 'type' not found in {embedding_file}. Check your saving logic.")
    types = df['type'].fillna('Unknown').astype(str)
    
    # 3. Extract only the embedding dimensions
    # We select all columns that start with 'dim_' to be safe
    dim_cols = [c for c in df.columns if c.startswith('dim_')]
    
    # Fallback: If for some reason columns aren't named 'dim_', 
    # use your requirement: start from index 3 to the end
    if not dim_cols:
        embeddings = df.iloc[:, 3:].values
    else:
        embeddings = df[dim_cols].values

    n_entities, dim = embeddings.shape
    print(f"\n[PCA] Loaded {n_entities} entities with {dim} dimensions.")

    os.makedirs(output_dir, exist_ok=True)

    # 4. PCA Calculation (2D)
    pca_2d = PCA(n_components=2, random_state=42)
    emb_pca_2d = pca_2d.fit_transform(embeddings)
    variance_retained = pca_2d.explained_variance_ratio_.sum()

    # 5. Scatter Plot
    plt.figure(figsize=(10, 8))
    unique_types = types.unique()
    
    for entity_type in unique_types:
        mask = (types == entity_type)
        plt.scatter(
            emb_pca_2d[mask, 0], 
            emb_pca_2d[mask, 1], 
            s=20, 
            label=entity_type,
            alpha=0.7 # Add transparency to see overlapping points
        )
    
    plt.title(f"PCA Projection (Variance: {variance_retained*100:.2f}%)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Place legend outside if you have many types
    plt.legend(title="Entity Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "pca_plot_2d_colored.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    # 6. Scree Plot
    pca_full = PCA(random_state=42).fit(embeddings)
    cumulative_variance_full = np.cumsum(pca_full.explained_variance_ratio_)
    
    plt.figure(figsize=(9, 6))
    plt.plot(range(1, len(cumulative_variance_full) + 1), cumulative_variance_full, marker='o')
    plt.axhline(y=0.80, color='r', linestyle='--')
    plt.title("Scree Plot: Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Ratio")
    plt.grid(True)
    
    save_path_scree = os.path.join(output_dir, "pca_scree_plot.png")
    plt.savefig(save_path_scree, dpi=200)
    plt.close()
    
    return save_path, save_path_scree

def plot_tsne_embeddings_colored(embedding_file, output_dir=INPUT_DIR, perplexity=30):
    """
    Load metadata-enriched embeddings and visualize using t-SNE.
    """
    # 1. Load the single combined CSV
    df = pd.read_csv(embedding_file)
    
    # 2. Extract types and handle missing data
    if 'type' not in df.columns:
        print("Warning: 'type' column not found. Defaulting to 'Unknown'.")
        df['type'] = 'Unknown'
    types = df['type'].fillna('Unknown').astype(str)
    
    # 3. Extract only the embedding dimensions
    # Look for columns starting with 'dim_'; fallback to index 3+
    dim_cols = [c for c in df.columns if c.startswith('dim_')]
    if not dim_cols:
        embeddings = df.iloc[:, 3:].values
    else:
        embeddings = df[dim_cols].values

    n_entities, dim = embeddings.shape
    print(f"\n[t-SNE] Loaded {n_entities} entities with {dim} dimensions.")

    os.makedirs(output_dir, exist_ok=True)

    # 4. Adjust perplexity for dataset size
    # For 80k+ entities, a perplexity of 30-50 is much better than 5
    perplexity = min(perplexity, n_entities - 1)
    if n_entities < 50:
        perplexity = max(5, n_entities // 2)

    # 5. Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",             
        learning_rate="auto",   
        n_jobs=-1, # Use all available CPU cores             
        random_state=42
    )
    emb_tsne = tsne.fit_transform(embeddings)

    print(f"[t-SNE] Completed t-SNE with perplexity={perplexity}.")

    # 6. Plotting
    plt.figure(figsize=(12, 9))
    unique_types = types.unique()
    
    for entity_type in unique_types:
        mask = (types == entity_type)
        plt.scatter(
            emb_tsne[mask, 0], 
            emb_tsne[mask, 1], 
            s=15, # Smaller dots for dense t-SNE plots
            label=entity_type,
            alpha=0.6
        )
    
    plt.title(f"t-SNE Projection (Perplexity={perplexity}) Colored by Type")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True, linestyle=':', alpha=0.5)
    
    # Move legend outside the plot area
    plt.legend(title="Entity Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(output_dir, "tsne_plot_colored.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[t-SNE] Colored Plot saved → {save_path}")
    return save_path

if __name__ == "__main__":
    # Example Execution (using the dummy files generated)
    pca_plot_path, pca_scree_path = plot_pca_embeddings_colored(EMBEDDING_FILE, TYPE_FILE)
    tsne_plot_path = plot_tsne_embeddings_colored(EMBEDDING_FILE, TYPE_FILE, perplexity=5)