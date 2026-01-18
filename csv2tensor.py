import os
import pandas as pd
import torch

# --- The directory where your embedding file is and where tensor will be saved ---
INPUT_DIR = "TransH_dim=128_epochs=30"

# Embedding CSV file
EMBEDDING_FILE = os.path.join(INPUT_DIR, "node_embeddings_dim=128.csv")

# Output tensor file
OUTPUT_TENSOR_FILE = os.path.join(INPUT_DIR, "node_embeddings_dim=128.pt")

# Load CSV
df = pd.read_csv(EMBEDDING_FILE)

# If the first column is node IDs, drop it
# Uncomment the next line if needed
# df = df.iloc[:, 1:]

# Convert to PyTorch tensor
embeddings_tensor = torch.tensor(df.values, dtype=torch.float32)

# Print tensor dimensions
print("Tensor shape:", embeddings_tensor.shape)
print("Number of nodes:", embeddings_tensor.shape[0])
print("Embedding dimension:", embeddings_tensor.shape[1])

# Save tensor
torch.save(embeddings_tensor, OUTPUT_TENSOR_FILE)

print(f"Tensor saved to: {OUTPUT_TENSOR_FILE}")
