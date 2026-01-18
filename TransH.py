import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, TransH
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.pipeline import pipeline
from torch.optim import Adam
from emb_analysis import plot_pca_embeddings, plot_tsne_embeddings, save_metrics
from visualize import plot_pca_embeddings_colored, plot_tsne_embeddings_colored

from pykeen.losses import MarginRankingLoss 
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.regularizers import LpRegularizer # <-- Import LpRegularizer
import pykeen.regularizers as regularizers
import torch
import os
import random
import numpy as np
import gc


random.seed(42)

# --- Configuration ---
FILE_PATH = 'processed_kg.csv'
EMBEDDING_DIM = 16
TRAINING_EPOCHS = 5
LEARNING_RATE = 0.0005
METHOD = 'TransH'
OUTPUT_DIR = METHOD + '_dim=' + str(EMBEDDING_DIM) + '_epochs=' + str(TRAINING_EPOCHS) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256 if DEVICE == 'cuda' else 8192
EVAL_BATCH_SIZE = 8192
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)



# --- Load Data ---
print("Loading data...")
df = pd.read_csv(
    FILE_PATH,
    dtype={
        'relation': str, 
        'x_name': str, 
        'y_name': str  
    },
    low_memory=False
)


df['x_entity'] = df['x_idx'].astype(str)
df['y_entity'] = df['y_idx'].astype(str)

# Combine x and y entity indices
all_entities = pd.concat([df['x_idx'], df['y_idx']], axis=0)

# Drop duplicates to get unique entities
unique_entities = all_entities.drop_duplicates()

# x_idx, relation, y_idx
triples = df[['x_entity', 'relation_idx', 'y_entity']].values

triples = np.array(triples, dtype=str)
# random.shuffle(triples)



print("Preparing data for PyKEEN...")
print("Total number of triples:", len(triples))
tf = TriplesFactory.from_labeled_triples(triples)
print("len of tf triples:", len(tf.mapped_triples))

print(f"Total unique entities in tf: {len(tf.entity_to_id)}")

training, validation, testing = tf.split([0.99, 0.005, 0.005])
training = tf




combined_regularizer = LpRegularizer(p=2, weight=1e-5)

model = TransH(
    triples_factory=training,
    embedding_dim=EMBEDDING_DIM,
    regularizer=combined_regularizer,
).to(DEVICE)

optimizer = Adam(
    params=model.get_grad_params(), 
    lr=LEARNING_RATE
)

negative_sampler = BernoulliNegativeSampler(
    mapped_triples=training.mapped_triples,
    num_negs_per_pos=16, # Commonly used value, try 16-64
)

training_loop = SLCWATrainingLoop(
    model=model,
    optimizer=optimizer,
    triples_factory=training,
    automatic_memory_optimization=False, # PyKEEN's automatic optimization
    negative_sampler=negative_sampler,
)

evaluator = RankBasedEvaluator(filtered=True)

# Early stopping, using validation set
early_stopper = EarlyStopper(
    model=model,
    training_triples_factory=training,
    evaluation_triples_factory=validation,
    evaluator=evaluator,
    metric='hits@10', # The metric to monitor
    patience=5,
    frequency=5, # Check every 5 epochs
    result_tracker=None,
)


# Execute training with early stopping and evaluation
metrics = training_loop.train(
    triples_factory=training,
    num_epochs=TRAINING_EPOCHS,
    batch_size=batch_size,
    use_tqdm_batch=False,
    use_tqdm=True, # Use TQDM for overall epoch progress
    stopper=early_stopper,
)


print("Evaluating model on test set...")
results = evaluator.evaluate(
    model=model,
    mapped_triples=testing.mapped_triples,
    additional_filter_triples=[training.mapped_triples, validation.mapped_triples],
    batch_size=EVAL_BATCH_SIZE,
    use_tqdm=True,
    slice_size=1024,   # avoid loading full tensors
)

print("\nTraining complete!")
save_metrics(results, OUTPUT_DIR, filename="evaluation_results.txt")

print("Matrics saved!")


# === Free Memory After Evaluation ===
del evaluator
gc.collect()

if DEVICE == 'cuda':
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

print("Memory cleaned after evaluation!")


import matplotlib.pyplot as plt

# ---- Plot Loss Curve ----
losses = np.asarray(metrics)

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
output_path = os.path.join(OUTPUT_DIR, "loss_curve.png")
plt.savefig(output_path)

plt.close()  # Close the figure to free memory
print(f"Loss curve saved at: {output_path}")

# ---- Saving Embedding ----
# 1️⃣ Get embeddings as numpy array
entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()
entity_labels = list(tf.entity_id_to_label.values())  # strings of indices

# 2️⃣ Prepare metadata: merge x/y entity info from original df
entity_meta = pd.concat([
    df[['x_idx','x_name','x_type']].rename(columns={'x_idx':'entity_idx', 'x_name':'name', 'x_type':'type'}),
    df[['y_idx','y_name','y_type']].rename(columns={'y_idx':'entity_idx', 'y_name':'name', 'y_type':'type'})
]).drop_duplicates(subset='entity_idx')

# Ensure entity_idx is int for proper numeric merge
entity_meta['entity_idx'] = entity_meta['entity_idx'].astype(int)
emb_df = pd.DataFrame(
    entity_embeddings,
    columns=[f"dim_{i}" for i in range(EMBEDDING_DIM)]
)
emb_df['entity_idx'] = [int(x) for x in entity_labels]  # cast PyKEEN labels to int

# 3️⃣ Merge metadata with embeddings
df_entities = emb_df.merge(entity_meta, on='entity_idx', how='left')

# 4️⃣ Reorder columns
cols = ['entity_idx', 'name', 'type'] + [f'dim_{i}' for i in range(EMBEDDING_DIM)]
df_entities = df_entities[cols]

# 5️⃣ Sort by entity_idx
df_entities = df_entities.sort_values('entity_idx').reset_index(drop=True)

# 6️⃣ Save CSV
csv_path = os.path.join(OUTPUT_DIR, f"node_embeddings_with_metadata_dim={EMBEDDING_DIM}.csv")
df_entities.to_csv(csv_path, index=False)
print(f"[CSV] Saved embeddings with metadata → {csv_path}")

# 7️⃣ Save tensor-only file
tensor_path = os.path.join(OUTPUT_DIR, f"node_embeddings_tensor_dim={EMBEDDING_DIM}.pt")
torch.save(torch.tensor(entity_embeddings), tensor_path)
print(f"[Tensor] Saved embedding tensor → {tensor_path}")






# ----- Free Model to Save RAM before PCA/TSNE -----
del model
gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache()
print("Model memory released!")

# # ----- Do embedding analysis -----
# plot_pca_embeddings(os.path.join(OUTPUT_DIR, f"node_embeddings_dim={EMBEDDING_DIM}.csv"),
#                     output_dir=OUTPUT_DIR)


# plot_tsne_embeddings(os.path.join(OUTPUT_DIR, f"node_embeddings_dim={EMBEDDING_DIM}.csv"),
#                     output_dir=OUTPUT_DIR, perplexity=5)



embedding_with_meta_path = os.path.join(OUTPUT_DIR, f"node_embeddings_with_metadata_dim={EMBEDDING_DIM}.csv")

try:
    # 1. Run PCA (Generates Scatter plot and Scree plot)
    # We pass the same file for both because it contains embeddings, names, and types
    pca_plot, scree_plot = plot_pca_embeddings_colored(
            embedding_file=embedding_with_meta_path, 
            output_dir=OUTPUT_DIR
        )

    # 2. Run t-SNE
    tsne_plot = plot_tsne_embeddings_colored(
            embedding_file=embedding_with_meta_path, 
            output_dir=OUTPUT_DIR, 
            perplexity=5 # Increased from 5 to 30 for better clustering on larger datasets
        )
    
except Exception as e:
    print(f"An error occurred during visualization: {e}")