import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.models import CompGCN
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from torch.optim import Adam
from emb_analysis import plot_pca_embeddings, plot_tsne_embeddings, save_metrics

from pykeen.losses import MarginRankingLoss 
from pykeen.sampling import BernoulliNegativeSampler
from pykeen.regularizers import LpRegularizer # <-- keep regularizer
import pykeen.regularizers as regularizers
import torch
import os
import random
import numpy as np
import gc

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
random.seed(42)

FILE_PATH = 'kg.csv'
EMBEDDING_DIM = 16
TRAINING_EPOCHS = 10
LEARNING_RATE = 0.001

METHOD = 'CompGCN'
OUTPUT_DIR = f"CompGCN_{METHOD}_dim={EMBEDDING_DIM}_epochs={TRAINING_EPOCHS}"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("Loading data...")
df = pd.read_csv(
    FILE_PATH,
    dtype={'relation': str, 'x_name': str, 'y_name': str},
    low_memory=False
)

print("Creating triples...")

triples = df[['x_name', 'relation', 'y_name']].values

print("Total triples:", len(triples))
triples = np.array(triples, dtype=str)
random.shuffle(triples)

print("Preparing data for PyKEEN...")
tf = TriplesFactory.from_labeled_triples(
    triples,
    create_inverse_triples=True     # <-- REQUIRED for CompGCN
)
training, validation, testing = tf.split([0.99, 0.005, 0.005])

# ---------------------------------------------------------
# MODEL — replaced TransH → CompGCN  (GNN-based)
# ---------------------------------------------------------

combined_regularizer = LpRegularizer(p=2, weight=1e-5)

model = CompGCN(
    triples_factory=training,
    embedding_dim=EMBEDDING_DIM,
).to(DEVICE)

optimizer = Adam(
    params=model.get_grad_params(),
    lr=LEARNING_RATE
)




training_loop = SLCWATrainingLoop(
    model=model,
    optimizer=optimizer,
    triples_factory=training,
    automatic_memory_optimization=True,

)

evaluator = RankBasedEvaluator(filtered=True)

early_stopper = EarlyStopper(
    model=model,
    training_triples_factory=training,
    evaluation_triples_factory=validation,
    evaluator=evaluator,
    metric='hits@10',
    patience=5,
    frequency=5,
)

batch_size = 256 if DEVICE=='cuda' else 16384

metrics = training_loop.train(
    triples_factory=training,
    num_epochs=TRAINING_EPOCHS,
    batch_size=batch_size,
    use_tqdm=True,
    stopper=early_stopper,
)

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------
print("\nEvaluating model...")
results = evaluator.evaluate(
    model=model,
    mapped_triples=testing.mapped_triples,
    additional_filter_triples=[training.mapped_triples, validation.mapped_triples],
    batch_size=256,
    use_tqdm=True,
    slice_size=1024,
)

save_metrics(results, OUTPUT_DIR, filename="evaluation_results.txt")
print("Metrics saved.")

# Free memory
del evaluator; gc.collect()
if DEVICE == 'cuda':
    torch.cuda.empty_cache(); torch.cuda.ipc_collect()

# ---------------------------------------------------------
# Plot loss curve
# ---------------------------------------------------------
import matplotlib.pyplot as plt

losses = np.asarray(metrics)
plt.figure(figsize=(8,5))
plt.plot(losses)
plt.title("Training Loss Curve - CompGCN")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
loss_path = os.path.join(OUTPUT_DIR,"loss_curve.png")
plt.savefig(loss_path)
plt.close()
print(f"Loss curve saved → {loss_path}")

# ---------------------------------------------------------
# Save embeddings
# ---------------------------------------------------------
print("Saving embeddings...")

entity_embeddings = model.entity_representations[0]().detach().cpu().numpy()
entity_names = list(tf.entity_id_to_label.values())

df_entities = pd.DataFrame(entity_embeddings, columns=[f"dim_{i}" for i in range(EMBEDDING_DIM)])
df_entities.insert(0,"name",entity_names)
df_entities.insert(0,"index",range(len(entity_names)))

out_emb = os.path.join(OUTPUT_DIR, f"node_embeddings_dim={EMBEDDING_DIM}.csv")
df_entities.to_csv(out_emb,index=False)

print(f"Entity embeddings stored → {out_emb}")

# ---------------------------------------------------------
# Free model, run PCA + t-SNE
# ---------------------------------------------------------
del model; gc.collect()
if DEVICE=='cuda': torch.cuda.empty_cache()
print("Model cleared from memory.")

plot_pca_embeddings(out_emb, output_dir=OUTPUT_DIR)
plot_tsne_embeddings(out_emb, output_dir=OUTPUT_DIR, perplexity=5)
print("PCA + TSNE plots saved.")
