import pandas as pd
from pykeen.triples import TriplesFactory
from pykeen.models import TransE, TransH
from pykeen.training import SLCWATrainingLoop
from pykeen.evaluation import RankBasedEvaluator
from pykeen.stoppers import EarlyStopper
from pykeen.pipeline import pipeline
from torch.optim import Adam
from emb_analysis import plot_pca_embeddings, plot_tsne_embeddings, save_metrics
# from visualize import plot_pca_embeddings_colored, plot_tsne_embeddings_colored

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
TRAINING_EPOCHS = 3
LEARNING_RATE = 0.001
METHOD = 'TransH'
OUTPUT_DIR = METHOD + '_dim=' + str(EMBEDDING_DIM) + '_epochs=' + str(TRAINING_EPOCHS) 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 256 if DEVICE == 'cuda' else 8192
EVAL_BATCH_SIZE = 1024
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
random.shuffle(triples)



print("Preparing data for PyKEEN...")
print("Total number of triples:", len(triples))
tf = TriplesFactory.from_labeled_triples(triples)
print("len of tf triples:", len(tf.mapped_triples))

training, validation, testing = tf.split([0.90, 0.00, 0.1])
training = tf

total_rows = triples.shape[0]
unique_rows = np.unique(triples, axis=0).shape[0]
duplicate_count = total_rows - unique_rows

print(f"Total Triples: {total_rows}")
print(f"Unique Triples: {unique_rows}")
print(f"Duplicate Triples Found: {duplicate_count}")
import pandas as pd

# Convert back to a DataFrame for easier analysis
check_df = pd.DataFrame(triples, columns=['head', 'rel', 'tail'])

# Find all rows that are duplicates (excluding the first occurrence)
duplicate_mask = check_df.duplicated(keep='first')
duplicate_rows = check_df[duplicate_mask]

print(f"Found {len(duplicate_rows)} duplicate rows.")
print(duplicate_rows.head(10)) # Show the first 10 duplicates