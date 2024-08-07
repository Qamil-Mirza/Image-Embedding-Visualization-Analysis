from transformers import pipeline, ViTImageProcessor, ViTModel
import torch

# Device Setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DATASET_ROOT_DIR = 'coffee_beans'
DATASET_ROOT_DIR = 'diamond_images'

# Number of images 
N_IMAGES = 100

# Google ViT
PROCESSOR = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
MODEL = ViTModel.from_pretrained('google/vit-base-patch16-224')
MODEL_NAME = 'Google VIT'

# Facebook Dino
# PROCESSOR = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
# MODEL = AutoModel.from_pretrained('facebook/dinov2-base')
# MODEL_NAME

# Clustering Algorithms
CLUSTERING_ALGORITHM = 'PCA'

# Number of components for dimensionality reduction
N_COMPONENTS = 3

# Summary
def run_settings_summary():
    print("Using the following configurations:")
    print(f"Dataset: {DATASET_ROOT_DIR}")
    print(f"Embedding Model: {MODEL_NAME}")
    print(f"Clustering Algorithm: {CLUSTERING_ALGORITHM}")