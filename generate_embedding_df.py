from utils import *
from settings import *

# Get image paths and labels
image_paths, labels = get_image_paths_and_labels(DATASET_ROOT_DIR, num_images=N_IMAGES)

# display summary of run settings
run_settings_summary()

# generate embeddings or load them from a file
# embeddings_array = np.load('embeddings.npy')
embeddings_array, total_time = get_image_embeddings(image_paths, PROCESSOR, MODEL, device=device)

# Reduce embedding dimensions
reduced_embeddings_array, time_to_reduce_embeddings = reduce_embedding_dims(embeddings_array, n_components=N_COMPONENTS, dim_reduction_algorithm=CLUSTERING_ALGORITHM)

df = make_embedding_image_df(reduced_embeddings=reduced_embeddings_array, image_paths=image_paths, image_labels=labels, save_to_csv=True)