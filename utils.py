from PIL import Image
import torch
import time
import os
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
import pandas as pd
import plotly.express as px
from tqdm import tqdm

# Function to get list of image paths
def get_image_paths_and_labels(root_dir, num_images=None):
    image_paths = []
    image_labels = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(dirpath, filename)
                label = os.path.basename(dirpath)  # Use the directory name as the label
                image_paths.append(image_path)
                image_labels.append(label)
                # Limit number of images to get
                if num_images is not None and len(image_paths) >= num_images:
                      return image_paths, image_labels
    return image_paths, image_labels

# Function to get embeddings in numpy array
def get_image_embeddings(image_paths, processor, model, device='cpu', batch_size=100):
    embeddings_list = []
    start_time = time.time()
    model.to(device)  # Move the model to the GPU

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = [Image.open(image_path).convert('RGB') for image_path in batch_paths]
        inputs = processor(images=batch_images, return_tensors="pt").to(device)  # Move inputs to the GPU
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state
        flatten_embeddings = embeddings.flatten(start_dim=1).cpu().numpy()  # Move embeddings back to CPU and convert to numpy
        embeddings_list.extend(flatten_embeddings)

    embeddings_array = np.array(embeddings_list)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Time taken to generate embeddings for {embeddings_array.shape[0]} images: {total_time} seconds")

    return embeddings_array, total_time

def reduce_embedding_dims(embeddings_array, n_components=3, dim_reduction_algorithm='PCA', batch_size=32):
    if dim_reduction_algorithm == 'PCA':
        pca = PCA(n_components=n_components)
        start_time = time.time()
        reduced_embeddings = pca.fit_transform(embeddings_array)
        end_time = time.time()
        time_to_reduce_embeddings = end_time - start_time
        
    elif dim_reduction_algorithm == 'IPCA':
        ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        start_time = time.time()

        # Fit IncrementalPCA in batches
        for i in tqdm(range(0, embeddings_array.shape[0], batch_size)):
            end_index = i + batch_size
            batch_data = embeddings_array[i:end_index]
            ipca.partial_fit(batch_data)

            # Transform the data in batches
            reduced_embeddings = []
            for i in range(0, embeddings_array.shape[0], batch_size):
                end_index = i + batch_size
                batch_data = embeddings_array[i:end_index]
                reduced_batch = ipca.transform(batch_data)
                reduced_embeddings.append(reduced_batch)

        reduced_embeddings = np.vstack(reduced_embeddings)
        end_time = time.time()
        time_to_reduce_embeddings = end_time - start_time

    else:
        raise ValueError("Invalid dimensionality reduction algorithm. Choose 'PCA' | 'TSNE' | 'UMAP'.")

    print(f"Time taken to reduce embeddings with {dim_reduction_algorithm}: {end_time - start_time}")
    return reduced_embeddings, time_to_reduce_embeddings

def plot_embeddings(reduced_embeddings, title='', labels=[]):
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    if labels:
        df['labels'] = labels
        fig = px.scatter_3d(df, x='x', y='y', z='z', title=f"{title}", color='labels')
    else:
        fig = px.scatter_3d(df, x='x', y='y', z='z', title=f"{title}")
    fig.show()


## miscellanous functions
def make_embedding_image_df(reduced_embeddings, image_paths,  image_labels, save_to_csv=True):
    df = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    df['image_path'] = image_paths
    df['label'] = image_labels
    if save_to_csv:
        df.to_csv('embedding_image_df.csv', index=False)
    return df