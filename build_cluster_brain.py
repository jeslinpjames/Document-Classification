import os
import json
import glob
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
TRAIN_DIR = r"dataset\dataset\training_data\annotations_flat"
IMAGE_DIR = r"dataset\dataset\training_data\images"
MEMORY_FILE = "text_memory.pkl"
NUM_CLUSTERS = 8  

def build_brain():
    print("🧠 Initializing Text Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    json_files = glob.glob(os.path.join(TRAIN_DIR, "*.json"))
    documents_text = []

    print(f"📂 Loading {len(json_files)} formatted documents...")
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            flat_string = " | ".join([f"{k}: {v}" for k, v in data.items()])
            documents_text.append(flat_string)

    print("🔮 Generating embeddings...")
    embeddings = model.encode(documents_text, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy()

    print(f"🧩 Clustering into {NUM_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_np)
    centroids = kmeans.cluster_centers_

    memory = []
    
    for i in range(NUM_CLUSTERS):
        # Find all documents in this cluster
        cluster_indices = [j for j in range(len(labels)) if labels[j] == i]
        
        # Match the JSON files back to their original PNG images
        cluster_images = []
        for idx in cluster_indices:
            base_name = os.path.basename(json_files[idx]).replace(".json", ".png")
            img_path = os.path.join(IMAGE_DIR, base_name)
            if os.path.exists(img_path):
                cluster_images.append(img_path)

        centroid_tensor = torch.tensor(centroids[i])
        
        memory.append({
            "label": f"Document_Type_{i}",
            "vector": centroid_tensor,
            # Save the text of the first document as a real sample
            "raw_text": documents_text[cluster_indices[0]], 
            # Save the list of all images in this class
            "images": cluster_images 
        })

    with open(MEMORY_FILE, 'wb') as f:
        pickle.dump(memory, f)

    print("✅ Brain upgraded with image mappings! Saved to text_memory.pkl")

if __name__ == "__main__":
    build_brain()