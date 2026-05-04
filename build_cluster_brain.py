import os
import json
import glob
import pickle
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# --- CONFIGURATION ---
TRAIN_DIR = r"dataset\dataset\training_data\annotations_flat"
MEMORY_FILE = "text_memory.pkl"
NUM_CLUSTERS = 8  # We are asking the AI to find 8 distinct document types

def build_brain():
    print("🧠 Initializing Text Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    json_files = glob.glob(os.path.join(TRAIN_DIR, "*.json"))
    documents_text = []

    print(f"📂 Loading {len(json_files)} formatted documents...")
    for filepath in json_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Flatten dictionary to a single string (e.g., "Key: Value | Key: Value")
            flat_string = " | ".join([f"{k}: {v}" for k, v in data.items()])
            if flat_string.strip():
                documents_text.append(flat_string)

    print("🔮 Generating mathematical vectors (embeddings)...")
    # Convert text to vectors
    embeddings = model.encode(documents_text, convert_to_tensor=True)
    
    # K-Means works better with numpy arrays
    embeddings_np = embeddings.cpu().numpy()

    print(f"🧩 Clustering {len(documents_text)} documents into {NUM_CLUSTERS} groups...")
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_np)
    
    # The 'centroids' are the exact mathematical center of each cluster
    centroids = kmeans.cluster_centers_

    memory = []
    print("\n" + "="*70)
    print("📊 CLUSTER ANALYSIS (What the AI found)")
    print("="*70)

    for i in range(NUM_CLUSTERS):
        # Find all documents that got sorted into this cluster
        cluster_docs = [documents_text[j] for j in range(len(labels)) if labels[j] == i]
        
        print(f"\n🔹 Cluster {i} (Contains {len(cluster_docs)} documents)")
        # Print a snippet of the first document so you can see what type of form it is
        print(f"   Sample: {cluster_docs[0][:120]}...")
        
        # For now, we auto-name them. In production, you'd name them based on the sample!
        class_name = f"Document_Type_{i}"
        
        # Convert the centroid back to a PyTorch tensor for the pipeline
        centroid_tensor = torch.tensor(centroids[i])
        
        memory.append({
            "label": class_name,
            "vector": centroid_tensor,
            "raw_text": f"Centroid representing {len(cluster_docs)} documents"
        })

    # Save the brain!
    with open(MEMORY_FILE, 'wb') as f:
        pickle.dump(memory, f)

    print("\n" + "="*70)
    print("✅ Cluster Brain built successfully!")
    print(f"💾 Saved {NUM_CLUSTERS} document archetypes to {MEMORY_FILE}")

if __name__ == "__main__":
    build_brain()