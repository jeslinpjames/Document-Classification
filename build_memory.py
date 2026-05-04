import os
import pickle
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
import glob

# Paths matching your directory structure
KNOWN_DIR = r"Data\known_memory"
MEMORY_FILE = r"Data\vector_memory.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Loading CLIP Model on {device}...")

# CLIP understands layout and text structure natively
model = SentenceTransformer('clip-ViT-B-32', device=device)

def build_vector_db():
    memory = []
    
    # Grab every PNG in the known_memory subfolders
    image_paths = glob.glob(os.path.join(KNOWN_DIR, "*", "*.png"))
    print(f"🖼️ Found {len(image_paths)} images. Encoding into memory...")

    for i, path in enumerate(image_paths):
        try:
            # The folder name is the label (e.g., 'invoice', 'form')
            label = os.path.basename(os.path.dirname(path))
            
            img = Image.open(path)
            embedding = model.encode(img, convert_to_tensor=True)
            
            memory.append({
                "path": path,
                "label": label,
                "embedding": embedding.cpu()
            })
            
            if (i + 1) % 50 == 0:
                print(f"  [+] Encoded {i + 1}/{len(image_paths)} documents")
                
        except Exception as e:
            print(f"  [!] Failed on {path}: {e}")

    # Save the database
    with open(MEMORY_FILE, 'wb') as f:
        pickle.dump(memory, f)
    
    print(f"\n✅ Memory built successfully! Saved to {MEMORY_FILE}")

if __name__ == "__main__":
    build_vector_db()