import os
from datasets import load_dataset
from PIL import Image
import io

# Silence the Windows symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PROJECT_NAME = "Aurelius"
BASE_DIR = f"{PROJECT_NAME}_Data"
KNOWN_DIR = os.path.join(BASE_DIR, "known_memory")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation_docs")

CLASS_NAMES = [
    "advertisement", "budget", "email", "file_folder", "form", 
    "handwritten", "invoice", "letter", "memo", "news_article", 
    "presentation", "questionnaire", "resume", "scientific_publication", 
    "scientific_report", "specification"
]

for name in CLASS_NAMES:
    os.makedirs(os.path.join(KNOWN_DIR, name), exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

print(f"🚀 Initializing {PROJECT_NAME} Resilient Data Engine...")

# Streaming the Parquet version
ds = load_dataset("chainyo/rvl-cdip", split="test", streaming=True)
ds_iter = iter(ds) # Create an explicit iterator

images_per_class = 30 
collected_counts = {i: 0 for i in range(16)}
eval_count = 0
max_eval = 20
errors_skipped = 0

print(f"📥 Pulling {images_per_class} images per class. Skipping corrupted files automatically...")

while True:
    try:
        # Manually fetch the next item to catch errors during decoding
        example = next(ds_iter)
        
        label_idx = example['label']
        img = example['image']
        
        # 1. Save to Known Memory
        if collected_counts[label_idx] < images_per_class:
            class_name = CLASS_NAMES[label_idx]
            save_path = os.path.join(KNOWN_DIR, class_name, f"{class_name}_{collected_counts[label_idx]}.png")
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(save_path)
            collected_counts[label_idx] += 1
            
            if collected_counts[label_idx] % 10 == 0:
                print(f"  [+] {class_name}: {collected_counts[label_idx]}/{images_per_class}")

        # 2. Save to Evaluation
        elif eval_count < max_eval:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(os.path.join(EVAL_DIR, f"eval_{eval_count}.png"))
            eval_count += 1

        # Check if we're finished
        if all(count >= images_per_class for count in collected_counts.values()) and eval_count >= max_eval:
            break
            
    except (StopIteration):
        print("Reached end of dataset stream.")
        break
    except Exception as e:
        # This catches UnidentifiedImageError or other decoding issues
        errors_skipped += 1
        continue # Just skip the bad row and keep going

print(f"\n✅ {PROJECT_NAME} Dataset Prepared.")
print(f"Total images saved: {sum(collected_counts.values()) + eval_count}")
print(f"Corrupted images skipped: {errors_skipped}")