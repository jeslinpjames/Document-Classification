import os
import json
import pickle
import glob
import torch
import base64
from PIL import Image
from io import BytesIO
from groq import Groq
from sentence_transformers import SentenceTransformer, util

# --- 1. CONFIGURATION ---
# Paste your Groq API Key here
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
client = Groq(api_key=GROQ_API_KEY)

IMAGE_DIR = r"dataset\dataset\testing_data\images"
MEMORY_FILE = "text_memory.pkl"
THRESHOLD = 0.50 

print("🧠 Waking up Agents...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. LOAD THE BRAIN ---
if not os.path.exists(MEMORY_FILE):
    print("❌ Fatal Error: Brain not found. Run build_cluster_brain.py first.")
    exit()

with open(MEMORY_FILE, 'rb') as f:
    memory = pickle.load(f)
print(f"📚 Loaded Aurelius Memory with {len(memory)} document archetypes.")

# Helper function to convert Image to Base64 for Groq
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_document(image_path):
    global memory
    filename = os.path.basename(image_path)
    print("\n" + "="*60)
    print(f"📄 INCOMING DOCUMENT: {filename}")
    
    # --- PHASE 1: EXTRACTION (GROQ / LLAMA 3.2 VISION) ---
    print("  [Agent 2] Extracting layout and handwriting using Llama 3.2...")
    base64_image = encode_image(image_path)
    
    prompt = """
    Extract all the key-value pairs from this form.
    Map the printed questions/headers (keys) to the handwritten or printed answers (values).
    Return ONLY a valid JSON object. No markdown, no intro text.
    """
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            temperature=0.1,
        )
        
        resp_text = chat_completion.choices[0].message.content.strip()
        
        # Clean markdown if Llama decides to add it
        if resp_text.startswith("```json"): resp_text = resp_text[7:-3].strip()
        elif resp_text.startswith("```"): resp_text = resp_text[3:-3].strip()
        extracted_data = json.loads(resp_text)
        
    except Exception as e:
        print(f"  [❌] Extraction failed: {e}")
        return

    # --- PHASE 2: VECTORIZATION ---
    print("  [Agent 1] Flattening data and routing...")
    flat_string = " | ".join([f"{k}: {v}" for k, v in extracted_data.items()])
    new_vector = embed_model.encode(flat_string, convert_to_tensor=True).cpu()
    
    # --- PHASE 3: CLASSIFICATION ---
    best_score = -1.0
    best_label = "Unknown"
    
    for item in memory:
        score = util.cos_sim(new_vector, item["vector"]).item()
        if score > best_score:
            best_score = score
            best_label = item["label"]
            
    # --- PHASE 4: HUMAN IN THE LOOP (HITL) ---
    if best_score >= THRESHOLD:
        print(f"  [✅] AUTO-CLASSIFIED: '{best_label}' (Confidence: {best_score:.2f})")
        print("\n  [📊] Extracted JSON Data:")
        print(json.dumps(extracted_data, indent=2)[:300] + "\n  ... [truncated]")
    else:
        print(f"  [⚠️] LOW CONFIDENCE (Top match was {best_label} at {best_score:.2f})")
        print("  [🙎‍♂️] Triggering Human-in-the-Loop review.")
        print("\nHere is the extracted data:")
        print(json.dumps(extracted_data, indent=2))
        
        img = Image.open(image_path)
        img.show()
        
        new_class = input(f"\n  [?] I don't recognize this form. What class is it? (Type name, or press Enter to skip): ").strip()
        
        if new_class:
            print(f"  [+] Learning new pattern for: {new_class}")
            memory.append({
                "label": new_class,
                "vector": new_vector,
                "raw_text": flat_string
            })
            with open(MEMORY_FILE, 'wb') as f:
                pickle.dump(memory, f)
            print("  [✅] Brain updated. I will remember this forever.")

if __name__ == "__main__":
    import time
    test_images = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not test_images:
        print("No testing images found!")
    else:
        print(f"\n🚀 Running Full Verification on 10 unseen documents...")
        
        for i, img_path in enumerate(test_images[:10]):
            process_document(img_path)
            
            if i < 9:
                print("\n  [⏳] Pausing for 5 seconds to respect Groq rate limits...")
                time.sleep(5)
                
        print("\n" + "="*60)
        print("✅ VERIFICATION COMPLETE.")
        print("If the documents were successfully assigned a 'Document_Type_X', your AI is working!")