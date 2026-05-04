import os
import json
import glob

# 1. Define Paths
BASE_DIR = r"dataset\dataset\training_data"
INPUT_DIR = os.path.join(BASE_DIR, "annotations")
OUTPUT_DIR = os.path.join(BASE_DIR, "annotations_flat")

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def flatten_funsd_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Dictionary to quickly look up blocks by their ID
    blocks_by_id = {block['id']: block for block in data['form']}
    
    flat_dict = {}
    
    # Iterate through all blocks looking for "questions" (the keys)
    for block in data['form']:
        if block['label'] == 'question':
            question_text = block['text'].strip()
            
            # Skip empty questions
            if not question_text:
                continue
                
            answer_texts = []
            
            # Look through the links to find the corresponding answers
            for link in block['linking']:
                # A link is a pair of IDs like [1, 14]. Find the ID that belongs to the answer.
                target_id = link[0] if link[1] == block['id'] else link[1]
                target_block = blocks_by_id.get(target_id)
                
                # If the linked block is an "answer", grab its text
                if target_block and target_block['label'] == 'answer':
                    answer_texts.append(target_block['text'].strip())
            
            # Join multiple answer parts together, or default to an empty string if no answer found
            flat_dict[question_text] = " ".join(answer_texts)
            
    return flat_dict

if __name__ == "__main__":
    print("🚀 Starting FUNSD Translation...")
    
    # Find all JSON files in the input directory
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    
    if not json_files:
        print(f"❌ No JSON files found in {INPUT_DIR}")
    else:
        print(f"📂 Found {len(json_files)} annotation files. Flattening...")
        
        success_count = 0
        for filepath in json_files:
            try:
                # 1. Flatten the data
                flat_data = flatten_funsd_json(filepath)
                
                # 2. Save it to the new folder with the exact same filename
                filename = os.path.basename(filepath)
                output_path = os.path.join(OUTPUT_DIR, filename)
                
                with open(output_path, 'w', encoding='utf-8') as out_file:
                    json.dump(flat_data, out_file, indent=4)
                    
                success_count += 1
                
            except Exception as e:
                print(f"  [!] Failed to process {filename}: {e}")
                
        print(f"\n✅ Translation Complete! {success_count}/{len(json_files)} files flattened.")
        print(f"📁 Check your new formatted data in: {OUTPUT_DIR}")