import os
import json
import glob
import google.generativeai as genai
from PIL import Image

# 1. Setup API Key
# Replace "YOUR_API_KEY" with your actual key, or set it as an environment variable
API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDyLro9edYPP1PDWsIFJJnO-3A65ztb1uI") 
genai.configure(api_key=API_KEY)

# 2. Paths based on your directory structure
IMAGE_DIR = r"dataset\dataset\testing_data\images"

def extract_form_data(image_path):
    print(f"📄 Loading image: {os.path.basename(image_path)}")
    img = Image.open(image_path)

    # 3. Initialize the Model
    # We use gemini-1.5-flash because it is fast and excellent at multimodal tasks
    model = genai.GenerativeModel('gemini-2.5-flash')

    # 4. The Extraction Prompt
    prompt = """
    You are an expert document data extractor. 
    Look at this form and extract all the key-value pairs.
    Map the printed questions/headers (keys) to the handwritten or printed answers (values).
    Return ONLY a valid JSON object. Do not include any markdown formatting like ```json.
    """

    print("🧠 Sending to Gemini 2.5 Flash for extraction... (This takes a few seconds)")
    
    try:
        response = model.generate_content([prompt, img])
        
        # 5. Clean and parse the response
        response_text = response.text.strip()
        # Sometimes models wrap JSON in markdown blocks even when told not to, so we clean it
        if response_text.startswith("```json"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```"):
             response_text = response_text[3:-3].strip()
             
        extracted_json = json.loads(response_text)
        
        print("\n✅ Extraction Successful!")
        print(json.dumps(extracted_json, indent=4))
        
        return extracted_json

    except Exception as e:
        print(f"\n❌ Extraction Failed: {e}")
        # Print raw response text if JSON parsing fails to debug
        if 'response' in locals():
            print("Raw output was:", response.text)
        return None

if __name__ == "__main__":
    # Find all PNG images in the testing directory
    test_images = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    if not test_images:
        print(f"Could not find any PNG images in {IMAGE_DIR}")
    else:
        # Test it on the very first image
        test_image_path = test_images[0]
        extract_form_data(test_image_path)