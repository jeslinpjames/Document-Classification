# --- SILENCE TRANSFORMERS WARNINGS ---
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pickle
import json
import torch
import base64
from PIL import Image
from groq import Groq
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

MEMORY_FILE = "text_memory.pkl"
THRESHOLD = 0.50

# Ensure the key exists
if not GROQ_API_KEY:
    st.error("API Key missing! Please add GROQ_API_KEY to your .env file.")

# --- 2. INITIALIZE MODELS ---
@st.cache_resource
def load_models():
    client = Groq(api_key=GROQ_API_KEY)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return client, embed_model

client, embed_model = load_models()

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'rb') as f:
            return pickle.load(f)
    return []

def save_memory(memory_data):
    with open(MEMORY_FILE, 'wb') as f:
        pickle.dump(memory_data, f)

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# --- 3. UI LAYOUT ---
st.set_page_config(page_title="Aurelius IDP", page_icon="📄", layout="wide")
st.title("📄 Aurelius Document Processing Engine")

tab1, tab2 = st.tabs(["🧠 Brain Manager (View & Rename)", "🚀 Live Pipeline"])

# ==========================================
# TAB 1: BRAIN MANAGER
# ==========================================
with tab1:
    st.header("Manage Document Memory")
    st.write("Review the archetypes the AI discovered, view their source images, and give them human-readable names.")
    
    memory = load_memory()
    
    if not memory:
        st.warning("Memory is empty! Run your cluster builder script first.")
    else:
        for i, item in enumerate(memory):
            with st.expander(f"📁 Class: {item['label']}", expanded=False):
                
                # --- IMAGE GALLERY ---
                images = item.get("images", [])
                if images:
                    st.write(f"**Source Images in this Class ({len(images)} total):**")
                    st.caption("💡 Click the arrows icon in the top right of any image to view it full screen and zoom in.")
                    
                    cols = st.columns(min(3, len(images)))
                    for col_idx, img_path in enumerate(images[:3]): 
                        with cols[col_idx]:
                            try:
                                st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
                            except Exception:
                                st.error("Image missing")
                
                st.write("---")
                st.write("**Sample Extracted Text for this Class:**")
                st.info(item.get('raw_text', 'No text available')[:500] + "...")
                
                # --- RENAME FORM ---
                with st.form(key=f"rename_form_{i}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        new_name = st.text_input("Rename this class to:", value=item['label'])
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        submit = st.form_submit_button("💾 Save Name")
                        
                    if submit and new_name != item['label']:
                        memory[i]['label'] = new_name
                        save_memory(memory)
                        st.success(f"Renamed to {new_name}!")
                        st.rerun()

# ==========================================
# TAB 2: LIVE PIPELINE
# ==========================================
with tab2:
    st.header("Drag & Drop Classification")
    
    uploaded_file = st.file_uploader("Upload a scanned document (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
            
        with col2:
            if st.button("🚀 Run Extraction & Classification", type="primary", use_container_width=True):
                
                # --- PHASE 1: EXTRACTION ---
                with st.spinner("Agent 2 (Groq) is extracting data..."):
                    try:
                        base64_image = encode_image(uploaded_file)
                        prompt = """
                        Extract all key-value pairs from this form.
                        Map printed questions to answers. Return ONLY a valid JSON object.
                        """
                        
                        chat_completion = client.chat.completions.create(
                            messages=[
                                {"role": "user", "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                                ]}
                            ],
                            model="meta-llama/llama-4-scout-17b-16e-instruct",
                            temperature=0.1,
                        )
                        
                        resp_text = chat_completion.choices[0].message.content.strip()
                        if resp_text.startswith("```json"): resp_text = resp_text[7:-3].strip()
                        elif resp_text.startswith("```"): resp_text = resp_text[3:-3].strip()
                        extracted_data = json.loads(resp_text)
                        
                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                        st.stop()

                # --- PHASE 2: VECTORIZATION & CLASSIFICATION ---
                with st.spinner("Agent 1 (MiniLM) is analyzing vectors..."):
                    flat_string = " | ".join([f"{k}: {v}" for k, v in extracted_data.items()])
                    new_vector = embed_model.encode(flat_string, convert_to_tensor=True).cpu()
                    
                    best_score = -1.0
                    best_label = "Unknown"
                    winning_vector = None
                    memory = load_memory()
                    
                    for item in memory:
                        score = util.cos_sim(new_vector, item["vector"]).item()
                        if score > best_score:
                            best_score = score
                            best_label = item["label"]
                            winning_vector = item["vector"]

                st.markdown("---")
                
                # --- PHASE 3: RESULT & ATTRIBUTION RENDERING ---
                if best_score >= THRESHOLD:
                    st.success(f"### ✅ Classified as: **{best_label}**")
                    st.progress(best_score, text=f"Confidence Score: {best_score:.2f}")
                    
                    # Calculate Semantic Attribution
                    attribution_scores = {}
                    for k, v in extracted_data.items():
                        field_text = f"{k}: {v}"
                        field_vector = embed_model.encode(field_text, convert_to_tensor=True).cpu()
                        field_score = util.cos_sim(field_vector, winning_vector).item()
                        attribution_scores[k] = field_score
                        
                    # Render the Heatmap
                    st.markdown("### 🔍 Semantic Feature Attribution")
                    st.caption("Fields highlighted in green heavily influenced the AI's classification choice.")
                    
                    html_content = "<div style='font-family: monospace; font-size: 15px; line-height: 1.6; border: 1px solid #333; padding: 10px; border-radius: 8px;'>"
                    for k, v in extracted_data.items():
                        score = attribution_scores.get(k, 0)
                        
                        # Set color intensity based on score (Using RGBA for dark/light mode compatibility)
                        if score > 0.5: bg_color = "rgba(16, 185, 129, 0.4)"     # Strong green
                        elif score > 0.3: bg_color = "rgba(16, 185, 129, 0.2)"   # Light green
                        elif score > 0.15: bg_color = "rgba(16, 185, 129, 0.05)" # Very faint green
                        else: bg_color = "transparent"
                        
                        tooltip = f"Impact Score: {score:.2f}"
                        
                        # FIX: Concatenated as single lines to prevent Markdown from turning it into a code block
                        html_content += f"<div style='background-color: {bg_color}; padding: 6px 10px; border-radius: 4px; margin-bottom: 6px;' title='{tooltip}'>"
                        html_content += f"<strong>{k}</strong>: {v} "
                        html_content += f"<span style='opacity: 0.6; font-size: 12px; float: right;'>{score:.2f}</span>"
                        html_content += "</div>"
                        
                    html_content += "</div>"
                    st.markdown(html_content, unsafe_allow_html=True)
                    
                    # Optional: Expandable raw JSON
                    with st.expander("View Raw JSON Output"):
                        st.json(extracted_data)

                else:
                    st.warning(f"### ⚠️ Unknown Document")
                    st.write(f"Top match was **{best_label}** (Score: {best_score:.2f}, below {THRESHOLD} threshold)")
                    
                    new_class = st.text_input("Teach the AI: What is this document called?")
                    if st.button("Learn Document Type"):
                        memory.append({
                            "label": new_class,
                            "vector": new_vector,
                            "raw_text": flat_string,
                            "images": []
                        })
                        save_memory(memory)
                        st.success(f"Brain updated with {new_class}!")
                
                    st.markdown("### 📊 Extracted Data")
                    st.json(extracted_data)