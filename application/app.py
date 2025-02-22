import os
import time
import streamlit as st
import torch
import faiss
import re
from PIL import Image
from rapidfuzz import fuzz, process
from utils import get_text_embedding, load_model
from utils import load_from_pickle, compute_rrf

# Ensure proper handling of parallel processing issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["QAI_HUB_AUTO_CLONE"] = "1" 
# Set the device (use GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
assets_dir = os.path.join(script_dir, "..", "assets")  # Path to assets folder

# Streamlit UI for model selection
# col1, col2 = st.columns(1)
# with col1:
#     model_choice = st.radio("Select Model", ["CLIP", "JINA"])
model_choice = "CLIP"

# Load the selected model and processor using utility functions
processor, model = load_model(device, model_choice)

# Load precomputed data (image paths, FAISS index, and reference embeddings)
image_paths = load_from_pickle(os.path.join(assets_dir, 'img_paths.pkl'))
image_faiss_index = faiss.read_index(os.path.join(assets_dir, f'{model_choice}_faiss_index.index'))
reference_embeddings = load_from_pickle(os.path.join(assets_dir, 'ref_emb.pkl'))

# Load face recognition data (FAISS index and indices for faces in images)
face_faiss_index = faiss.read_index(os.path.join(assets_dir, 'face_faiss_index.index'))
face_indices = load_from_pickle(os.path.join(assets_dir, 'face_ind.pkl'))

# Text input for query

# with col_2:
#     query = st.text_input("Enter your query:")
query = st.text_input("Enter your query:") 

# Sidebar settings for adjustable thresholds
with st.sidebar:
    st.header("Parameters")
    clip_threshold = st.number_input("Image-Text Similarity Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    fr_threshold = st.number_input("Face Recognition Similarity Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.01)
    rrf_k = st.number_input("RRF k Value", min_value=1, max_value=100, value=60, step=10)

# If user enters a query, proceed with retrieval
if query:
    start_time = time.time()  # Track execution time

    # Compute the text embedding for the query using the loaded model and processor
    text_embedding = get_text_embedding(query, processor, model, device, model_choice)
    if model_choice == "JINA":
        text_embedding = text_embedding[:256]  # Ensure compatibility with JINA model
    
    # Perform image-text similarity search using FAISS index
    scores, indices = image_faiss_index.search(text_embedding.numpy().astype('float32'), len(image_paths))
    
    # Filter results based on image-text similarity threshold
    filtered_ranks = [(image_paths[i], score) for i, score in zip(indices[0], scores[0]) if score > clip_threshold]
    filtered_ranks.sort(key=lambda x: x[1], reverse=True)
    
    # Store ranking for RRF computation (Image-Text)
    clip_rank = {image_path: {"rank": rank + 1, "score": score} for rank, (image_path, score) in enumerate(filtered_ranks)}
    clip_paths = set(clip_rank.keys())
    
    # Perform face recognition search if the query matches known references
    matched_keywords = [keyword for keyword in reference_embeddings if keyword.lower() in query.lower()]
    
    # Flag indicating if face recognition is required (based on matched keywords)
    is_fr = bool(matched_keywords)
    fr_rank = {}

    if is_fr:
        # Initialize a dictionary to hold image scores for face recognition
        image_scores = {}
        
        for keyword in matched_keywords:
            ref_emb = reference_embeddings[keyword]
            ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
            
            # Perform face recognition search using the face FAISS index
            scores, indices = face_faiss_index.search(ref_emb.numpy(), face_faiss_index.ntotal)
            
            # Collect images with matching faces and store their scores
            for idx, score in zip(indices[0], scores[0]):
                if score > fr_threshold:
                    img_idx, face_idx = face_indices[idx]
                    image_path = image_paths[img_idx]
                    
                    if image_path not in image_scores:
                        image_scores[image_path] = {"matching_keywords": set(), "score": []}
                    
                    image_scores[image_path]["matching_keywords"].add(keyword)
                    image_scores[image_path]["score"].append(score)
        
        # Rank images based on the number of matching keywords and similarity scores
        ranked_images = [(image_path, len(data["matching_keywords"]), data["score"]) for image_path, data in image_scores.items() if len(data["matching_keywords"]) == len(matched_keywords)]
        ranked_images.sort(key=lambda x: (x[1], sum(x[2])), reverse=True)
        
        # Store ranking for RRF computation (Face Recognition)
        for rank, (image_path, matching_count, score) in enumerate(ranked_images, 1):
            fr_rank[image_path] = {"rank": rank, "score": score}
        
        fr_paths = set(fr_rank.keys())
    else:
        # If no face recognition, use only image-text paths
        fr_paths = set(clip_paths)

    # Compute RRF scores to combine image-text and face recognition rankings
    rrf_scores = {path: compute_rrf(clip_rank.get(path, {"rank": 1000})["rank"], fr_rank.get(path, {"rank": 1000})["rank"], rrf_k) for path in fr_paths.union(clip_paths)}
    
    # Sort results based on RRF scores
    top_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    end_time = time.time()  # Track execution time end
    execution_time = end_time - start_time  # Calculate execution time
    minutes, seconds = divmod(execution_time, 60)
    
    # Display results in Streamlit
    st.divider()
    st.markdown("### Image Search Results")
    cols_per_row = 3  # Number of images per row
    col_idx = 0
    cols = st.columns(cols_per_row)
    
    # Display top 10 results
    for image_path, _ in top_results[:10]:
        img = Image.open(image_path)
        with cols[col_idx]:
            st.image(img, use_column_width=True)
        col_idx = (col_idx + 1) % cols_per_row
    
    # Display execution time in sidebar
    with st.sidebar:
        st.divider()
        st.header("Execution Time")
        st.subheader(f"{seconds:.2f}s")
