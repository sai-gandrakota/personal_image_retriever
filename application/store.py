import os
import time
import torch
import faiss
from utils import load_images, load_model, get_image_embedding, load_fr_model, get_face_embeddings, create_faiss_index, save_to_pickle

# Set device (GPU if available, otherwise CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define paths relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
assets_folder = os.path.join(script_dir, "..", "assets")  # Store all output files in 'assets' folder
weights_path = os.path.join(assets_folder, "20180402-114759-vggface2.pt")  # Path for model weights
image_set_folder = os.path.join(script_dir, "..", "dataset", "image_collection")  # Images in a dataset folder
reference_folder = os.path.join(script_dir, "..", "dataset", "reference_images")  # Reference images folder

# Load face recognition model
mtcnn, resnet = load_fr_model(device, weights_path)

# Load image paths from the dataset folder
image_paths = load_images(image_set_folder)
save_to_pickle(image_paths, os.path.join(assets_folder, "img_paths.pkl"))  # Save image paths for later use

# Process multiple models
model_choices = ["CLIP", "JINA"]
for model_choice in model_choices:
    print(f"Processing for model: {model_choice}")
    processor, model = load_model(device, model_choice)
    
    # Generate image embeddings and store in FAISS index
    image_embeddings = []
    for i, image_path in enumerate(image_paths):
        embeddings = get_image_embedding(image_path, processor, model, device)
        if model_choice == "JINA":
            embeddings = embeddings[:256]  # Trim embedding size for JINA model compatibility
        image_embeddings.append(embeddings)
        if i % 10 == 0:
            print(f"Processed {i} images for {model_choice}")

    image_embeddings = torch.cat(image_embeddings)
    img_faiss_index = create_faiss_index(image_embeddings)
    img_faiss_file = os.path.join(assets_folder, f"{model_choice}_faiss_index.index")
    faiss.write_index(img_faiss_index, img_faiss_file)

# Process reference images for face recognition
reference_embeddings = {}
for ref_image in os.listdir(reference_folder):
    if ref_image.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(reference_folder, ref_image)
        person_name = os.path.splitext(ref_image)[0]  # Extract person name from filename
        embeddings = get_face_embeddings(img_path, mtcnn, resnet)
        if embeddings is not None:
            reference_embeddings[person_name] = embeddings[0]

save_to_pickle(reference_embeddings, os.path.join(assets_folder, "ref_emb.pkl"))  # Save reference embeddings

# Generate face embeddings and store them
face_embeddings = []
face_indices = []
for i, image_path in enumerate(image_paths):
    embeddings = get_face_embeddings(image_path, mtcnn, resnet)
    if embeddings is not None:
        for face_idx, face_embedding in enumerate(embeddings):
            face_embeddings.append(face_embedding)
            face_indices.append((i, face_idx))
    if i % 10 == 0:
        print(f"Processed {i} face images")

save_to_pickle(face_indices, os.path.join(assets_folder, "face_ind.pkl"))  # Save face indices

face_embeddings = torch.stack(face_embeddings)
face_faiss_index = create_faiss_index(face_embeddings)
face_faiss_file = os.path.join(assets_folder, "face_faiss_index.index")
faiss.write_index(face_faiss_index, face_faiss_file)