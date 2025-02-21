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
image_set_folder = os.path.join(script_dir, "..", "dataset", "image_collection")  # Folder containing image collection
reference_folder = os.path.join(script_dir, "..", "dataset", "reference_images")  # Folder containing reference images

# Load image paths from the dataset folder
image_paths = load_images(image_set_folder)
save_to_pickle(image_paths, os.path.join(assets_folder, "img_paths.pkl"))  # Save image paths for later use

# Process multiple models (CLIP and JINA)
#model_choices = ["CLIP", "JINA"]
model_choices = ["CLIP"]
for model_choice in model_choices:
    print(f"Processing for model: {model_choice}")
    processor, model = load_model(device, model_choice)  # Load the selected model
    
    # Generate image embeddings and store them in a FAISS index
    image_embeddings = []
    for i, image_path in enumerate(image_paths):
        embeddings = get_image_embedding(image_path, processor, model, device, model_choice)  # Get embeddings for each image
        if model_choice == "JINA":
            embeddings = embeddings[:256]  # Trim embedding size for JINA model compatibility
        image_embeddings.append(embeddings)
        if i % 10 == 0:
            print(f"Processed {i} images for {model_choice}")
    
    # Convert list of embeddings to a tensor and create FAISS index
    image_embeddings = torch.cat(image_embeddings)  # Concatenate all embeddings into one tensor
    img_faiss_index = create_faiss_index(image_embeddings)  # Create a FAISS index for image embeddings
    img_faiss_file = os.path.join(assets_folder, f"{model_choice}_faiss_index.index")  # Define file path for saving the index
    faiss.write_index(img_faiss_index, img_faiss_file)  # Save FAISS index to file

# Load face recognition model (MediaPipe for face detection and CLIP for feature extraction)
mediapipe_app, clip_encoder = load_fr_model(device)

# Process reference images for face recognition
reference_embeddings = {}  # Dictionary to store embeddings for reference images
for ref_image in os.listdir(reference_folder):
    if ref_image.endswith((".jpg", ".jpeg", ".png")):  # Process only supported image formats
        img_path = os.path.join(reference_folder, ref_image)
        person_name = os.path.splitext(ref_image)[0]  # Extract person name from filename (without extension)
        embeddings = get_face_embeddings(img_path, mediapipe_app, clip_encoder, device)  # Get face embeddings
        if embeddings is not None:
            reference_embeddings[person_name] = torch.stack(embeddings).squeeze(0)  # Store the embeddings in dictionary

# Save reference embeddings for later use
save_to_pickle(reference_embeddings, os.path.join(assets_folder, "ref_emb.pkl"))

# Generate face embeddings from dataset images and store them
face_embeddings = []  # List to store face embeddings
face_indices = []  # List to store indices for each face (image index, face index)
for i, image_path in enumerate(image_paths):
    embeddings = get_face_embeddings(image_path, mediapipe_app, clip_encoder, device)  # Extract face embeddings
    if embeddings is not None:
        for face_idx, face_embedding in enumerate(embeddings):
            face_embeddings.append(face_embedding)  # Append individual face embeddings
            face_indices.append((i, face_idx))  # Track which image and face it belongs to
    if i % 10 == 0:
        print(f"Processed {i} face images")

# Save face indices to keep track of where embeddings came from
save_to_pickle(face_indices, os.path.join(assets_folder, "face_ind.pkl"))

# Convert list of face embeddings to a tensor and normalize them
face_embeddings = torch.stack(face_embeddings).squeeze(1)  # Stack embeddings and remove singleton dimension
face_emb_norm = torch.nn.functional.normalize(face_embeddings, p=2, dim=1)  # L2 normalize the embeddings

# Create and save FAISS index for face embeddings
face_faiss_index = create_faiss_index(face_emb_norm)  # Create a FAISS index for face embeddings
face_faiss_file = os.path.join(assets_folder, "face_faiss_index.index")  # Define file path for saving the index
faiss.write_index(face_faiss_index, face_faiss_file)  # Save FAISS index to file
