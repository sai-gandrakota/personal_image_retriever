import os
import faiss
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle

def load_images(folder):
    """
    Loads image file paths from a given folder.
    Returns a list of image paths for supported formats (.jpg, .jpeg, .png, .JPG).
    """
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.JPG')):
            image_path = os.path.join(folder, filename)
            images.append(image_path)
    return images

def load_model(device, model_choice):
    """
    Loads a specified model (CLIP or JINA) onto the given device (CPU/GPU).
    Returns the processor and model.
    """
    if model_choice == "CLIP":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    elif model_choice == "JINA":
        processor = AutoProcessor.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True)
        model = AutoModel.from_pretrained("jinaai/jina-clip-v2", trust_remote_code=True).to(device)
    else:
        print("Please select a valid model")
        return None
    return processor, model

def get_image_embedding(image_path, processor, model, device):
    """
    Extracts an image embedding using the specified processor and model.
    Returns a normalized embedding tensor.
    """
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return torch.nn.functional.normalize(outputs, p=2, dim=1)

def get_text_embedding(text_query, processor, model, device):
    """
    Extracts a text embedding using the specified processor and model.
    Returns a normalized embedding tensor.
    """
    inputs = processor(text=text_query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return torch.nn.functional.normalize(outputs, p=2, dim=1)

def create_faiss_index(embeddings):
    """
    Creates a FAISS index from a set of embeddings.
    Returns a FAISS index for similarity search.
    """
    embeddings_np = embeddings.numpy().astype('float32')  # Convert to NumPy array
    dimension = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)  # Initialize FAISS index for inner product search
    faiss_index.add(embeddings_np)  # Add embeddings to the index
    return faiss_index

def load_fr_model(device, weights_path):
    """
    Loads a face recognition model (MTCNN for face detection, InceptionResnetV1 for feature extraction).
    Returns the MTCNN and ResNet models.
    """
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained=None).eval().to(device)
    
    # Load pretrained weights for InceptionResnetV1
    pretrained_dict = torch.load(weights_path)
    filtered_dict = {k: v for k, v in pretrained_dict.items() if k in resnet.state_dict()}
    resnet.load_state_dict(filtered_dict, strict=False)
    
    return mtcnn, resnet

def get_face_embeddings(img_path, mtcnn, resnet):
    """
    Extracts face embeddings from an image using MTCNN and InceptionResnetV1.
    Returns normalized face embeddings or None if no faces are detected.
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    faces = mtcnn(img)
    if faces is None:
        return None
    return torch.nn.functional.normalize(resnet(faces).detach().cpu(), p=2, dim=1)

def compute_rrf(rank_clip, rank_face, k=30):
    """
    Computes Reciprocal Rank Fusion (RRF) score for two ranked lists.
    Returns an RRF score for fusion-based ranking.
    """
    return 1 / (k + rank_clip) + 1 / (k + rank_face)

def normalize_embeddings(embeddings):
    """
    Normalizes embeddings using L2 normalization.
    Returns normalized embeddings.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def save_to_pickle(obj, file_path):
    """
    Saves an object to a file using pickle.
    """
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def load_from_pickle(file_path):
    """
    Loads an object from a pickle file.
    Returns the loaded object.
    """
    with open(file_path, 'rb') as f:
        obj = pickle.load(f)
    return obj
