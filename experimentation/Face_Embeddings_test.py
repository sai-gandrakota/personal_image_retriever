import os
import torch
import numpy as np
import faiss
from PIL import Image
from facenet_pytorch import MTCNN
from qai_hub_models.models.openai_clip.model import Clip
from qai_hub_models.models.openai_clip.app import ClipApp


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
clip_model = Clip.from_pretrained()
image_encoder_model = clip_model.image_encoder.to(device)
text_encoder_model = clip_model.text_encoder.to(device)


# Image Paths
reference_dir = "/Users/prasasth/Documents/ref_test"
dataset_dir = "/Users/prasasth/Documents/test_df"

df_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

# Process Reference Embeddings
ref_emb = {}

for f in os.listdir(reference_dir):
    if f.lower().endswith(('jpg', 'jpeg', 'png')):
        name = os.path.splitext(f)[0]
        img_path = os.path.join(reference_dir, f)
        
      
        image = Image.open(img_path)

        image = image.convert("RGB")

        faces = mtcnn(image)
        if faces is None or len(faces) == 0:
            print(f"No faces detected in {img_path}, skipping.")
            continue

        embeddings = []
        for face in faces:
            # Ensure correct shape for CLIP input
            face = torch.nn.functional.interpolate(face.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
            emb = image_encoder_model(face.to(device))
            embeddings.append(emb)

        ref_emb[name] = torch.stack(embeddings).squeeze(0)

# Normalize reference embeddings
for key in ref_emb:
    ref_emb[key] = torch.nn.functional.normalize(ref_emb[key], p=2, dim=1)

# Process Dataset Embeddings
face_emb = []
face_ind = []

for i, img_path in enumerate(df_paths):
    image = Image.open(img_path).convert("RGB")
    faces = mtcnn(image)
    
    if faces is None or len(faces) == 0:
        print(f"No faces detected in {img_path}, skipping.")
        continue

    for face_idx, face in enumerate(faces):
        # Ensure proper input size for CLIP
        face = torch.nn.functional.interpolate(face.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        emb = image_encoder_model(face.to(device))
        face_emb.append(emb)
        face_ind.append((i, face_idx))

# Stack embeddings and normalize
if face_emb:
    face_embeddings = torch.stack(face_emb).squeeze(1)
    face_emb_norm = torch.nn.functional.normalize(face_embeddings, p=2, dim=1)
else:
    print("No face embeddings found in dataset.")
    exit()

# Add dataset embeddings to FAISS
def create_faiss_index(embeddings):
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    dimension = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_np)
    return faiss_index

face_faiss_index = create_faiss_index(face_emb_norm)

# Input Query
query = "me"

if query not in ref_emb:
    print(f"Query '{query}' not found in reference images.")
    exit()

query_emb = ref_emb[query]
print(query_emb.shape)
query_emb = query_emb.cpu().numpy().astype('float32')

# Calculate Similarities
scores, indices = face_faiss_index.search(query_emb, k=face_faiss_index.ntotal)

image_scores = {}
for idx, score in zip(indices[0], scores[0]):
    img_idx, face_idx = face_ind[idx]
    image_path = df_paths[img_idx]

    if image_path not in image_scores:
        image_scores[image_path] = []
    
    image_scores[image_path].append(score)

# Sort results by highest similarity score
image_scores = dict(sorted(image_scores.items(), key=lambda item: max(item[1])))

for path, score in image_scores.items():
    print(f"Image: {os.path.basename(path)}, Similarity Score: {[f'{s:.4f}' for s in score]}")
