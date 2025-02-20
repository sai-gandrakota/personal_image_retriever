import os
import torch
import numpy as np
import faiss
from PIL import Image
from facenet_pytorch import MTCNN
from qai_hub_models.models.openai_clip.model import Clip
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace
from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
clip_model = Clip.from_pretrained()
image_encoder_model = clip_model.image_encoder.to(device)
text_encoder_model = clip_model.text_encoder.to(device)
model = MediaPipeFace.from_pretrained()
face_detector_model = model.face_detector.to(device)
app = MediaPipeFaceApp(model=model)

# Define the transform to convert PIL image to tensor and normalize it as expected by CLIP
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])  # Normalize
])

# Image Paths
reference_dir = "/Users/prasasth/Documents/ref_test"
dataset_dir = "/Users/prasasth/Documents/test_df"

df_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

# Process Reference Embeddings
mtcnn_ref_emb = {}
mp_ref_emb = {}

for f in os.listdir(reference_dir):
    if f.lower().endswith(('jpg', 'jpeg', 'png')):
        name = os.path.splitext(f)[0]
        img_path = os.path.join(reference_dir, f)
        
      
        image = Image.open(img_path)

        image = image.convert("RGB")

        mtcnn_faces = mtcnn(image)

        
        if mtcnn_faces is None or len(mtcnn_faces) == 0:
            print(f"No faces detected in {img_path}, skipping.")
            continue

        embeddings = []
        for face in mtcnn_faces:
            # Ensure correct shape for CLIP input
            face = torch.nn.functional.interpolate(face.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
            emb = image_encoder_model(face.to(device))
            embeddings.append(emb)

        mtcnn_ref_emb[name] = torch.stack(embeddings).squeeze(0)

        embeddings = []
        batched_selected_boxes, batched_selected_keypoints, batched_roi_4corners, *landmarks_out = app.predict_landmarks_from_image(image, raw_output=True)
        for box in batched_selected_boxes[0]:
            # Extract coordinates (x1, y1, x2, y2)
            x1, y1 = box[0][0].int().item(), box[0][1].int().item()  # Convert to integers
            x2, y2 = box[1][0].int().item(), box[1][1].int().item()
            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image = transform(cropped_image).unsqueeze(0).to(device) 
            emb = image_encoder_model(cropped_image)
            embeddings.append(emb)
        mp_ref_emb[name] = torch.stack(embeddings).squeeze(0)
# Normalize reference embeddings
for key in mtcnn_ref_emb:
    mtcnn_ref_emb[key] = torch.nn.functional.normalize(mtcnn_ref_emb[key], p=2, dim=1)
    mp_ref_emb[key] = torch.nn.functional.normalize(mp_ref_emb[key], p=2, dim=1)

# Process Dataset Embeddings
mtcnn_face_emb = []
mtcnn_face_ind = []

mp_face_emb = []
mp_face_ind = []

for i, img_path in enumerate(df_paths):
    image = Image.open(img_path).convert("RGB")
    mtcnn_faces = mtcnn(image)
    
    if mtcnn_faces is None or len(mtcnn_faces) == 0:
        print(f"No faces detected in {img_path}, skipping.")
        continue

    for face_idx, face in enumerate(mtcnn_faces):
        # Ensure proper input size for CLIP
        face = torch.nn.functional.interpolate(face.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
        emb = image_encoder_model(face.to(device))
        mtcnn_face_emb.append(emb)
        mtcnn_face_ind.append((i, face_idx))

    batched_selected_boxes, batched_selected_keypoints, batched_roi_4corners, *landmarks_out = app.predict_landmarks_from_image(image, raw_output=True)
    for face_idx, box in enumerate(batched_selected_boxes[0]):
        # Extract coordinates (x1, y1, x2, y2)
        x1, y1 = box[0][0].int().item(), box[0][1].int().item()  # Convert to integers
        x2, y2 = box[1][0].int().item(), box[1][1].int().item()
        # Crop the image using the bounding box coordinates
        cropped_image = image.crop((x1, y1, x2, y2))
        cropped_image = transform(cropped_image).unsqueeze(0).to(device) 
        emb = image_encoder_model(cropped_image)
        mp_face_emb.append(emb)
        mp_face_ind.append((i, face_idx))


# Stack embeddings and normalize
if mtcnn_face_emb:
    mtcnn_face_embeddings = torch.stack(mtcnn_face_emb).squeeze(1)
    mtcnn_face_emb_norm = torch.nn.functional.normalize(mtcnn_face_embeddings, p=2, dim=1)
else:
    print("No face embeddings found in dataset.")
    exit()
if mp_face_emb:
    mp_face_embeddings = torch.stack(mp_face_emb).squeeze(1)
    mp_face_emb_norm = torch.nn.functional.normalize(mp_face_embeddings, p=2, dim=1)
else:
    print("No face embeddings found in dataset.")
    exit()

# Add dataset embeddings to FAISS
def create_faiss_index(embeddings):
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    dimension = embeddings_np.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings_np)
    return faiss_index

mtcnn_faiss_index = create_faiss_index(mtcnn_face_emb_norm)
mp_faiss_index = create_faiss_index(mp_face_emb_norm)
# Input Query
query = "me"

if query not in mtcnn_ref_emb:
    print(f"Query '{query}' not found in reference images.")
    exit()

query_emb = mtcnn_ref_emb[query]
query_emb = query_emb.cpu().numpy().astype('float32')

# Calculate Similarities
mtcnn_scores, mtcnn_indices = mtcnn_faiss_index.search(query_emb, k=mtcnn_faiss_index.ntotal)

image_mtcnn_scores = {}
for idx, score in zip(mtcnn_indices[0], mtcnn_scores[0]):
    img_idx, face_idx = mtcnn_face_ind[idx]
    image_path = df_paths[img_idx]

    if image_path not in image_mtcnn_scores:
        image_mtcnn_scores[image_path] = []
    
    image_mtcnn_scores[image_path].append(score)

# Sort results by highest similarity score
image_mtcnn_scores = dict(sorted(image_mtcnn_scores.items(), key=lambda item: max(item[1]), reverse=True))

print("MTCNN")
for path, score in image_mtcnn_scores.items():
    print(f"Image: {os.path.basename(path)}, Similarity Score: {[f'{s:.4f}' for s in score]}")

if query not in mp_ref_emb:
    print(f"Query '{query}' not found in reference images.")
    exit()

query_emb = mp_ref_emb[query]
query_emb = query_emb.cpu().numpy().astype('float32')

# Calculate Similarities
mp_scores, mp_indices = mp_faiss_index.search(query_emb, k=mp_faiss_index.ntotal)
image_mp_scores = {}
for idx, score in zip(mp_indices[0], mp_scores[0]):
    img_idx, face_idx = mp_face_ind[idx]
    image_path = df_paths[img_idx]

    if image_path not in image_mp_scores:
        image_mp_scores[image_path] = []
    
    image_mp_scores[image_path].append(score)

# Sort results by highest similarity score
image_mp_scores = dict(sorted(image_mp_scores.items(), key=lambda item: max(item[1]), reverse=True))

print("MediaPipe")
for path, score in image_mp_scores.items():
    print(f"Image: {os.path.basename(path)}, Similarity Score: {[f'{s:.4f}' for s in score]}")