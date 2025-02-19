import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from qai_hub_models.models.mediapipe_face.model import MediaPipeFace
from qai_hub_models.models.mediapipe_face.app import MediaPipeFaceApp
from qai_hub_models.models.face_det_lite.app import FaceDetLiteApp
from qai_hub_models.models.face_det_lite.model import FaceDetLite_model

# Environment setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Qualcomm AI Hub MediaPipe face model
model = MediaPipeFace.from_pretrained()
face_detector_model = model.face_detector.to(device)
app = MediaPipeFaceApp(model=model)

# Image Paths
reference_dir = "/Users/prasasth/Documents/ref_test"
dataset_dir = "/Users/prasasth/Documents/test_df"

df_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

print(df_paths[3])

# Open the image
image = Image.open(df_paths[3]).convert('RGB')
# image_width, image_height = image.size  # Get the image dimensions

# Run landmark detection
result = app.predict_landmarks_from_image(image)
print(result)
out_image = Image.fromarray(result[0], "RGB")
out_image.show()

# # Check if the result contains any landmarks
# if result[3] is None:
#     print("No landmarks detected")
# else:
#     landmarks = result[3][0]

#     # Convert the landmarks tensor to a NumPy array if needed
#     landmarks = landmarks.cpu().numpy() if isinstance(landmarks, torch.Tensor) else landmarks

#     # Print out the landmarks to verify their structure and values
#     print("Landmarks (first 5):", landmarks[:5])  # Check the first few landmarks

#     # Ignore the confidence scores (third column) and just take the x, y coordinates
#     x_coords = landmarks[:, 0] * image_width
#     y_coords = landmarks[:, 1] * image_height

#     # Calculate the bounding box based on the min/max coordinates
#     min_x, min_y = np.min(x_coords), np.min(y_coords)
#     max_x, max_y = np.max(x_coords), np.max(y_coords)

#     print(f"Bounding box: ({min_x}, {min_y}) -> ({max_x}, {max_y})")

#     # Draw the bounding box on the image using Pillow
#     draw = ImageDraw.Draw(image)
#     draw.rectangle([min_x, min_y, max_x, max_y], outline="green", width=3)

#     # Display the image with the bounding box
#     image.show()
