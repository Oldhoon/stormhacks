import torch
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import requests

# Load the pre-trained EfficientNet model
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)  # Using EfficientNet B0, you can change to B1, B2, etc.
model.eval()

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the image transformations: resize, center crop, convert to tensor, and normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Define the mapping from ImageNet classes to custom categories
category_mapping = {
    "banana": "organic",
    "apple": "organic",
    "orange": "organic",
    "paper": "paper",
    "newspaper": "paper",
    "bottle": "containers",
    "can": "containers",
    "plastic": "containers",
    "waste": "garbage",
    "trash": "garbage",
    "garbage": "garbage",
    "oil filter" : "container",
}

# Define custom label mapping to replace certain predictions with custom labels
custom_label_mapping = {
    "oil filter": "drink can",  # Replace "oil filter" with "drink can"
    "pill bottle": "drink can",
    "hair spray": "drink can",
}

# Function to preprocess the camera frame for the model
def preprocess_frame(frame):
    # Convert frame from OpenCV (BGR) to PIL Image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the necessary transforms
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)  # Move to GPU if available

# Function to categorize the prediction based on label
def categorize_prediction(predicted_label):
    for keyword, category in category_mapping.items():
        if keyword in predicted_label.lower():
            return category
    return "undefined"  # Default if no match is found

# Function to replace specific labels with custom labels
def apply_custom_label_mapping(predicted_label):
    return custom_label_mapping.get(predicted_label.lower(), predicted_label)

# Function to make a prediction on the frame
def predict(frame):
    # Preprocess the frame
    image_tensor = preprocess_frame(frame)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 1 prediction
    top1_prob, top1_catid = torch.topk(probabilities, 1)
    
    # Return the top 1 prediction with label and probability
    return labels[top1_catid.item()], top1_prob.item()

# Open a connection to the camera
cap = cv2.VideoCapture(1)  # You can change the camera index if necessary

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Run the camera feed in a loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Make predictions
    top_label, probability = predict(frame)

    # Apply custom label mapping if necessary
    top_label = apply_custom_label_mapping(top_label)

    category = categorize_prediction(top_label)  # Categorize the top prediction

    # Display the top prediction on the frame
    text = f"{top_label}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the categorized label at the top of the frame
    category_text = f"Category: {category}"
    cv2.putText(frame, category_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()