import torch
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights  # Import weight enums
from torchvision import transforms
from PIL import Image
import cv2  # OpenCV
import numpy as np

# Load the pre-trained EfficientNet model
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.eval()

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the ImageNet class labels
import json
import requests

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
    "oil filter": "container",
}

# Function to preprocess the camera frame for the model
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to categorize the prediction
def categorize_prediction(predicted_label):
    # Loop through the category mapping to find the category
    for keyword, category in category_mapping.items():
        if keyword in predicted_label.lower():
            return category
    return "unknown"  # Default if no match is found

# Function to make a prediction on the frame
def predict(frame):
    image_tensor = preprocess_frame(frame)
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top1_prob, top1_catid = torch.topk(probabilities, 1)  # Get the top prediction
    predicted_label = labels[top1_catid]
    
    # Categorize the prediction
    category = categorize_prediction(predicted_label)
    return predicted_label, top1_prob.item(), category

# Open a connection to the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

# Run the camera feed in a loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Make predictions
    predicted_label, probability, category = predict(frame)

    # Display the top prediction and its category on the frame
    text = f"{predicted_label} ({category}): {probability:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Check if the window is closed
    if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()