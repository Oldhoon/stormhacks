import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import cv2  # OpenCV
import numpy as np

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)
model.eval()

# Define the image transformations: resize, center crop, convert to tensor, and normalize
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

# Function to preprocess the camera frame for the model
def preprocess_frame(frame):
    # Convert frame from OpenCV (BGR) to PIL Image (RGB)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Apply the necessary transforms
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Function to make a prediction on the frame
def predict(frame):
    # Preprocess the frame
    image_tensor = preprocess_frame(frame)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Return top prediction with label
    return [(labels[top5_catid[i]], top5_prob[i].item()) for i in range(top5_prob.size(0))]

# Open a connection to the camera
cap = cv2.VideoCapture(0)

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
    predictions = predict(frame)

    # Display the top prediction on the frame
    text = f"{predictions[0][0]}: {predictions[0][1]:.2f}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Camera', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()