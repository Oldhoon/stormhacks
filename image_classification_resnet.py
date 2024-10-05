import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Load the pre-trained ResNet model
model = models.resnet50(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Define the image transformations: resize, center crop, convert to tensor, and normalize
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Make a prediction
def predict(image_path):
    image_tensor = preprocess_image(image_path)

    # Disable gradient calculation for inference
    with torch.no_grad():
        output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    return top5_prob, top5_catid

# Load the class labels from ImageNet
import json
import requests

LABELS_URL = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels = requests.get(LABELS_URL).json()

# Predict the image and display the top 5 results
image_path = "can.png"  # Replace with your image path
top5_prob, top5_catid = predict(image_path)

for i in range(top5_prob.size(0)):
    print(f"{labels[top5_catid[i]]}: {top5_prob[i].item()}")