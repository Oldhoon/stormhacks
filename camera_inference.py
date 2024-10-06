import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
from PIL import Image

# Load the trained model
def load_model(class_names, device):
    model = models.efficientnet_b0(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))
    model.load_state_dict(torch.load('model_weights.pth'))
    model.eval()  # Set model to evaluation mode
    model.to(device)
    return model

# Preprocess the frame for model input
def preprocess_frame(frame):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame = preprocess(frame).unsqueeze(0)  # Add batch dimension
    return frame

# Get model prediction for the frame
def predict_frame(model, frame, device):
    frame = frame.to(device)
    with torch.no_grad():
        outputs = model(frame)
        _, predicted = torch.max(outputs, 1)
    return predicted[0]

# Open the webcam for real-time object classification
def open_camera(class_names, device):
    model = load_model(class_names, device)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_frame = preprocess_frame(frame)
        prediction = predict_frame(model, input_frame, device)

        # Display the prediction on the frame
        cv2.putText(frame, f'Prediction: {class_names[prediction]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object Classification', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
