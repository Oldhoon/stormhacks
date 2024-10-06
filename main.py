import os
import torch
import cv2  
from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolov11n-cls.pt")

    model.train(
        data="data.yaml",  
        epochs=5,  
        imgsz=640,  
        device="cuda" if torch.cuda.is_available() else "cpu"  
    )

    # Save the model
    model.save("best_yolov8_model.pt")

    return model

def run_yolo_inference(model, source):
    # If the source is '0', use the webcam
    if source == '0':
        cap = cv2.VideoCapture(0)  # Open the webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()  # Capture frame-by-frame
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Perform object detection on the frame
            results = model(frame)

            # Draw the bounding boxes and labels on the frame
            annotated_frame = results[0].plot()

            cv2.imshow('YOLOv8 Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        results = model(source)
        
        annotated_image = results[0].plot()

        cv2.imshow('YOLOv8 Detection - Image', annotated_image)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

    model.export(format="onnx")

if __name__ == '__main__':
    model_path = "best_yolov8_model.pt"

    if not os.path.exists(model_path):
        print("No pre-trained YOLOv8 model found. Training a new model...")
        model = train_yolo()
    else:
        print("Pre-trained YOLOv8 model found. Loading...")
        model = YOLO(model_path)  

    source_input = input("Enter the path to an image file, video file, or '0' for webcam: ").strip()
    run_yolo_inference(model, source=source_input)
