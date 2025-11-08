import argparse
from ultralytics import YOLO
import os

def detect(image_path):
    """
    Performs object detection on an image using a pre-trained YOLOv8 model.
    
    Args:
        image_path (str): The path to the input image.
    """
    # Load a pre-trained YOLOv8 model.
    # 'yolov8n.pt' is a small and fast model suitable for general use.
    # Other models like 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' are available
    # offering higher accuracy at the cost of speed and computational resources.
    print("Loading YOLOv8 model...")
    model = YOLO('yolov8n.pt')

    # Run inference on the source image.
    print(f"Running detection on {os.path.basename(image_path)}...")
    results = model(image_path)

    # --- Process and Print Results ---
    print(f"\n--- Detections for {os.path.basename(image_path)} ---")
    
    # The result object contains the detection information.
    # We iterate through each detected object.
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the class name from the model's names dictionary.
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # Get the confidence score of the detection.
            confidence = float(box.conf[0])
            
            # Get the bounding box coordinates (xyxy format).
            x1, y1, x2, y2 = box.xyxy[0]
            
            print(f"  - Object: {class_name}")
            print(f"    Confidence: {confidence:.2f}")
            print(f"    Bounding Box: (x1={x1:.0f}, y1={y1:.0f}), (x2={x2:.0f}, y2={y2:.0f})")

    # --- Save the annotated image ---
    # The results object has a 'save()' method that draws the detections on the image.
    # By default, it saves to a 'runs/detect/predict' directory.
    # We can specify a filename to have more control.
    output_dir = "detection_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    
    results[0].save(filename=output_path)
    print(f"\nAnnotated image saved to: {output_path}")


if __name__ == '__main__':
    # Set up the command-line argument parser.
    parser = argparse.ArgumentParser(description="Detect objects in an image using YOLOv8.")
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    # Check if the provided image path exists before running detection.
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
    else:
        detect(args.image)
