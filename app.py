import gradio as gr
import json
import numpy as np
import cv2
import os
import time

# Import our custom modules
from src.client import initialize_gemini, get_gemini_model, upload_file
from src.evaluation import perform_evaluation
from src.maskrcnn_detector import MaskRCNNDetector

# --- Initialization ---
# Initialize the AI models once when the app starts.
print("Initializing models...")
initialize_gemini()
gemini_model = get_gemini_model()
maskrcnn_detector = MaskRCNNDetector()
print("Models initialized.")

# --- Backend Functions ---

def run_evaluation(image_array):
    """
    Takes an image, saves it temporarily, and runs the Gemini evaluation.
    """
    if image_array is None:
        return { "error": "No image provided." }

    # Gradio provides the image as a NumPy array (H, W, C). cv2 needs BGR.
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Save to a temporary file to upload to Gemini
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = int(time.time())
    temp_path = os.path.join(temp_dir, f"temp_image_{timestamp}.png")
    cv2.imwrite(temp_path, image_bgr)

    # Upload file and perform evaluation
    gemini_image = upload_file(temp_path)
    if gemini_image is None:
        return { "error": "Failed to upload image for evaluation." }

    # Get the raw JSON string from the model
    model_output = perform_evaluation(gemini_model, gemini_image)
    
    # Clean up the temporary file
    os.remove(temp_path)

    # Parse the JSON output and return it
    try:
        if model_output.strip().startswith('```json'):
            model_output = model_output.strip()[7:-4]
        result_data = json.loads(model_output)
        return result_data
    except json.JSONDecodeError:
        return { "error": "Could not decode JSON from model output.", "raw_output": model_output }


def run_detection_step1(image_array):
    """
    Takes an image and runs the initial object detection.
    Returns the annotated image, a list of detected classes for the filter, and the raw instances.
    """
    if image_array is None:
        return None, gr.update(choices=[], value=[], visible=False), None

    # Gradio provides RGB, convert to BGR for cv2 and Detectron2
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    # Run detection
    instances = maskrcnn_detector.run_detection(image_bgr)
    
    # Get the annotated image (with all detections)
    annotated_image_bgr = maskrcnn_detector.visualize_instances(image_bgr, instances)
    annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)

    # Get unique class names for the filter
    unique_classes = maskrcnn_detector.get_unique_class_names(instances)
    
    # Return the annotated image, update the checkbox group with choices, and the raw instances
    return annotated_image_rgb, gr.update(choices=unique_classes, value=unique_classes, visible=True), instances


def filter_detected_image(image_array, instances, selected_classes):
    """
    Re-draws the annotations on an image based on the user's filter selection.
    """
    if image_array is None or instances is None:
        return None
        
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Visualize with the filtered classes
    filtered_image_bgr = maskrcnn_detector.visualize_instances(image_bgr, instances, classes_to_show=selected_classes)
    filtered_image_rgb = cv2.cvtColor(filtered_image_bgr, cv2.COLOR_BGR2RGB)
    
    return filtered_image_rgb


# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft(), title="AI CleanCheck") as demo:
    gr.Markdown("# AI CleanCheck ✅")
    gr.Markdown("An AI-powered tool for classifying, evaluating, and detecting objects in images.")

    with gr.Row():
        with gr.Column(scale=1):
            # --- Inputs ---
            image_input = gr.Image(type="numpy", label="Upload Image")
            analysis_choice = gr.Radio(
                ["Evaluate Cleanliness", "Detect Objects (Mask R-CNN)"],
                label="Choose Analysis Type"
            )
            submit_btn = gr.Button("Run Analysis", variant="primary")
            
            # --- State Management (for multi-step detection) ---
            original_image_state = gr.State()
            detection_instances_state = gr.State()

        with gr.Column(scale=2):
            # --- Outputs ---
            json_output = gr.JSON(label="Evaluation Result", visible=False)
            image_output = gr.Image(type="numpy", label="Detection Result", visible=False)
            filter_checkboxes = gr.CheckboxGroup(label="Filter Detected Objects", visible=False)

    # --- Event Handling ---

    def main_handler(image, choice):
        """
        Main handler that routes to the correct function based on user's choice.
        It returns updates for the 5 output components.
        """
        if image is None:
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None

        if choice == "Evaluate Cleanliness":
            json_res = run_evaluation(image)
            # Show JSON output, hide detection outputs
            return gr.update(value=json_res, visible=True), gr.update(visible=False), gr.update(visible=False), None, None
        
        elif choice == "Detect Objects (Mask R-CNN)":
            annotated_img, checkbox_update, instances = run_detection_step1(image)
            # Hide JSON output, show detection outputs and filter
            return gr.update(visible=False), gr.update(value=annotated_img, visible=True), checkbox_update, image, instances
        
        # Default return if no choice is made
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), None, None

    # When the user clicks the main "Run Analysis" button
    submit_btn.click(
        fn=main_handler,
        inputs=[image_input, analysis_choice],
        outputs=[
            json_output,
            image_output,
            filter_checkboxes,
            original_image_state,
            detection_instances_state
        ]
    )

    # When the user changes the filter checkboxes
    filter_checkboxes.change(
        fn=filter_detected_image,
        inputs=[original_image_state, detection_instances_state, filter_checkboxes],
        outputs=[image_output]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()