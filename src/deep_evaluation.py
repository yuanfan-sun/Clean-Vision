import json
import numpy as np
import cv2
import os
import time
import gradio as gr
import re

from src.client import upload_file
from src.maskrcnn_detector import MaskRCNNDetector
from src.segmenter import SAM_Segmenter

def get_deep_evaluation_prompt(original_image, segmented_images, example_images):
    """
    Builds a prompt for Gemini to evaluate objects without calculating a final score.
    """
    prompt = """
You are a meticulous AI quality assurance inspector. Your task is to identify all cleaning issues in a room.

You will be given:
1.  An 'OVERVIEW IMAGE' showing the entire scene.
2.  A series of 'SEGMENTED OBJECT IMAGES' which are close-ups of individual items.
3.  A few 'EXAMPLE IMAGES' showing a high standard of cleanliness.

**Your instructions are:**
1.  **Identify, Group, and Inspect:** Review all images. For each distinct issue, identify the object(s) involved. If multiple instances of the same object have the same issue (e.g., two dirty mugs), group them into a single entry.
2.  **Score, Justify, and Act:** For each object or group, provide:
    - `object_name`: A short name for the object or group (e.g., "Mugs", "Coffee grounds").
    - `cleanliness_score`: A score from 0-2 based on the rubric.
    - `justification`: A brief description of the issue.
    - `next_action`: A concise, actionable instruction for the cleaner.
3.  **Provide a Summary:** Write a final 1-2 sentence `summary_comment`.

**Rubric for `cleanliness_score`:**
- **Score 2:** Perfectly clean, tidy, and in its correct place.
- **Score 1:** Partially clean or has minor issues.
- **Score 0:** Visibly dirty, messy, or requires significant cleaning.

**Respond in STRICT JSON format ONLY.**

**JSON Output Format (DO NOT include `overall_score`):**
```json
{
  "individual_objects": [
    {
      "object_name": "<Name of the object or group>",
      "cleanliness_score": <number from 0 to 2>,
      "justification": "<Briefly describe the issue.>",
      "next_action": "<A short, clear instruction for the cleaner.>"
    }
  ],
  "summary_comment": "<Your 1-2 sentence summary.>"
}
```
"""
    
    content = [prompt]
    content.append("\n--- EXAMPLE IMAGES OF CLEAN AREAS ---")
    for img in example_images:
        content.append(img)

    content.append("\n--- IMAGE TO EVALUATE ---")
    content.append("OVERVIEW IMAGE:")
    content.append(original_image)
    content.append("\nSEGMENTED OBJECT IMAGES:")

    for i, seg_img in enumerate(segmented_images):
        content.append(f"Object {i+1}:")
        content.append(seg_img)
        
    return content

def perform_deep_evaluation(model, image_array: np.ndarray, progress=gr.Progress()):
    """
    Performs a multi-stage deep evaluation, calculates the score, saves, and returns.
    """
    if image_array is None:
        return {"error": "No image provided."}

    progress(0.1, desc="Detecting objects...")
    maskrcnn_detector = MaskRCNNDetector()
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    instances = maskrcnn_detector.run_detection(image_bgr)
    boxes = instances.pred_boxes.tensor.cpu().numpy() if instances.has("pred_boxes") else np.array([])
    
    if boxes.shape[0] == 0:
        return {"error": "No objects were detected."}

    progress(0.3, desc="Segmenting objects...")
    sam_segmenter = SAM_Segmenter()
    masked_objects_rgb = sam_segmenter.segment_from_boxes(image_array, boxes)

    progress(0.6, desc="Uploading images...")
    # ... (upload logic is the same) ...
    temp_dir = "temp_uploads"; os.makedirs(temp_dir, exist_ok=True)
    original_path = os.path.join(temp_dir, f"deep_eval_original_{int(time.time())}.png")
    cv2.imwrite(original_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    gemini_original_image = upload_file(original_path)
    os.remove(original_path)
    if not gemini_original_image: return {"error": "Failed to upload original image."}
    gemini_segmented_images = []
    for i, img_rgb in enumerate(masked_objects_rgb):
        if img_rgb is not None:
            seg_path = os.path.join(temp_dir, f"deep_eval_seg_{i}_{int(time.time())}.png")
            cv2.imwrite(seg_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            uploaded_img = upload_file(seg_path)
            os.remove(seg_path)
            if uploaded_img: gemini_segmented_images.append(uploaded_img)
    if not gemini_segmented_images: return {"error": "Failed to upload segmented images."}
    example_images = [upload_file(p) for p in ["examples/clean_table.jpg", "examples/clean_bin.jpg"] if os.path.exists(p)]

    progress(0.8, desc="Generating deep evaluation...")
    content = get_deep_evaluation_prompt(gemini_original_image, gemini_segmented_images, example_images)
    response = model.generate_content(content)
    
    try:
        model_output = response.text
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', model_output, re.DOTALL)
        if json_match: model_output = json_match.group(1)
        
        json_data = json.loads(model_output)
        
        # --- Calculate Percentage Score ---
        individual_objects = json_data.get("individual_objects", [])
        if individual_objects:
            actual_score = sum(obj.get("cleanliness_score", 0) for obj in individual_objects)
            max_score = len(individual_objects) * 2
            percentage_score = round((actual_score / max_score) * 100) if max_score > 0 else 100
            json_data["overall_score"] = percentage_score
        else:
            # If no objects are reported, assume it's perfectly clean.
            json_data["overall_score"] = 100

        # --- Save Final JSON ---
        timestamp = int(time.time())
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"deep_eval_{timestamp}.json")
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Deep evaluation result saved to {file_path}")
        
        return json_data

    except (json.JSONDecodeError, AttributeError):
        return {"error": "Could not decode JSON from model output.", "raw_output": response.text}