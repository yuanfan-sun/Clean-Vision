import gradio as gr
import json
import numpy as np
import cv2
import os
import time
import supervision as sv
from pathlib import Path
import pandas as pd
import random

# Import our custom modules
from src.client import initialize_gemini, get_gemini_model, upload_file
from src.evaluation import perform_evaluation
from src.maskrcnn_detector import MaskRCNNDetector
from src.segmenter import SAM_Segmenter
from src.deep_evaluation import perform_deep_evaluation

# --- Initialization ---
print("Initializing models...")
initialize_gemini()
gemini_model = get_gemini_model()
maskrcnn_detector = MaskRCNNDetector()
sam_segmenter = None # Lazy load
print("Core models initialized.")

def get_sam_segmenter():
    global sam_segmenter
    if sam_segmenter is None:
        print("First-time initialization of SAM_Segmenter...")
        sam_segmenter = SAM_Segmenter()
    return sam_segmenter

# --- Report Formatting Helpers ---

def format_quick_report(report_json):
    if not report_json or "evaluation" not in report_json:
        return "### Quick Evaluation\n\nCould not generate a report."
    eval_data = report_json["evaluation"]
    score = eval_data.get('score', 'N/A')
    rating = eval_data.get('rating', 'Unknown')
    justification = eval_data.get('justification', 'No details provided.')
    
    report = f"### Quick Evaluation Result\n\n"
    report += f"**Overall Rating:** {rating} (Score: {score}/2)\n\n"
    report += f"**Justification:** {justification}"
    return report

def format_deep_dive_summary(report_json):
    if not report_json or "individual_objects" not in report_json:
        return "### Deep Dive Summary\n\nNo report to summarize."
    
    score = report_json.get("overall_score", "N/A")
    summary = f"### Deep Dive Summary (Overall Score: {score}%)\n\n"
    
    action_items = [
        (obj.get("object_name", "Unnamed Object"), obj.get("next_action", "No action specified."))
        for obj in report_json["individual_objects"]
        if obj.get("cleanliness_score", 2) < 2
    ]
    
    if not action_items:
        summary += "✅ **Excellent! No actions required.**"
    else:
        summary += "**To-Do List:**\n"
        for i, (name, action) in enumerate(action_items):
            summary += f"{i+1}. **{name}:** {action}\n"
            
    return summary

def format_validation_report(report_json):
    if not report_json or "validation_summary" not in report_json:
        return "### Validation Report\n\nCould not generate a report."
    summary = report_json["validation_summary"]
    total = summary.get("total_images_processed", 0)
    avg_score = summary.get("average_overall_score", "N/A")
    
    report = f"### Validation Benchmark Report\n\n"
    report += f"**Total Images Processed:** {total}\n"
    report += f"**Average Score (0-2 scale):** {avg_score}"
    return report

def format_accuracy_report(report_json):
    if not report_json or "accuracy_summary" not in report_json:
        return "### Accuracy Report\n\nCould not generate a report."
    summary = report_json["accuracy_summary"]
    total = summary.get("total_samples", 0)
    correct = summary.get("correct_predictions", 0)
    accuracy = summary.get("accuracy_percent", "N/A")
    
    report = f"### Ground Truth Accuracy Report\n\n"
    report += f"**Accuracy:** {accuracy}%\n"
    report += f"**Correct Predictions:** {correct} / {total}"
    return report

# --- Backend Functions ---

def run_evaluation(image_array):
    if image_array is None: return "Please upload an image.", gr.update()
    temp_dir = "temp_uploads"; os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_eval_{int(time.time())}.png")
    cv2.imwrite(temp_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    gemini_image = upload_file(temp_path)
    os.remove(temp_path)
    if not gemini_image: return "Failed to upload image.", gr.update()
    json_result = perform_evaluation(gemini_model, gemini_image)
    return format_quick_report(json_result), gr.update(selected="eval_tab")

def run_detection(image_array, progress=gr.Progress()):
    if image_array is None: return None, None, None, None
    progress(0.5, desc="Running Mask R-CNN...")
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    instances = maskrcnn_detector.run_detection(image_bgr)
    annotated_bgr = maskrcnn_detector.visualize_instances(image_bgr, instances)
    annotated_image_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    boxes = instances.pred_boxes.tensor.cpu().numpy() if instances.has("pred_boxes") else np.array([])
    detections_sv = sv.Detections(xyxy=boxes)
    progress(1.0, desc="Done!")
    return annotated_image_rgb, detections_sv, image_array, gr.update(visible=True)

def run_segmentation(image_array, detections, progress=gr.Progress()):
    if image_array is None or detections is None: return None
    progress(0.2, desc="Initializing Segmenter...")
    segmenter = get_sam_segmenter()
    progress(0.6, desc="Running SAM Segmentation...")
    masks = segmenter.segment_from_boxes(image_array, detections.xyxy)
    progress(1.0, desc="Done!")
    return masks

def run_deep_dive_and_summarize(image_array, progress=gr.Progress()):
    if image_array is None: return "Please upload an image first.", gr.update()
    json_result = perform_deep_evaluation(gemini_model, image_array, progress=progress)
    summary = format_deep_dive_summary(json_result)
    return summary, gr.update(selected="deep_eval_tab")

def run_validation_set_evaluation(progress=gr.Progress()):
    # ... (logic is the same, just returns formatted report)
    val_dir = Path("data/val");
    if not val_dir.is_dir(): return f"Validation directory '{val_dir}' not found.", gr.update()
    image_paths = list(val_dir.rglob("*.jpg")) + list(val_dir.rglob("*.png"))
    if not image_paths: return f"No images found in '{val_dir}'.", gr.update()
    total_images, all_results, total_score = len(image_paths), [], 0
    for i, image_path in enumerate(image_paths):
        progress(i / total_images, desc=f"Processing {image_path.name} ({i+1}/{total_images})")
        gemini_image = upload_file(str(image_path))
        if not gemini_image:
            all_results.append({"image_file": str(image_path), "evaluation": {"error": "Failed to upload."}})
            continue
        result = perform_evaluation(gemini_model, gemini_image)
        all_results.append({"image_file": str(image_path), "evaluation": result})
        if "evaluation" in result and "score" in result["evaluation"]: total_score += result["evaluation"]["score"]
    average_score = total_score / total_images if total_images > 0 else 0
    summary_report = {"validation_summary": {"total_images_processed": total_images, "average_overall_score": round(average_score, 2)}, "individual_results": all_results}
    timestamp = int(time.time()); output_dir = "evaluation_results"; os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"validation_report_{timestamp}.json")
    with open(file_path, 'w') as f: json.dump(summary_report, f, indent=2)
    print(f"Validation report saved to {file_path}")
    return format_validation_report(summary_report), gr.update(selected="validation_tab")

def run_ground_truth_validation(progress=gr.Progress()):
    # ... (logic is the same, just returns formatted report)
    dataset_path = Path("data/generated_dataset.csv")
    if not dataset_path.exists(): return "Ground truth file 'data/generated_dataset.csv' not found.", gr.update()
    df = pd.read_csv(dataset_path)
    val_df = df[df['Image'].str.contains('data/val', na=False)].copy()
    if len(val_df) == 0: return "No validation entries found in the dataset CSV.", gr.update()
    sample_size = min(10, len(val_df)); val_sample = val_df.sample(n=sample_size)
    correct_predictions, comparison_results = 0, []
    for i, row in enumerate(val_sample.itertuples()):
        progress(i / sample_size, desc=f"Comparing {Path(row.Image).name} ({i+1}/{sample_size})")
        ground_truth_score = int(row.Score)
        gemini_image = upload_file(row.Image)
        if not gemini_image:
            comparison_results.append({"image_file": row.Image, "error": "Failed to upload."})
            continue
        result = perform_evaluation(gemini_model, gemini_image)
        predicted_score = result.get("evaluation", {}).get("score", -1)
        is_correct = (predicted_score == ground_truth_score)
        if is_correct: correct_predictions += 1
        comparison_results.append({"image_file": row.Image, "ground_truth_score": ground_truth_score, "predicted_score": predicted_score, "is_correct": is_correct, "justification": result.get("evaluation", {}).get("justification", "")})
    accuracy = (correct_predictions / sample_size) * 100 if sample_size > 0 else 0
    report = {"accuracy_summary": {"total_samples": sample_size, "correct_predictions": correct_predictions, "accuracy_percent": round(accuracy, 2)}, "comparison_details": comparison_results}
    timestamp = int(time.time()); output_dir = "evaluation_results"; os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"accuracy_report_{timestamp}.json")
    with open(file_path, 'w') as f: json.dump(report, f, indent=2)
    print(f"Accuracy report saved to {file_path}")
    return format_accuracy_report(report), gr.update(selected="validation_tab")

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI CleanCheck") as demo:
    gr.Markdown("# AI CleanCheck ✅")
    gr.Markdown("A multi-tool for cleaning inspection.")

    original_image_state = gr.State()
    detections_state = gr.State()

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="numpy", label="Upload Image")
            with gr.Tabs():
                with gr.TabItem("Evaluation"):
                    eval_btn = gr.Button("Run Quick Evaluation", variant="primary")
                    deep_eval_btn = gr.Button("Run Deep Dive Evaluation", variant="secondary")
                    with gr.Accordion("Validation Metrics", open=False):
                        validation_btn = gr.Button("Run Validation Benchmark")
                        accuracy_btn = gr.Button("Run Accuracy Test (vs. Ground Truth)")
                with gr.TabItem("Detection & Segmentation"):
                    detect_btn = gr.Button("Detect Objects (Mask R-CNN)", variant="primary")
                    segment_btn = gr.Button("Segment Detected Objects (SAM)", variant="secondary", visible=False)
        with gr.Column(scale=2):
            with gr.Tabs() as output_tabs:
                with gr.TabItem("Evaluation Result", id="eval_tab"):
                    quick_summary_output = gr.Markdown(label="Quick Evaluation Summary")
                with gr.TabItem("Deep Dive Result", id="deep_eval_tab"):
                    deep_summary_output = gr.Markdown(label="Deep Dive Action Summary")
                with gr.TabItem("Validation Report", id="validation_tab"):
                    validation_output = gr.Markdown(label="Validation Report")
                with gr.TabItem("Detection & Segmentation Result", id="detect_tab"):
                    image_output = gr.Image(type="numpy", label="Detection Result")
                    gallery_output = gr.Gallery(label="Segmented Object Masks", columns=4)

    eval_btn.click(fn=run_evaluation, inputs=[image_input], outputs=[quick_summary_output, output_tabs], api_name="evaluate_cleanliness")
    deep_eval_btn.click(fn=run_deep_dive_and_summarize, inputs=[image_input], outputs=[deep_summary_output, output_tabs], api_name="deep_evaluate_cleanliness")
    validation_btn.click(fn=run_validation_set_evaluation, inputs=[], outputs=[validation_output, output_tabs], api_name="run_validation")
    accuracy_btn.click(fn=run_ground_truth_validation, inputs=[], outputs=[validation_output, output_tabs], api_name="run_accuracy_test")
    detect_btn.click(fn=run_detection, inputs=[image_input], outputs=[image_output, detections_state, original_image_state, segment_btn]).then(lambda: gr.update(selected="detect_tab"), None, output_tabs)
    segment_btn.click(fn=run_segmentation, inputs=[original_image_state, detections_state], outputs=[gallery_output], api_name="segment_objects")

if __name__ == "__main__":
    demo.launch(share=True)