# This script is used to bootstrap the image datasets.
# It scans the 'data/train' and 'data/val' directories, analyzes each image using the AI model,
# and generates a corresponding CSV file with the analysis.

import os
import csv
import json
from src.client import initialize_gemini, get_gemini_model, upload_file
from src.evaluation import TASKS

def get_bootstrap_prompt():
    """
    Returns the prompt used for bootstrapping the dataset.
    This prompt asks the model to classify and evaluate an image without prior examples.
    """
    return """
You are an AI quality assurance inspector for cleaning tasks.
Your task is to analyze the provided image and generate data for a dataset.

**Tasks:**
1. Clean / clear the table surface
2. Empty / check the trash bin
3. Clean / organize the whiteboard
4. Tidy up the windowsill / shelf area
5. Close windows

**Rubric:**
- Score 2 (Green): Task fully and visibly completed; clear before/after difference.
- Score 1 (Orange): Task started but not fully completed.
- Score 0 (Red): Task not visibly completed or unclear if performed; no improvement from starting state.

**Instructions:**
1.  **Classification:** First, determine which of the five tasks is the primary subject of the input image.
2.  **Evaluation:** Evaluate the cleanliness of the image for the classified task using the rubric. Provide a score (0, 1, or 2) and a brief justification for your score (the 'reason').

**Input Image:**
[The image to process]

**Output Format:**
Please provide the output in a JSON format like this:
```json
{
  "classification": {
    "task_number": 1,
    "task_name": "Clean / clear the table surface"
  },
  "evaluation": {
    "score": 1,
    "justification": "The table has some crumbs on it."
  }
}
```
"""

def process_directory(model, input_dir, output_csv):
    """
    Processes all images in a directory and its subdirectories, and writes the analysis to a CSV file.
    
    Args:
        model: The Gemini GenerativeModel instance.
        input_dir (str): The directory containing the images to process.
        output_csv (str): The path to the output CSV file.
    """
    print(f"--- Processing directory: {input_dir} ---")
    
    # Find all image paths recursively
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print(f"No images found in {input_dir}.")
        return

    # Prepare the CSV file and write the header
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'Score', 'reason', 'task type', 'place']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for image_path in image_paths:
            print(f"  - Analyzing {image_path}...")

            # Upload the image
            image = upload_file(image_path)
            if not image:
                continue

            # Get the analysis from the model
            prompt = get_bootstrap_prompt()
            response = model.generate_content([prompt, image])
            
            # Parse the model's response
            try:
                model_output = response.text
                if model_output.strip().startswith('```json'):
                    model_output = model_output.strip()[7:-4]
                
                data = json.loads(model_output)
                
                task_type = data.get('classification', {}).get('task_number')
                score = data.get('evaluation', {}).get('score')
                reason = data.get('evaluation', {}).get('justification')

                # Write the data to the CSV file
                if task_type is not None and score is not None and reason is not None:
                    writer.writerow({
                        'Image': image_path,
                        'Score': score,
                        'reason': reason,
                        'task type': task_type,
                        'place': ''  # Place is left empty as requested
                    })
                else:
                    print(f"    - Warning: Could not extract all required fields from model output for {image_path}.")

            except (json.JSONDecodeError, AttributeError) as e:
                print(f"    - Error processing model response for {image_path}: {e}")
                print(f"      Model output was: {response.text}")


def main():
    """
    Main function to run the dataset bootstrapping process.
    """
    initialize_gemini()
    model = get_gemini_model()
    
    # Define the directories and output files
    train_dir = 'data/train'
    val_dir = 'data/val'
    train_csv = 'train_dataset.csv'
    val_csv = 'val_dataset.csv'
    
    # Create directories if they don't exist to avoid errors
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Process both the training and validation directories
    process_directory(model, train_dir, train_csv)
    process_directory(model, val_dir, val_csv)
    
    print("\n--- Dataset bootstrapping complete! ---")
    print(f"Training data saved to: {train_csv}")
    print(f"Validation data saved to: {val_csv}")


if __name__ == '__main__':
    main()