# This file contains the core logic for building prompts and performing evaluations.
import os
import json
import time
from src.client import upload_file

def _load_general_examples():
    """
    Loads curated 'clean' example images from the 'examples/' directory to provide context.
    """
    example_prompt_parts = []
    example_files = [
        "examples/clean_table.jpg",
        "examples/clean_bin.jpg",
        "examples/clean_whiteboard.jpg"
    ]

    example_text_part = "\n**Example Images of Clean Areas:**\n"
    example_image_parts = []
    images_found = False

    for path in example_files:
        if os.path.exists(path):
            example_image = upload_file(path)
            if example_image:
                images_found = True
                example_image_parts.append(example_image)

    if images_found:
        example_prompt_parts.append(example_text_part)
        example_image_parts.extend(example_image_parts)
    else:
        print("Warning: No example images found in 'examples/' directory.")
        
    return example_prompt_parts

def _get_evaluation_content(image):
    """
    Builds a generic prompt for evaluating the cleanliness of an image.
    """
    content = []
    prompt_start = """
You are an AI quality assurance inspector. Your task is to evaluate the cleanliness of the 'Input Image'.
Use the 'Example Images of Clean Areas' as a reference for a high standard of cleanliness.

**Instructions:**
Evaluate the cleanliness of the 'Input Image' using the following simple rubric:
- **Score 2 (Green):** The area is completely clean, tidy, and well-organized.
- **Score 1 (Orange):** The area is partially clean or has minor issues.
- **Score 0 (Red):** The area is visibly dirty or messy.

Provide a score and a brief justification for your rating.

**Respond in STRICT JSON format ONLY.**
"""
    content.append(prompt_start)

    # Load and add general clean example images.
    content.extend(_load_general_examples())

    prompt_end = "\n**Input Image:**"
    content.append(prompt_end)
    content.append(image)

    prompt_final = """
**Output Format:**
```json
{
  "evaluation": {
    "rating": "Orange",
    "score": 1,
    "justification": "The table has some crumbs and a cup left on it."
  }
}
```
"""
    content.append(prompt_final)
    return content

def perform_evaluation(model, image):
    """
    Performs a general cleanliness evaluation and saves the result to a file.
    """
    content = _get_evaluation_content(image)
    response = model.generate_content(content)
    
    # --- Save the output ---
    try:
        model_output = response.text
        if model_output.strip().startswith('```json'):
            model_output = model_output.strip()[7:-4]
        
        # Save the JSON to a file
        timestamp = int(time.time())
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"quick_eval_{timestamp}.json")
        
        # Re-parse to save it pretty-printed
        json_data = json.loads(model_output)
        with open(file_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Quick evaluation result saved to {file_path}")
        
        return json_data # Return the parsed JSON object

    except (json.JSONDecodeError, AttributeError):
        return { "error": "Could not decode JSON from model output.", "raw_output": response.text }