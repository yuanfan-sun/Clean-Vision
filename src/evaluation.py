# This file contains the core logic for building prompts and performing evaluations.
import csv
from src.client import upload_file

# Define the list of tasks that the AI can classify and evaluate.
TASKS = [
    "Clean / clear the table surface",
    "Empty / check the trash bin",
    "Clean / organize the whiteboard",
    "Tidy up the windowsill / shelf area",
    "Close windows"
]

def _load_clean_examples():
    """
    Loads the 'clean' (Score=2) example images from the dataset CSV.
    This is a private helper function to avoid code duplication.
    
    Returns:
        A list of content parts (text and images) for the 'Green' examples.
    """
    example_prompt_parts = []
    
    try:
        with open('image_dataset.csv', 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            
            example_text_part = "\n**'Green' Example Images:**\n"
            example_image_parts = []
            images_found = False

            for row in reader:
                try:
                    # Only use examples with a score of 2 as 'Green' examples
                    if row.get('Score') == '2':
                        task_number = int(row['task type'])
                        image_path = row['Image']
                        
                        example_image = upload_file(image_path)
                        if example_image:
                            images_found = True
                            
                            if 1 <= task_number <= len(TASKS):
                                task_name = TASKS[task_number - 1]
                                example_text_part += f"Task {task_number}: {task_name}\n"
                                example_image_parts.append(example_image)

                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row in dataset CSV due to missing or invalid data: {row}. Error: {e}")
            
            if images_found:
                example_prompt_parts.append(example_text_part)
                example_prompt_parts.extend(example_image_parts)

    except FileNotFoundError:
        print("Warning: `image_dataset.csv` not found. This may impact accuracy.")
        
    return example_prompt_parts


def _get_task_classification_content(image):
    """
    Builds the prompt content for classifying an image into one of the 5 main tasks.
    """
    content = []
    prompt_start = """
You are an AI quality assurance inspector for cleaning tasks.
Your task is to classify an image into one of the cleaning tasks below.
Use the 'Green' Example Images provided as a reference to help you determine which of the five tasks is the primary subject of the 'Input Image'.
"""
    content.append(prompt_start)

    content.append("**Tasks:**\n" + "\n".join(f"{i+1}. {task}" for i, task in enumerate(TASKS)))

    # Load and add clean example images from the dataset.
    content.extend(_load_clean_examples())

    prompt_end = "\n**Input Image:**"
    content.append(prompt_end)
    content.append(image)

    prompt_final = """
**Output Format:**

Please provide the output in a JSON format like this:

```json
{
  "classification": {
    "task_number": 1,
    "task_name": "Clean / clear the table surface"
  }
}
```
"""
    content.append(prompt_final)
    return content


def _get_evaluation_content(image):
    """
    Builds the prompt content for both classifying and evaluating an image.
    """
    content = []
    prompt_start = """
You are an AI quality assurance inspector for cleaning tasks.
Your task is to first classify an image into one of the cleaning tasks below, and then evaluate its cleanliness based on the provided rubric and a 'Green' example of a perfectly clean state for that task.
"""
    content.append(prompt_start)

    content.append("**Tasks:**\n" + "\n".join(f"{i+1}. {task}" for i, task in enumerate(TASKS)))

    # Load and add clean example images from the dataset.
    content.extend(_load_clean_examples())

    prompt_end = """
**Rubric:**

| No. | Task | Evaluation Criteria | Green (2 Points) | Orange (1 Point) | Red (0 Points) |
|---|---|---|---|---|---|
| 1 | Clean / clear the table surface | No visible stains or crumbs, no trash left on the surface | Table completely cleared and clean | Some items or stains still visible | Surface largely unchanged or only partially cleaned |
| 2 | Empty / check the trash bin | Bin visibly empty, new bag inserted, no waste on the floor | Bin empty, new bag properly inserted | Partially emptied, bag not replaced | Bin still full or dirty |
| 3 | Clean / organize the whiteboard | Text fully removed without residues, magnets/markers organized | Whiteboard clean, no residues, fully organized | Light traces of text, partially organized | Text still visible, board unorganized |
| 4 | Tidy up the windowsill / shelf area | Waste removed, dirt (e.g., soil) cleared, surface evenly clean | Windowsill clean and free of dirt and waste | Some objects or light dirt still visible | Area messy and dirty |
| 5 | Close windows | All windows closed | All windows closed | Only a windows closed | All Windows open |

**Instructions:**

1.  **Classification:** First, analyze the 'Input Image'. Use the 'Green' Example Images provided as a reference to help you determine which of the five tasks is the primary subject of the 'Input Image'.
2.  **Evaluation:** Once you have classified the image, evaluate its cleanliness for that specific task using the rubric. Compare the 'Input Image' to the corresponding 'Green' example for that task to help you decide the rating.

**Input Image:**
"""
    content.append(prompt_end)
    content.append(image)

    prompt_final = """
**Output Format:**

Please provide the output in a JSON format like this:

```json
{
  "classification": {
    "task_number": 1,
    "task_name": "Clean / clear the table surface"
  },
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

def perform_classification(model, image):
    """
    Performs image classification using the Gemini model.
    """
    content = _get_task_classification_content(image)
    response = model.generate_content(content)
    return response.text

def perform_evaluation(model, image):
    """
    Performs image classification and evaluation using the Gemini model.
    """
    content = _get_evaluation_content(image)
    response = model.generate_content(content)
    return response.text