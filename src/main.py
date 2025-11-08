# This file is the main entry point for the application.
# It provides a command-line interface (CLI) to interact with the different
# functionalities of the AI CleanCheck tool.

import argparse
import os
import json
from datetime import datetime
from PIL import Image
from src.client import initialize_gemini, get_gemini_model, upload_file
from src.evaluation import perform_classification, perform_evaluation

def get_image_timestamp(image_path):
    """
    Extracts the timestamp from the image's EXIF data ('Date Taken').
    Falls back to the file's last modification time if EXIF data is not available.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        str: The timestamp in ISO 8601 format.
    """
    try:
        # Open the image file using Pillow
        with Image.open(image_path) as img:
            # Get EXIF data if it exists
            exif_data = img._getexif()
            if exif_data and 36867 in exif_data:
                # EXIF tag 36867 corresponds to DateTimeOriginal
                date_time_original = exif_data[36867]
                # Parse the EXIF timestamp and format it as ISO 8601
                dt_object = datetime.strptime(date_time_original, '%Y:%m:%d %H:%M:%S')
                return dt_object.isoformat()
    except Exception:
        # If any error occurs (e.g., no EXIF data, file not an image),
        # fall back to the file's modification time.
        pass
    
    # Fallback to file modification time
    mod_time = os.path.getmtime(image_path)
    return datetime.fromtimestamp(mod_time).isoformat()


def main():
    """
    The main function that sets up the CLI, parses arguments, and executes commands.
    """
    # Initialize the Gemini client and get the model.
    initialize_gemini()
    model = get_gemini_model()

    # Set up the main argument parser.
    parser = argparse.ArgumentParser(description="AI CleanCheck: Using AI for Enhanced Quality Assurance in Cleaning.")
    subparsers = parser.add_subparsers(dest='command', required=True, help="Available commands")

    # --- 'classify' command ---
    parser_classify = subparsers.add_parser('classify', help='Classify one or more images into one of the 5 main tasks.')
    parser_classify.add_argument('--images', type=str, nargs='+', required=True, help='Paths to the images to be classified.')

    # --- 'evaluate' command ---
    parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate one or more images for cleanliness.')
    parser_evaluate.add_argument('--images', type=str, nargs='+', required=True, help='Paths to the images to be evaluated.')

    # Parse the command-line arguments provided by the user.
    args = parser.parse_args()

    # --- Command Execution ---
    # Loop through each image path provided by the user.
    for image_path in args.images:
        print(f"--- Processing {image_path} ---")
        
        # Get image metadata before performing any heavy operations.
        image_name = os.path.basename(image_path)
        timestamp = get_image_timestamp(image_path)
        
        # Upload the image to the Gemini API.
        image = upload_file(image_path)
        if not image:
            # If upload fails, skip to the next image.
            continue

        # Perform the requested command (classify or evaluate).
        if args.command == 'classify':
            model_output = perform_classification(model, image)
        elif args.command == 'evaluate':
            model_output = perform_evaluation(model, image)
        
        # The model's output is expected to be a JSON string.
        # We need to parse it to combine it with our own metadata.
        try:
            # The model sometimes wraps the JSON in markdown ```json ... ```, so we clean it.
            if model_output.strip().startswith('```json'):
                model_output = model_output.strip()[7:-4]
            result_data = json.loads(model_output)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from model output for {image_path}")
            print(f"Model output:\n{model_output}")
            continue

        # Construct the final JSON output with the added metadata.
        final_output = {
            "image_name": image_name,
            "timestamp": timestamp,
            "result": result_data
        }
        
        # Print the final, enriched JSON to the console.
        print(json.dumps(final_output, indent=2))


if __name__ == '__main__':
    # This ensures that the main() function is called only when the script is executed directly.
    main()