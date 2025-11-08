# This file contains helper functions for interacting with the Google Gemini API.
import google.generativeai as genai
import os
from dotenv import load_dotenv

def initialize_gemini():
    """
    Loads environment variables from a .env file and configures the Gemini API key.
    This function must be called before any other Gemini API operations.
    """
    # Load environment variables from a .env file in the project root
    load_dotenv()
    try:
        # Configure the Gemini API with the key from the environment variables
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    except KeyError:
        # Handle the case where the API key is not found in the environment
        print("GEMINI_API_KEY not found in .env file. Please create a .env file and add your API key.")
        exit()

def get_gemini_model(model_name='gemini-2.5-pro'):
    """
    Initializes and returns a GenerativeModel instance.
    
    Args:
        model_name (str): The name of the Gemini model to use.
    
    Returns:
        A Gemini GenerativeModel instance.
    """
    return genai.GenerativeModel(model_name)

def upload_file(file_path):
    """
    Uploads a file to the Gemini API. This is necessary for including images in prompts.
    
    Args:
        file_path (str): The local path to the file to upload.
        
    Returns:
        The uploaded file object, or None if an error occurred.
    """
    try:
        # The display name is what the file will be called in the API
        return genai.upload_file(path=file_path, display_name=os.path.basename(file_path))
    except FileNotFoundError:
        print(f"Error: The file was not found at the specified path: {file_path}")
        return None
    except Exception as e:
        # Catch other potential exceptions during file upload
        print(f"An unexpected error occurred while uploading {file_path}: {e}")
        return None