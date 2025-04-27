import requests
import logging
from config import Config  


HF_API_URL = Config.HF_API_URL
HF_API_TOKEN = Config.HF_API_TOKEN

headers = {
    "Accept" : "application/json",
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def generate_from_huggingface_api(title, genre, length):
    prompt = f"Type of Literature: {genre} Length: {length} Title: {title} [SEP]"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 700,
            "num_return_sequences": 1,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7
        }
    }

    try:
        # Make the POST request to Hugging Face API
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        
        # Extract the story from the API response
        story = result[0].get('generated_text', "")
        
        # Process the response to extract meaningful content
        story = story.split("[SEP]")[-1].strip()  # Get part after [SEP]
        story = story.split('[END]')[0].strip()  # Get part before [END]

        return story
    except Exception as e:
        logging.error(f"Error calling Hugging Face API: {e}")
        return None
