import requests, logging
from config import Config  


logging.basicConfig(level=logging.INFO)
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
            "do_sample": True,
            "max_length": 700,
            "num_return_sequences": 1,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7, 
            "use_cache": False,
            "skip_special_tokens": False,
            "stop": ["[END]"],
            "output_scores": True,
            "decode": False
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        response.raise_for_status()  
        result = response.json()
        
        story = result[0].get('generated_text')
        logging.info(f"Raw story: {story}")

        
        story = story.split("[SEP]")[-1].strip() 
        story = story.split('[END]')[0].strip() 
        story = story.strip()

        return story
    except Exception as e:
        logging.error(f"Error calling Hugging Face API: {e}")
        return None
