import requests, logging
from config import Config
from openai import OpenAI  


logging.basicConfig(level=logging.INFO)
HF_API_URL = Config.HF_API_URL
HF_API_TOKEN = Config.HF_API_TOKEN
AI_API_KEY = Config.AI_API_KEY

headers = {
    "Accept" : "application/json",
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}
client = OpenAI(api_key=AI_API_KEY)

def generate_from_huggingface_api(title, genre, theme, length):
    prompt = f"Type of Literature: {genre} Theme: {theme} Length: {length} Title: {title} [SEP]"

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

def generate_from_openai_api(title, genre, theme, length):
    logging.info(f"Received request with title: {title}, genre: {genre}, theme: {theme}, length: {length}")
    system_prompt = (
        "You are a helpful assistant that generates simple children stories in pure Tagalog. "
        "Follow these rules strictly:\n"
        "- If Length is 'Short', the story must be between 75 and 100 words.\n"
        "- If Length is 'Long', the story must be between 101 and 250 words.\n"
        "- Do NOT include a title, explanation, or any extra text — return ONLY the story body.\n"
        "- The story must be written fluently and naturally in Tagalog, based on the given Genre, Theme, Title, and Length."
        "- The story must be appropriate for children, avoiding any adult themes or language.\n"
        "- Make it look AI generated, and dumb it down a little."
    )
    user_prompt = f"Genre: {genre} Theme: {theme} Length: {length} Title: {title}"
    try:
        logging.info(f"Calling API with prompt: {user_prompt}")
        logging.info(f"System prompt: {system_prompt}")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            store=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        result = response.choices[0].message.content
        logging.info(f"Raw story: {result}")
        return result
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None