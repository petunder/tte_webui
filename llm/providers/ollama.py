import requests
import logging
#from configparser import ConfigParser
from classes.settings import Settings  # Импорт класса Settings

logger = logging.getLogger(__name__)

# Создайте экземпляр класса Settings
settings = Settings()
# Извлеките URL из настроек
ollama_url = settings.get_setting('ollama_url')

def process_chunk(chunk, model, system_prompt=None, temperature=0.3, top_k=40, top_p=0.9, repeat_penalty=1.2, max_tokens=2048):
    logger.debug(f"Sending request to Ollama API. Chunk length: {len(chunk)}")

    payload = {
        "model": model,
        "prompt": f"Please edit this text, correcting any errors and improving its clarity and coherence:\n\n{chunk}",
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "num_predict": max_tokens
        }
    }

    if system_prompt:
        payload["system"] = system_prompt

    headers = {'Content-Type': 'application/json'}

    try:
        logger.debug("Sending POST request to Ollama API")
        response = requests.post(ollama_url, headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json()['response']
        logger.debug(f"Received response from Ollama API. Response length: {len(result)}")
        return f"<edited_text>{result}</edited_text>"
    except requests.RequestException as e:
        logger.error(f"Error in Ollama API call: {e}")
        return f"<edited_text>{chunk}</edited_text>"
