# classes/settings.py
import json
import os

class Settings:
    def __init__(self):
        self.settings_file = 'settings.json'
        self.default_settings = {
            'sample_rate': 44100,
            'file_format': 'wav',
            'silence_duration': 0.4,
            'silence_threshold': -40,
            'lambd': 0.8,
            'tau': 0.5,
            'solver': 'euler',
            'nfe': 64,
            'whisper_model_language': 'multilingual',
            'whisper_model_size': 'base',
            'whisper_language': 'original',
            'silero_sample_rate': 24000,
            'use_llm_for_ssml': False,
            'tts_language': 'en',
            'num_inference_steps': 28,
            'guidance_scale': 7,
            'num_images': 1,
            'width': 1024,
            'height': 1024,
            'image_format': 'png',
            'provider': 'ollama',  # Новый параметр выбора провайдера
            'ollama_model':'aya:35b-23-q8_0',
            'ollama_url': 'http://192.168.1.70:11434/api/generate',
            'togetherai_model': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',  # Модель для TogetherAI
            'together_api_key': '',
            'groq_model':'llama3-8b-8192',
            'groq_api_key': '',
            'openAI_model': 'gpt-4o',
            'openAI_api_key': '',
            'transcription_provider': 'ollama',
            'resemble_enhance_path': ''
        }
        self.load_settings()

    def load_settings(self):
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'r') as f:
                loaded_settings = json.load(f)
                self.settings = {**self.default_settings, **loaded_settings}
        else:
            self.settings = self.default_settings.copy()

    def save_settings(self):
        with open(self.settings_file, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def get_setting(self, key):
        return self.settings.get(key, self.default_settings.get(key))

    def update_setting(self, key, value):
        self.settings[key] = value

    def reset_to_default(self):
        self.settings = self.default_settings.copy()
