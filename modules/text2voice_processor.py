# modules/text2voice_processor.py
from classes.text import Text
from classes.voice import Voice
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Text2VoiceProcessor:
    def __init__(self):
        self.text_processor = Text()
        self.voice_processor = None

    def get_available_speakers(self):
        return Voice().get_available_speakers()

    def process_text_to_voice(self, text, model, language, speaker):
        self.voice_processor = Voice(language=language, speaker=speaker)
        audio = self.voice_processor.generate_audio(text)
        audio_file = self.voice_processor.get_audio_file()
        return audio_file

def get_available_languages():
    # Здесь можно определить список доступных языков
    return ['ru', 'en', 'de', 'es', 'fr']
