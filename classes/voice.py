import torch
import soundfile as sf
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Voice:
    def __init__(self, language='ru', speaker=None, device='cuda'):
        self.device = torch.device(device)
        self.sample_rate = 48000
        self.language = language
        self.speaker = speaker
        self.available_speakers = self.get_available_speakers()
        if self.speaker is None or self.speaker not in self.available_speakers:
            self.speaker = self.available_speakers[0]  # Выбор первого доступного спикера, если текущий не поддерживается
        self.model, self.utils = self.load_model()
        self.audio_file = None  # Инициализация переменной для хранения пути к файлу
        self.project_root = Path(__file__).parent.parent

    def load_model(self):
        print(f"Loading Silero TTS model for language {self.language} and speaker {self.speaker}...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-models',
            model='silero_tts',
            language=self.language,
            speaker=self.speaker
        )
        model.to(self.device)
        print("Model loaded successfully.")
        return model, utils

    def generate_audio(self, text):
        print(f"Generating audio for text: {text[:30]}...")
        audio = self.model.apply_tts(text=text, 
                                     speaker=self.speaker, 
                                     sample_rate=self.sample_rate)
        self.save_raw_audio(audio)
        return audio
    
    def save_raw_audio(self, audio):
        timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
        output_dir = self.project_root / "output" / "tts" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "output.wav"
        audio_numpy = audio.cpu().numpy()

        logger.debug(f"Shape of audio_numpy: {audio_numpy.shape}")
        logger.debug(f"Data type of audio_numpy: {audio_numpy.dtype}")
        logger.debug(f"Min and max values of audio_numpy: {np.min(audio_numpy)}, {np.max(audio_numpy)}")

        sf.write(str(output_path), audio_numpy, self.sample_rate)
        print(f"Raw audio saved to {output_path}")
        
        self.audio_file = str(output_path)  # Сохранение пути к файлу в переменную
    
    def get_audio_file(self):
        return self.audio_file

    def get_available_speakers(self):
        # Код для получения списка спикеров
#       return ['v4_ru', 'v3_1_ru', 'ru_v3', 'aidar_v2', 'aidar_8khz', 'aidar_16khz', 'baya_v2', 'baya_8khz', 'baya_16khz', 'irina_v2', 'irina_8khz', 'irina_16khz', 'kseniya_v2', 'kseniya_8khz', 'kseniya_16khz', 'natasha_v2', 'natasha_8khz', 'natasha_16khz', 'ruslan_v2', 'ruslan_8khz', 'ruslan_16khz', 'v3_en', 'v3_en_indic', 'lj_v2', 'lj_8khz', 'lj_16khz', 'v3_de', 'thorsten_v2', 'thorsten_8khz', 'thorsten_16khz', 'v3_es', 'tux_v2', 'tux_8khz', 'tux_16khz', 'v3_fr', 'gilles_v2', 'gilles_8khz', 'gilles_16khz', 'aigul_v2', 'v3_xal', 'erdni_v2', 'v3_tt', 'dilyara_v2', 'v4_uz', 'v3_uz', 'dilnavoz_v2', 'v4_ua', 'v3_ua', 'mykyta_v2', 'v4_indic', 'v3_indic', 'v4_cyrillic', 'multi_v2']

        return ['aidar', 'baya', 'kseniya', 'xenia', 'eugene', 'random']
