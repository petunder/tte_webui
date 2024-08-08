# modules/transcription_processor.py
from classes.audio import Audio
from classes.settings import Settings
from classes.text import Text  # Добавляем импорт Text, если он необходим для использования провайдера
import json

def transcribe_audio(audio_input, model_language, model_size, language):
    #settings=Settings()
    audio = Audio(audio_input)
    settings = Settings()
    log = []

    def log_callback(msg):
        log.append(msg)
    
    # Предобработка аудио
    log.append("Changing sample rate to 22050 Hz...")
    audio.change_sample_rate(22050)
    
    log.append("Converting to mono...")
    audio.stereo_to_mono()
    
    #log.append("Applying volume filter...")
    #audio.apply_filter("volume=2.0")
    
    # Получаем настройки для улучшения аудио
    lambd = settings.get_setting('lambd')
    tau = settings.get_setting('tau')
    solver = settings.get_setting('solver')
    nfe = settings.get_setting('nfe')
    log.append("Denoising audio...")
    audio.denoise_audio(lambd, tau, solver, nfe, log_callback)
    
    # Получаем настройки для удаления тишины
    silence_duration = settings.get_setting('silence_duration')
    silence_threshold = settings.get_setting('silence_threshold')
    log.append("Removing silence...")
    audio.remove_silence(silence_duration, silence_threshold, log_callback)
    
    # Определяем провайдера для транскрипции
    provider = settings.get_setting('provider')
    log.append(f"Selected provider: {provider}")

    # Транскрипция
    if provider == 'together':
        log.append(f"Transcribing with Together (model: {model_language}.{model_size}, language: {language})...")
        # Используем Together API для транскрипции
        text, edited_text, timestamp_view, timestamp_table, json_output, json_raw = audio.transcribe(model_language, model_size, language)
    elif provider == 'ollama':
        log.append(f"Transcribing with Ollama (model: {model_language}.{model_size}, language: {language})...")
        # Используем Ollama API для транскрипции
        text, edited_text, timestamp_view, timestamp_table, json_output, json_raw = audio.transcribe(model_language, model_size, language)
    else:
        log.append("Error: Unknown provider selected.")
        return None, None, None, None, None, None, "\n".join(log)
    
    log.append("Transcription complete.")
    
    return text, edited_text, timestamp_view, timestamp_table, json.dumps(json_output, indent=2), json.dumps(json_raw, indent=2), "\n".join(log)
