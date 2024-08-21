# classes/audio.py
import subprocess
import os
from datetime import datetime
import soundfile as sf
import shutil
import numpy as np
import whisper
from classes.text import Text
import json
from llm.providers.together import process_chunk as together_process_chunk
from llm.providers.groq import improve_text as groq_process_chunk
from classes.settings import Settings
settings = Settings()
class Audio:
    def __init__(self, input_audio):
        self.temp_dir = "/tmp/resemble-enhance"
        self.temp_file = self._create_temp_file(input_audio)
        self.sample_rate = input_audio[0]
        self.channels = 1 if input_audio[1].ndim == 1 else 2
        self.duration = input_audio[1].shape[0] / self.sample_rate

    def _create_temp_file(self, input_audio):
        if isinstance(input_audio, tuple):
            rate, data = input_audio
            temp_dir = f"{self.temp_dir}/{datetime.now().strftime('%d%m%y_%H%M%S')}"
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, "input_audio.wav")
            sf.write(temp_file_path, data, rate)
            return temp_file_path
        else:
            print(f"Unknown file format: {str(input_audio)}")

    def change_sample_rate(self, new_sample_rate):
        output_file = f"{self.temp_file[:-4]}_{new_sample_rate}.wav"
        subprocess.run(["ffmpeg", "-i", self.temp_file, "-ar", str(new_sample_rate), output_file])
        self.temp_file = output_file
        self.sample_rate = new_sample_rate

    def stereo_to_mono(self):
        if self.channels == 2:
            output_file = f"{self.temp_file[:-4]}_mono.wav"
            subprocess.run(["ffmpeg", "-i", self.temp_file, "-ac", "1", output_file])
            self.temp_file = output_file
            self.channels = 1

    def apply_filter(self, filter_str):
        output_file = f"{self.temp_file[:-4]}_filtered.wav"
        subprocess.run(["ffmpeg", "-i", self.temp_file, "-af", filter_str, output_file])
        self.temp_file = output_file

    def remove_silence(self, silence_duration=1, silence_threshold=-50, output_callback=None):
        output_file = f"{self.temp_file[:-4]}_silenced.wav"
        result = self._remove_silence(self.temp_file, output_file, silence_duration, silence_threshold)
        if result != self.temp_file:
            self.temp_file = result
        if output_callback:
            output_callback("Silence removal completed.\n")

    def denoise_audio(self, lambd, tau, solver, nfe, output_callback):
        input_file = os.path.abspath(self.temp_file)
        output_callback(f"Input file: {input_file}\n")
        output_callback(f"File exists: {os.path.exists(input_file)}\n")
        output_callback(f"File size: {os.path.getsize(input_file)} bytes\n")

        # Создаем директорию для входного файла
        input_dir = os.path.join(os.path.dirname(input_file), "denoise_input")
        os.makedirs(input_dir, exist_ok=True)

        # Копируем входной файл в новую директорию
        input_file_copy = os.path.join(input_dir, os.path.basename(input_file))
        shutil.copy2(input_file, input_file_copy)
        output_callback(f"Copied input file to: {input_file_copy}\n")

        # Создаем директорию для выходного файла
        output_dir = os.path.join(os.path.dirname(input_file), "denoise_output")
        os.makedirs(output_dir, exist_ok=True)

        resemble_enhance_path = shutil.which("resemble-enhance")
        if resemble_enhance_path is None:
            resemble_enhance_path = settings.get_setting('resemble_enhance_path')

        output_callback(f"resemble-enhance path: {resemble_enhance_path}\n")

        if not resemble_enhance_path:
            output_callback("Error: resemble-enhance not found in system path or settings.\n")
            return

        command_denoise = [
            resemble_enhance_path,
            "--denoise_only",
            input_dir,
            output_dir,
            "--lambd", str(lambd),
            "--tau", str(tau),
            "--solver", solver,
            "--nfe", str(nfe)
        ]

        output_callback(f"Running command: {' '.join(command_denoise)}\n")

        try:
            output_callback("Starting denoise process...\n")
            result = subprocess.run(command_denoise, check=True, capture_output=True, text=True)
            output_callback(f"Command output:\n{result.stdout}\n")
            output_callback(f"Command errors:\n{result.stderr}\n")
            
            # Находим выходной файл
            output_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
            if output_files:
                output_file = os.path.join(output_dir, output_files[0])
                output_callback(f"Output file: {output_file}\n")
                output_callback(f"Output file size: {os.path.getsize(output_file)} bytes\n")
                if os.path.getsize(output_file) > 0:
                    self.temp_file = output_file
                    output_callback("Denoise process completed successfully.\n")
                else:
                    output_callback("Error: Output file is empty.\n")
            else:
                output_callback("Error: No output file was created.\n")
        except subprocess.CalledProcessError as e:
            output_callback(f"Error during denoise process: {e.stderr}\n")
        except Exception as e:
            output_callback(f"Unexpected error: {str(e)}\n")

        # Очистка временных директорий
        shutil.rmtree(input_dir)
        if not os.listdir(output_dir):
            shutil.rmtree(output_dir)
            
    def enhance_audio(self, lambd, tau, solver, nfe, output_callback):
        input_file = os.path.abspath(self.temp_file)
        output_callback(f"Input file: {input_file}\n")
        output_callback(f"File exists: {os.path.exists(input_file)}\n")
        output_callback(f"File size: {os.path.getsize(input_file)} bytes\n")

        # Создаем директорию для входного файла
        input_dir = os.path.join(os.path.dirname(input_file), "enhance_input")
        os.makedirs(input_dir, exist_ok=True)

        # Копируем входной файл в новую директорию
        input_file_copy = os.path.join(input_dir, os.path.basename(input_file))
        shutil.copy2(input_file, input_file_copy)
        output_callback(f"Copied input file to: {input_file_copy}\n")

        # Создаем директорию для выходного файла
        output_dir = os.path.join(os.path.dirname(input_file), "enhance_output")
        os.makedirs(output_dir, exist_ok=True)

        resemble_enhance_path = shutil.which("resemble-enhance")
        output_callback(f"resemble-enhance path from shutil: {shutil.which(\"resemble-enhance\")}\n")
        if resemble_enhance_path is None:
            resemble_enhance_path = settings.get_setting('resemble_enhance_path')
            output_callback(f"resemble-enhance path from settings: {settings.get_setting('resemble_enhance_path')}\n")

        output_callback(f"resemble-enhance path: {resemble_enhance_path}\n")

        if not resemble_enhance_path:
            output_callback("Error: resemble-enhance not found in system path or settings.\n")
            return

        command_enhance = [
            resemble_enhance_path,
            input_dir,
            output_dir,
            "--lambd", str(lambd),
            "--tau", str(tau),
            "--solver", solver,
            "--nfe", str(nfe)
        ]

        output_callback(f"Running command: {' '.join(command_enhance)}\n")

        try:
            output_callback("Starting enhance process...\n")
            result = subprocess.run(command_enhance, check=True, capture_output=True, text=True)
            output_callback(f"Command output:\n{result.stdout}\n")
            output_callback(f"Command errors:\n{result.stderr}\n")
            
            # Находим выходной файл
            output_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
            if output_files:
                output_file = os.path.join(output_dir, output_files[0])
                output_callback(f"Output file: {output_file}\n")
                output_callback(f"Output file size: {os.path.getsize(output_file)} bytes\n")
                if os.path.getsize(output_file) > 0:
                    self.temp_file = output_file
                    output_callback("Enhance process completed successfully.\n")
                else:
                    output_callback("Error: Output file is empty.\n")
            else:
                output_callback("Error: No output file was created.\n")
        except subprocess.CalledProcessError as e:
            output_callback(f"Error during enhance process: {e.stderr}\n")
        except Exception as e:
            output_callback(f"Unexpected error: {str(e)}\n")

        # Очистка временных директорий
        try:
            if os.path.exists(input_dir):
                shutil.rmtree(input_dir)
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                shutil.rmtree(output_dir)
        except Exception as e:
            output_callback(f"Error during cleanup: {str(e)}\n")
            
            
    def _remove_silence(self, input_file, output_file, silence_duration=1, silence_threshold=-50):
        if not os.path.exists(input_file):
            print("Ошибка: входной файл не существует")
            return input_file
        if os.path.getsize(input_file) == 0:
            print("Ошибка: входной файл пустой")
            return input_file

        output_dir = os.path.dirname(output_file)
        if not os.access(output_dir, os.W_OK):
            print(f"Ошибка: нет прав на запись в директорию {output_dir}")
            return input_file

        try:
            if isinstance(silence_threshold, str):
                threshold_db = float(silence_threshold.replace('dB', '').strip())
            else:
                threshold_db = float(silence_threshold)
            threshold_amplitude = 10 ** (threshold_db / 20)
        except ValueError:
            print(f"Ошибка: неверный формат порога тишины: {silence_threshold}")
            return input_file

        command = [
            'ffmpeg',
            '-i', input_file,
            '-af', f'silenceremove=stop_periods=-1:stop_duration={silence_duration}:stop_threshold={threshold_amplitude}',
            '-c:a', 'pcm_s16le',
            output_file
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"FFmpeg stdout: {result.stdout}")
            print(f"FFmpeg stderr: {result.stderr}")

            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"Тишина удалена, файл сохранен как {output_file}")
                return output_file
            else:
                print("Ошибка: выходной файл не создан или пустой")
                return input_file

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return input_file


    def _run_process(self, command, output_callback):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            output_callback(line)
        process.stdout.close()
        process.wait()
        
    def get_audio_data(self):
        data, rate = sf.read(self.temp_file)
        return data, rate
    
    def transcribe(self, model_language, model_size, language):
        model_name = f"{model_language}.{model_size}" if model_language == "english-only" else model_size
        model = whisper.load_model(model_name)
        
        transcribe_options = {}
        if language != "original":
            transcribe_options["task"] = "translate"
        
        result = model.transcribe(self.temp_file, **transcribe_options)
        text = result["text"]
        PROVIDER  = settings.get_setting('provider')
        if PROVIDER == "ollama":
            LLM_MODEL = settings.get_setting('ollama_model')
            LLM_SYSTEM_PROMPT = """You are an experienced editor tasked with improving a given text. Your goal is to correct errors and enhance readability while staying close to the original text and preserving its initial meaning.
            Follow these steps to edit the text:
            Carefully read through the text and identify any grammatical, spelling, or punctuation errors. Correct these errors while maintaining the original word choice as much as possible.

            Improve the readability of the text by:
            a. Breaking up overly long sentences into shorter, clearer ones.
            b. Rearranging words or phrases for better flow and clarity.
            c. Replacing unclear or awkward phrasing with more natural alternatives.

            d. Ensuring consistent tense usage throughout the text.
            While making these improvements, be careful to preserve the original meaning and style of the text. Do not add new information or change the author's intent.
            If you encounter any specialized terminology or proper nouns, assume they are correct unless there is an obvious spelling error.
            After editing, review your changes to ensure they enhance the text without altering its core message or tone.

            Provide your edited version of the text within <edited_text> tags. Do not include any explanations, comments, or lists of changes made.
            Remember, your goal is to improve the text while keeping it as close to the original as possible. Make only necessary changes to correct errors and enhance readability.
            IMPORTANT: Always respond in the language of the original text. Do not translate or switch to any other language under any circumstances.
            Here is the original text you will be working with:

            """
            LLM_CHUNK_SIZE = 600
            #text = result["text"]
            text_processor = Text()
            edited_text = text_processor.enhance_text(
                text,
                LLM_MODEL,
                LLM_SYSTEM_PROMPT
        )
        elif PROVIDER == "together":
            os.environ['TOGETHER_API_KEY'] = settings.get_setting('together_api_key')
            model=settings.get_setting('togetherai_model')
            # Вызовем TogetherAI API для обработки текста
            edited_text = together_process_chunk(
                text)
        elif PROVIDER == "groq":
            os.environ['GROQ_API_KEY'] = settings.get_setting('groq_api_key')
            model_name = settings.get_setting('groq_model')
            edited_text = groq_process_chunk(text)
        else:
            # Если указанный провайдер не поддерживается, вернуть оригинальный текст
            print(f"Provider '{PROVIDER}' is not supported. Returning original text.")
            edited_text = text
        # Дополнительные обработки для получения других форматов вывода
        timestamp_view = self.whisper_to_timestamp_view(result)
        timestamp_table = self.whisper_to_timestamp_table(result)
        json_output = self.whisper_to_json(result)
        json_raw = self.whisper_to_json_raw(result)
        
        return text, edited_text, timestamp_view, timestamp_table, json_output, json_raw

    def whisper_to_timestamp_view(self, whisper_output):
        output = [
            "| Start | End | Text |",
            "|----|----|----|"
        ]
        for segment in whisper_output['segments']:
            output.append(
                f"| {segment['start']:.2f} | {segment['end']:.2f} | {segment['text']} |")
        return "\n".join(output)

    def whisper_to_timestamp_table(self, whisper_output):
        return [[round(segment['start'], 2), round(segment['end'], 2), segment['text']]
                for segment in whisper_output['segments']]

    def whisper_to_json(self, whisper_output):
        output = []
        for segment in whisper_output['segments']:
            output.append({
                'start': round(segment['start'], 2),
                'end': round(segment['end'], 2),
                'text': segment['text'],
            })
        return output

    def whisper_to_json_raw(self, whisper_output):
        output = []
        for segment in whisper_output['segments']:
            output.append({
                'id': segment['id'],
                'start': round(segment['start'], 2),
                'end': round(segment['end'], 2),
                'text': segment['text'],
                'seek': segment['seek'],
                'temperature': segment['temperature'],
                'avg_logprob': segment['avg_logprob'],
                'compression_ratio': segment['compression_ratio'],
                'no_speech_prob': segment['no_speech_prob'],
            })
        return output

    def get_file_path(self, output_format='wav'):
        supported_formats = ['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a']
        
        if output_format not in supported_formats:
            raise ValueError(f"Unsupported format: {output_format}. Supported formats are: {', '.join(supported_formats)}")

        if os.path.splitext(self.temp_file)[1][1:].lower() == output_format.lower():
            return self.temp_file

        output_file = f"{os.path.splitext(self.temp_file)[0]}.{output_format}"

        ffmpeg_command = [
            'ffmpeg',
            '-i', self.temp_file,
            '-y'  # Overwrite output file if it exists
        ]

        if output_format == 'mp3':
            ffmpeg_command.extend(['-acodec', 'libmp3lame', '-b:a', '192k'])
        elif output_format == 'ogg':
            ffmpeg_command.extend(['-acodec', 'libvorbis', '-q:a', '4'])
        elif output_format == 'flac':
            ffmpeg_command.extend(['-acodec', 'flac', '-compression_level', '5'])
        elif output_format == 'aac':
            ffmpeg_command.extend(['-acodec', 'aac', '-b:a', '192k'])
        elif output_format == 'm4a':
            ffmpeg_command.extend(['-acodec', 'aac', '-b:a', '192k', '-f', 'mp4'])

        ffmpeg_command.append(output_file)

        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return output_file
            else:
                print(f"Error: Output file {output_file} was not created or is empty.")
                return self.temp_file
        except subprocess.CalledProcessError as e:
            print(f"Error during file conversion: {e.stderr}")
            return self.temp_file


#    def __del__(self):
#        if os.path.exists(self.temp_file):
#            os.remove(self.temp_file)
#        shutil.rmtree(os.path.dirname(self.temp_file))
#        print(f"Temporary directory and file {self.temp_file} have been deleted.")
