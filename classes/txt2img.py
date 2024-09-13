# classes/txt2img.py
import os
import subprocess
import shutil
from datetime import datetime
from classes.settings import Settings
import urllib.request


class Text2ImageProcessor:
    MODEL_URLS = {
        "SD3": {
            "model_file": "sd3_medium_incl_clips_t5xxlfp8.safetensors",
            "download_url": "https://huggingface.co/ckpt/stable-diffusion-3-medium/resolve/main/sd3_medium_incl_clips_t5xxlfp8.safetensors"
        },
        "Flux.1-DEV": {
            "model_file": "flux1-dev-Q8_0.gguf",
            "download_url": "https://huggingface.co/city96/FLUX.1-dev-gguf/resolve/main/flux1-dev-Q8_0.gguf",
            "additional_files": {
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
            }
        },
        "Flux.1-SCHNELL": {
            "model_file": "flux1-schnell-Q8_0.gguf",
            "download_url": "https://huggingface.co/city96/FLUX.1-schnell-gguf/resolve/main/flux1-schnell-Q8_0.gguf",
            "additional_files": {
                "clip_l.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors",
                "t5xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
                "ae.safetensors": "https://huggingface.co/your-repo/ae.safetensors"
            }
        }
    }

    def __init__(self, model_path=None, quantization='q8_0'):
        self.settings = Settings()
        self.provider = self.settings.get_setting('txt2img_provider')
        self.quantization = quantization

        if self.provider == "SD3":
            self.model_name = "stable-diffusion-3-medium"
            model_info = self.MODEL_URLS["SD3"]
        elif self.provider == "Flux.1-DEV":
            self.model_name = "FLUX.1-dev"
            model_info = self.MODEL_URLS["Flux.1-DEV"]
        elif self.provider == "Flux.1-SCHNELL":
            self.model_name = "FLUX.1-schnell"
            model_info = self.MODEL_URLS["Flux.1-SCHNELL"]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        self.models_dir = os.path.join(os.getcwd(), "models", self.model_name)
        os.makedirs(self.models_dir, exist_ok=True)

        # Основная модель
        self.model_file = model_info.get("model_file") if self.provider == "SD3" else model_info.get("model_file")
        self.model_path = model_path if model_path else os.path.join(self.models_dir, self.model_file)

        # Проверка и загрузка основной модели
        if not os.path.exists(self.model_path):
            self.download_model(model_info)

        # Дополнительные файлы для Flux
        self.additional_files = model_info.get("additional_files", {})
        for file_name, url in self.additional_files.items():
            file_path = os.path.join(self.models_dir, file_name)
            if not os.path.exists(file_path):
                self.download_file(url, file_path)

    def download_file(self, url, destination):
        print(f"Скачивание {url} в {destination}...")
        try:
            with urllib.request.urlopen(url) as response, open(destination, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print(f"Скачивание завершено: {destination}")
        except Exception as e:
            raise RuntimeError(f"Не удалось скачать файл {url}. Ошибка: {e}")

    def download_model(self, model_info):
        download_url = model_info.get("download_url")
        if not download_url:
            raise ValueError(f"No download URL provided for provider {self.provider}")
        self.download_file(download_url, self.model_path)

    def generate_image(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, num_images=1,
                       image_format="png"):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        image_paths = []
        for i in range(num_images):
            output_path = os.path.join(output_dir, f"generated_image_{timestamp}_{i}.{image_format}")
            command = self.build_command(prompt, negative_prompt, num_inference_steps, guidance_scale, width, height,
                                         output_path)

            try:
                print(f"Выполнение команды: {' '.join(command)}")
                subprocess.run(command, check=True)
                image_paths.append(output_path)
            except subprocess.CalledProcessError as e:
                print(f"Ошибка при генерации изображения {i + 1}: {e}")

        return image_paths

    def build_command(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, output_path):
        sd_executable = os.path.join(os.getcwd(), "bin", "sd")
        if not os.path.exists(sd_executable):
            raise FileNotFoundError(f"Исполняемый файл 'sd' не найден по пути: {sd_executable}")

        base_command = [sd_executable]

        if self.provider == "SD3":
            base_command += [
                "-m", self.model_path,
                "-p", prompt,
                "--cfg-scale", str(guidance_scale),
                "--steps", str(num_inference_steps),
                "--sampling-method", "euler",
                "-H", str(height),
                "-W", str(width),
                "--seed", str(self.settings.get_setting('seed') or -1),
                "-o", output_path,
                "--clip-on-cpu"
            ]
        elif self.provider in ["Flux.1-DEV", "Flux.1-SCHNELL"]:
            flux_command = [
                "--diffusion-model", self.model_path,
                "--clip_l", os.path.join(self.models_dir, "clip_l.safetensors"),
                "--t5xxl", os.path.join(self.models_dir, "t5xxl_fp16.safetensors"),
                "--vae", os.path.join(self.models_dir, "ae.safetensors"),  # Добавьте эту строку
                "-p", prompt,
                "--cfg-scale", str(guidance_scale),
                "--steps", str(num_inference_steps),
                "--sampling-method", "euler",
                "-H", str(height),
                "-W", str(width),
                "--seed", str(self.settings.get_setting('seed') or -1),
                "-o", output_path,
                "-v"
            ]
            base_command += flux_command
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        return base_command