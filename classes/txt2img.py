# txt2img.py
import torch
import gc
from diffusers import StableDiffusion3Pipeline
from datetime import datetime
import os

class Text2ImageProcessor:
    def __init__(self, model_path="v2ray/stable-diffusion-3-medium-diffusers"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self):
        if self.device == "cuda":
            pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, torch_dtype=torch.float32)
        return pipe.to(self.device)

    @staticmethod
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    def generate_image(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height,
                       num_images=1, image_format="png"):
        self.clear_memory()

        # Ensure width and height are multiples of 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        pipe = self._load_model()
        try:
            with torch.no_grad():
                output = pipe(
                    prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    num_images_per_prompt=num_images,
                )
                images = [img.copy() for img in output.images]  # Make copies of the images
        finally:
            # Aggressively clear CUDA memory
            if self.device == "cuda":
                for param in pipe.parameters():
                    param.data = param.data.to("cpu")
                    if param._grad is not None:
                        param._grad.data = param._grad.data.to("cpu")
            del pipe
            del output
            self.clear_memory()

        image_paths = []
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.makedirs("output", exist_ok=True)

        for idx, image in enumerate(images):
            filename = f"generated_image_{timestamp}_{idx}.{image_format}"
            output_path = os.path.join("output", filename)

            # Save the image in the specified format
            if image_format == "png":
                image.save(output_path, "PNG")
            else:
                image = image.convert("RGB")
                image.save(output_path, "JPEG", quality=95)

            image_paths.append(output_path)

        del images
        self.clear_memory()

        return image_paths