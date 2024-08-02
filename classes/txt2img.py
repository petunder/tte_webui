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
        self.pipe = None

    def _load_model(self):
        if self.pipe is None:
            if self.device == "cuda":
                self.pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, torch_dtype=torch.float16)
            else:
                self.pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path)
            self.pipe = self.pipe.to(self.device)
        return self.pipe

    def generate_image(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height,
                       image_format="png"):
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Ensure width and height are multiples of 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        pipe = self._load_model()
        with torch.no_grad():
            output = pipe(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            )
            images = output.images

        # Clear resources
        if self.device == "cuda":
            pipe = pipe.to("cpu")
            del pipe
            del output
            gc.collect()
            torch.cuda.empty_cache()

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

        return image_paths
