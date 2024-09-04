import torch
import gc
import os
from diffusers import StableDiffusion3Pipeline, FluxPipeline
from datetime import datetime
from classes.settings import Settings

class Text2ImageProcessor:
    def __init__(self, model_path=None):
        self.settings = Settings()
        self.provider = self.settings.get_setting('txt2img_provider')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.provider == "SD3":
            self.model_path = model_path if model_path else "v2ray/stable-diffusion-3-medium-diffusers"
        elif self.provider in ["Flux.1-DEV", "Flux.1-SCHNELL"]:
            self.model_path = model_path if model_path else "jellyhe/flux1-dev-fp8"  # Default path for Flux
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _load_model(self):
        if self.provider == "SD3":
            pipe = StableDiffusion3Pipeline.from_pretrained(self.model_path, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32)
        elif self.provider == "Flux.1-DEV":
            pipe = FluxPipeline.from_pretrained("jellyhe/flux1-dev-fp8", torch_dtype=torch.half)
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
        elif self.provider == "Flux.1-SCHNELL":
#            pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
            pipe = FluxPipeline.from_pretrained("drbaph/FLUX.1-schnell-dev-merged-fp8", torch_dtype=torch.half)
            if self.device == "cuda":
                pipe.enable_model_cpu_offload()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        if self.device == "cuda":
            pipe = pipe.to(self.device)
        return pipe

    @staticmethod
    def clear_memory():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    def _unload_model(self, pipe):
        if self.device == "cuda":
            for component in pipe.components.values():
                if hasattr(component, 'to'):
                    component.to('cpu')
                if hasattr(component, 'parameters'):
                    for param in component.parameters():
                        if param.data is not None:
                            param.data = param.data.to('cpu')
                        if param._grad is not None:
                            param._grad.data = param._grad.data.to('cpu')
        del pipe
        self.clear_memory()

    def generate_image(self, prompt, negative_prompt, num_inference_steps, guidance_scale, width, height, num_images=1, image_format="png"):
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
            self._unload_model(pipe)
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
