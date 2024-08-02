# text2image_processor.py
from classes.txt2img import Text2ImageProcessor

def generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, width, height, image_format):
    processor = Text2ImageProcessor()
    image_paths = []
    for _ in range(num_images):
        image_path = processor.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            image_format=image_format
        )
        image_paths.append(image_path)
    return image_paths