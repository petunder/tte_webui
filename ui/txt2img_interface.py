# text2image_interface.py
import gradio as gr
from modules.txt2img_processor import generate_images
from modules.settings_processor import get_all_settings
from classes.settings import Settings


def create_text2image_interface():
    def load_current_settings():
        current_settings = get_all_settings()
        settings=Settings()
        if settings.get_setting('txt2img_provider')=="SD3":
            return (
                gr.update(value=current_settings.get('num_inference_steps_sd3')),
                gr.update(value=current_settings.get('guidance_scale_sd3')),
                gr.update(value=current_settings.get('num_images')),
                gr.update(value=current_settings.get('width')),
                gr.update(value=current_settings.get('height')),
                gr.update(value=current_settings.get('image_format'))
            )
        if settings.get_setting('txt2img_provider')=="Flux.1-DEV":
            return (
                gr.update(value=current_settings.get('num_inference_steps_flux1-dev')),
                gr.update(value=current_settings.get('guidance_scale_flux1-dev')),
                gr.update(value=current_settings.get('num_images')),
                gr.update(value=current_settings.get('width')),
                gr.update(value=current_settings.get('height')),
                gr.update(value=current_settings.get('image_format'))
            )
        if settings.get_setting('txt2img_provider')=="Flux.1-SCHNELL":
            return (
                gr.update(value=current_settings.get('num_inference_steps_flux1-schnell')),
                gr.update(value=current_settings.get('guidance_scale_flux1-schnell')),
                gr.update(value=current_settings.get('num_images')),
                gr.update(value=current_settings.get('width')),
                gr.update(value=current_settings.get('height')),
                gr.update(value=current_settings.get('image_format'))
            )

    with gr.Blocks() as text2image_tab:
        gr.Markdown("## Generate images from text")

        with gr.Row():
            with gr.Column():
                
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt (optional)")

                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=28,
                                                label="Number of Inference Steps")
                guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.0, label="Guidance Scale")
                num_images = gr.Slider(minimum=1, maximum=9, step=1, value=1, label="Number of Images to Generate")
                with gr.Row():
                    width = gr.Slider(minimum=256, maximum=2048, step=64, value=512, label="Image Width")
                    height = gr.Slider(minimum=256, maximum=2048, step=64, value=512, label="Image Height")

                image_format = gr.Radio(["png", "jpg"], label="Image Format", value="png")
                generate_btn = gr.Button("Generate Images")

            with gr.Column():
                gallery = gr.Gallery(
                    label="Generated Images",
                    show_label=True,
                    elem_id="gallery",
                    columns=3,
                    rows=3,
                    object_fit="contain",
                    height="auto"
                )
        
        

        def display_images( prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, width, height,
                           image_format):
            image_paths = generate_images( prompt, negative_prompt, num_inference_steps, guidance_scale, num_images,
                                          width, height, image_format)
            return image_paths  # Здесь просто возвращаем список путей к изображениям

        generate_btn.click(
            display_images,
            inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, width, height,
                    image_format],
            outputs=gallery
        )

        def update():
            return load_current_settings()

        text2image_tab.update = update

    text2image_tab.load(fn=load_current_settings,
                        outputs=[ num_inference_steps, guidance_scale, num_images, width, height, image_format])
    return text2image_tab
