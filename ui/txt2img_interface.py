# text2image_interface.py
import gradio as gr
from modules.txt2img_processor import generate_images

def create_text2image_interface():
    with gr.Blocks() as text2image_tab:
        gr.Markdown("## Generate images from text using Stable Diffusion 3")

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
                negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter negative prompt (optional)")
                num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=28, label="Number of Inference Steps")
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

        def display_images(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, width, height,
                           image_format):
            image_paths = generate_images(prompt, negative_prompt, num_inference_steps, guidance_scale, num_images,
                                          width, height, image_format)
            return [(path, "") for path in image_paths]

        generate_btn.click(
            display_images,
            inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, num_images, width, height, image_format],
            outputs=gallery
        )

    return text2image_tab
