import gradio as gr
from modules.settings_processor import get_all_settings, update_settings, reset_settings
from modules.text2voice_processor import get_available_languages

def create_settings_interface():
    available_languages = get_available_languages()
    sample_rate_choices = [8000, 16000, 24000, 48000]

    def load_current_settings():
        current_settings = get_all_settings()
        return (
            current_settings['sample_rate'],
            current_settings['file_format'],
            current_settings['silence_duration'],
            current_settings['silence_threshold'],
            current_settings['lambd'],
            current_settings['tau'],
            current_settings['solver'],
            current_settings['nfe'],
            current_settings['whisper_model_language'],
            current_settings['whisper_model_size'],
            current_settings['whisper_language'],
            current_settings['silero_sample_rate'],
            current_settings['use_llm_for_ssml'],
            current_settings['tts_language'],
            current_settings['num_inference_steps'],
            current_settings['guidance_scale'],
            current_settings['num_images'],
            current_settings['width'],
            current_settings['height'],
            current_settings['image_format'],
        )

    with gr.Blocks() as settings_interface:
        gr.Markdown("# Settings")
        
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Audio Settings")
                    sample_rate = gr.Dropdown(label="Default Sample Rate", choices=[8000, 11025, 22050, 44100, 48000, 96000])
                    file_format = gr.Dropdown(label="Default Output Format", choices=['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a'])

            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Silence Removal Settings")
                    silence_duration = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="Default Silence Duration (seconds)")
                    silence_threshold = gr.Slider(minimum=-60, maximum=-30, step=1, label="Default Silence Threshold (dB)")

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Audio Enhancement and Denoising Settings")
                    lambd = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Default Denoise Strength (lambda)")
                    tau = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Default CFM Prior Temperature (tau)")
                    solver = gr.Dropdown(label="Default Numerical Solver", choices=["midpoint", "rk4", "euler"])
                    nfe = gr.Slider(minimum=0, maximum=128, step=1, label="Default Number of Function Evaluations (NFE)")

            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Whisper Transcription Settings")
                    whisper_model_language = gr.Radio(label="Model Language", choices=["multilingual", "english-only"], value="multilingual")
                    whisper_model_size = gr.Radio(label="Model Size", choices=["tiny", "base", "small", "medium", "large"], value="base")
                    whisper_language = gr.Radio(label="Transcription Language", choices=["original", "english"], value="original")

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Silero TTS Settings")
                    silero_sample_rate = gr.Dropdown(label="Sample Rate", choices=sample_rate_choices)
                    use_llm_for_ssml = gr.Checkbox(label="Generate SSML using LLM", value=False)
                    tts_language = gr.Dropdown(label="TTS Language", choices=available_languages)
            with gr.Column():
                with gr.Group():
                    gr.Markdown("## Text to Image Settings")
                    num_inference_steps = gr.Slider(minimum=1, maximum=100, step=1, value=28, label="Number of Inference Steps")
                    guidance_scale = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.0, label="Guidance Scale")
                    num_images = gr.Slider(minimum=1, maximum=9, step=1, value=1, label="Number of Images to Generate")
                    width = gr.Slider(minimum=256, maximum=2048, step=64, value=512, label="Image Width")
                    height = gr.Slider(minimum=256, maximum=2048, step=64, value=512, label="Image Height")
                    image_format = gr.Radio(["png", "jpg"], label="Image Format", value="png")
        with gr.Row():
            with gr.Group():
                gr.Markdown("## Ollama IP Settings")
                ollama_ip_key = gr.Textbox(label="Ollama IP key", placeholder="Enter ollama IP here")
        with gr.Row():
            save_button = gr.Button("Save Changes")
            reset_button = gr.Button("Reset to Default")
        
        result = gr.Textbox(label="Result")

        def save_changes(sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe, 
                         whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                         num_inference_steps, guidance_scale, num_images, width, height, image_format):
            new_settings = {
                'sample_rate': sample_rate,
                'file_format': file_format,
                'silence_duration': silence_duration,
                'silence_threshold': silence_threshold,
                'lambd': lambd,
                'tau': tau,
                'solver': solver,
                'nfe': nfe,
                'whisper_model_language': whisper_model_language,
                'whisper_model_size': whisper_model_size,
                'whisper_language': whisper_language,
                'silero_sample_rate': silero_sample_rate,
                'use_llm_for_ssml': use_llm_for_ssml,
                'tts_language': tts_language,
                'num_inference_steps': num_inference_steps,
                'guidance_scale': guidance_scale,
                'num_images': num_images,
                'width': width,
                'height': height,
                'image_format': image_format
            }
            update_settings(new_settings)
            return "Settings saved successfully!"

        def reset_to_default():
            new_settings = reset_settings()
            return (
                new_settings['sample_rate'],
                new_settings['file_format'],
                new_settings['silence_duration'],
                new_settings['silence_threshold'],
                new_settings['lambd'],
                new_settings['tau'],
                new_settings['solver'],
                new_settings['nfe'],
                new_settings['whisper_model_language'],
                new_settings['whisper_model_size'],
                new_settings['whisper_language'],
                new_settings['silero_sample_rate'],
                new_settings['use_llm_for_ssml'],
                new_settings['tts_language'],
                new_settings['num_inference_steps'],
                new_settings['guidance_scale'],
                new_settings['num_images'],
                new_settings['width'],
                new_settings['height'],
                new_settings['image_format'],
                "Settings reset to default values!"
            )

        save_button.click(
            save_changes,
            inputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                    whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                    num_inference_steps, guidance_scale, num_images, width, height, image_format],
            outputs=result
        )
        reset_button.click(
            reset_to_default,
            outputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                     whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                     num_inference_steps, guidance_scale, num_images, width, height, image_format, result]
        )

        # Загрузка текущих настроек при инициализации интерфейса
        settings_interface.load(
            load_current_settings,
            outputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                     whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                     num_inference_steps, guidance_scale, num_images, width, height, image_format]
        )

    return settings_interface
