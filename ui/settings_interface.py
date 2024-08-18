import gradio as gr
from modules.settings_processor import get_all_settings, update_settings, reset_settings
from modules.text2voice_processor import get_available_languages
from classes.settings import Settings

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
            current_settings['provider'],
            current_settings['ollama_model'],
            current_settings['ollama_url'],
            current_settings['togetherai_model'],
            current_settings['together_api_key'],
            current_settings['groq_model'],
            current_settings['groq_api_key']
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
                gr.Markdown("## Provider Settings")
                provider = gr.Radio(label="#Provider", choices=["ollama", "together", "groq"])

            # Ollama settings (shown only when "ollama" is selected)
                settings=Settings()
                with gr.Column():
                    with gr.Group():
                        if settings.get_setting('provider')=="ollama":
                            
                            ollama_model = gr.Textbox(label="Ollama model", placeholder="Enter ollama model name here", visible=True)
                            ollama_url = gr.Textbox(label="Ollama URL", placeholder="Enter ollama URL here", visible=True)
                        elif settings.get_setting('provider')=="together" :
                            ollama_model = gr.Textbox(label="Ollama model", placeholder="Enter ollama model name here", visible=False)
                            ollama_url = gr.Textbox(label="Ollama URL", placeholder="Enter ollama URL here", visible=False)
                        elif settings.get_setting('provider')=="groq" :
                            ollama_model = gr.Textbox(label="Ollama model", placeholder="Enter ollama model name here", visible=False)
                            ollama_url = gr.Textbox(label="Ollama URL", placeholder="Enter ollama URL here", visible=False)


            # TogetherAI settings (shown only when "together" is selected)
                with gr.Column():
                    with gr.Group():
                        if settings.get_setting('provider')=="together":
                            
                            togetherai_model = gr.Textbox(label="TogetherAI Model", placeholder="Enter model name", value="llama 3.1 70b", visible=True)
                            together_api_key = gr.Textbox(label="TogetherAI API Key", placeholder="Enter your API key", value="", visible=True)
                        elif settings.get_setting('provider')=="ollama":
                            togetherai_model = gr.Textbox(label="TogetherAI Model", placeholder="Enter model name", value="llama 3.1 70b", visible=False)
                            together_api_key = gr.Textbox(label="TogetherAI API Key", placeholder="Enter your API key", value="", visible=False)
                        elif settings.get_setting('provider')=="groq":
                            togetherai_model = gr.Textbox(label="TogetherAI Model", placeholder="Enter model name", value="llama 3.1 70b", visible=False)
                            together_api_key = gr.Textbox(label="TogetherAI API Key", placeholder="Enter your API key", value="", visible=False)

                with gr.Column():
                    with gr.Group():
                        if settings.get_setting('provider')=="groq":
                            
                            groq_model = gr.Textbox(label="GroqAI Model", placeholder="Enter model name", value="llama3-8b-8192", visible=True)
                            groq_api_key = gr.Textbox(label="GroqAI API Key", placeholder="Enter your API key", value="", visible=True)
                        elif settings.get_setting('provider')=="ollama":
                            groq_model = gr.Textbox(label="GroqAI Model", placeholder="Enter model name", value="llama3-8b-8192", visible=False)
                            groq_api_key = gr.Textbox(label="GroqAI API Key", placeholder="Enter your API key", value="", visible=False)
                        elif settings.get_setting('provider')=="groq":
                            groq_model = gr.Textbox(label="GroqAI Model", placeholder="Enter model name", value="llama3-8b-8192", visible=False)
                            groq_api_key = gr.Textbox(label="GroqAI API Key", placeholder="Enter your API key", value="", visible=False)


                            
        # Function to update the visibility of provider-specific settings
        def toggle_provider_settings(provider_choice):
            if provider_choice == "ollama":
                return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            elif provider_choice == "together":
                return gr.update(visible=False), gr.update(visible=False),  gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
            elif provider_choice == "groq":
                return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.update(visible=False),gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),gr.update(visible=False)
        
        provider.change(
            fn=toggle_provider_settings,
            inputs=provider,
            outputs=[ollama_model, ollama_url, togetherai_model, together_api_key, groq_model, groq_api_key]
        )
        
        with gr.Row():
            save_button = gr.Button("Save Changes")
            reset_button = gr.Button("Reset to Default")
        
        result = gr.Textbox(label="Result")

        def save_changes(sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe, 
                         whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                         num_inference_steps, guidance_scale, num_images, width, height, image_format, provider, ollama_model, ollama_url, togetherai_model, together_api_key, groq_model, groq_api_key):
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
                'image_format': image_format,
                'provider': provider,
                'ollama_model': ollama_model,
                'ollama_url': ollama_url,
                'togetherai_model': togetherai_model,
                'together_api_key': together_api_key,
                'groq_model': groq_model,
                'groq_api_key': groq_api_key
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
                new_settings['provider'], 
                new_settings['ollama_model'],
                new_settings['ollama_url'],
                new_settings['togetherai_model'],
                new_settings['together_api_key'],
                new_settings['groq_model'],
                new_settings['groq_api_key'],
                "Settings reset to default values!"
            )

        save_button.click(
            save_changes,
            inputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                    whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                    num_inference_steps, guidance_scale, num_images, width, height, image_format, provider, ollama_model, ollama_url, togetherai_model, together_api_key, groq_model, groq_api_key],
            outputs=result
        )
        
        reset_button.click(
            reset_to_default,
            outputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                     whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                     num_inference_steps, guidance_scale, num_images, width, height, image_format, provider, ollama_model, ollama_url, togetherai_model, together_api_key, groq_model, groq_api_key, result]
        )

        # Load current settings on interface initialization
        settings_interface.load(
            load_current_settings,
            outputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                     whisper_model_language, whisper_model_size, whisper_language, silero_sample_rate, use_llm_for_ssml, tts_language,
                     num_inference_steps, guidance_scale, num_images, width, height, image_format, provider, ollama_model, ollama_url, togetherai_model, together_api_key, groq_model, groq_api_key]
        )

    return settings_interface
