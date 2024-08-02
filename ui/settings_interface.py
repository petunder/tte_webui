# ui/settings_interface.py
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
            save_button = gr.Button("Save Changes")
            reset_button = gr.Button("Reset to Default")
        
        result = gr.Textbox(label="Result")

        def save_changes(sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe, 
                         whisper_model_language, whisper_model_size, whisper_language):
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
                'whisper_language': whisper_language
            }
            update_settings(new_settings)
            return "Settings saved successfully!"

        def reset_to_default():
            new_settings = reset_settings()
            available_languages = get_available_languages()
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
                "Settings reset to default values!"
            )

        save_button.click(
            save_changes,
            inputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                    whisper_model_language, whisper_model_size, whisper_language],
            outputs=result
        )
        reset_button.click(
            reset_to_default,
            outputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                     whisper_model_language, whisper_model_size, whisper_language, result]
        )

        # Загрузка текущих настроек при инициализации интерфейса
        settings_interface.load(
            load_current_settings,
            outputs=[sample_rate, file_format, silence_duration, silence_threshold, lambd, tau, solver, nfe,
                     whisper_model_language, whisper_model_size, whisper_language]
        )

    return settings_interface

