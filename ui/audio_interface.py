# ui/audio_interface.py
import gradio as gr
from modules.audio_processor import process_audio
from modules.settings_processor import get_all_settings

def create_audio_interface():

    settings = get_all_settings()

    with gr.Blocks() as audio_interface:
        gr.Markdown("# Audio Processor")
        with gr.Row():
            with gr.Column():
                file_input = gr.Audio(label="Upload Audio")
                
                change_sample_rate = gr.Checkbox(label="Change Sample Rate", value=False)
                new_sample_rate = gr.Dropdown(label="New Sample Rate", choices=[8000, 11025, 22050, 44100, 48000, 96000], visible=False)
                
                to_mono = gr.Checkbox(label="Convert to Mono", value=False)
                
                apply_filter = gr.Checkbox(label="Apply Audio Filter", value=False)
                filter_choices = gr.Dropdown(label="Select Audio Filter", choices=[
                    "volume=2.0", "lowpass=f=1000", "highpass=f=1000", "atempo=1.5",
                    "areverse", "aphaser", "acompressor", "aecho=0.8:0.88:60:0.4"
                ], visible=False)
                
                remove_silence = gr.Checkbox(label="Remove Silence", value=False)
                silence_duration = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="Silence Duration (seconds)", visible=False)
                silence_threshold = gr.Slider(minimum=-60, maximum=-30, step=1, label="Silence Threshold (dB)", visible=False)
                
                audio_processing = gr.Radio(["None", "Denoise", "Enhance"], label="Audio Processing", value="None")
                lambd = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="Denoise Strength (lambda)", visible=False)
                tau = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label="CFM Prior Temperature (tau)", visible=False)
                solver = gr.Dropdown(["midpoint", "rk4", "euler"], label="Numerical Solver", visible=False)
                nfe = gr.Slider(minimum=0, maximum=128, step=1, label="Number of Function Evaluations (NFE)", visible=False)

                output_format = gr.Dropdown(label="Output Format", choices=['wav', 'mp3', 'ogg', 'flac', 'aac', 'm4a'])

                submit_button = gr.Button("Process Audio")
            
            with gr.Column():
                with gr.Tabs() as tabs:
                    with gr.TabItem("Processed Audio", id="processed_audio_audio_tab"):
                        output_audio = gr.Audio(label="Processed Audio", type="filepath")
                    with gr.TabItem("Processing Log", id="processing_log_audio_tab"):
                        output_text = gr.Textbox(label="Processing Log", lines=10)

        def update_visibility(change_rate, apply_filt, remove_sil, audio_proc):
            return {
                new_sample_rate: gr.update(visible=change_rate),
                filter_choices: gr.update(visible=apply_filt),
                silence_duration: gr.update(visible=remove_sil),
                silence_threshold: gr.update(visible=remove_sil),
                lambd: gr.update(visible=audio_proc != "None"),
                tau: gr.update(visible=audio_proc != "None"),
                solver: gr.update(visible=audio_proc != "None"),
                nfe: gr.update(visible=audio_proc != "None")
            }

        def load_current_settings():
            current_settings = get_all_settings()
            return (
                gr.update(value=current_settings['sample_rate']),
                gr.update(value=current_settings['silence_duration']),
                gr.update(value=current_settings['silence_threshold']),
                gr.update(value=current_settings['lambd']),
                gr.update(value=current_settings['tau']),
                gr.update(value=current_settings['solver']),
                gr.update(value=current_settings['nfe']),
                gr.update(value=current_settings['file_format'])
            )


        change_sample_rate.change(fn=update_visibility, inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing], 
                                  outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe])
        apply_filter.change(fn=update_visibility, inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing], 
                            outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe])
        remove_silence.change(fn=update_visibility, inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing], 
                              outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe])
        audio_processing.change(fn=update_visibility, inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing], 
                                outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe])

        submit_button.click(
            fn=process_audio,
            inputs=[file_input, change_sample_rate, new_sample_rate, to_mono, apply_filter, filter_choices,
                    remove_silence, silence_duration, silence_threshold, audio_processing, lambd, tau, solver, nfe, output_format],
            outputs=[output_audio, output_text]
        )

        # Загрузка текущих настроек при инициализации интерфейса и при каждом обновлении вкладки
        audio_interface.load(
            load_current_settings,
            outputs=[new_sample_rate, silence_duration, silence_threshold, lambd, tau, solver, nfe, output_format]
        )
        
        def update():
            return load_current_settings()

        audio_interface.update = update
    return audio_interface
