import gradio as gr
from modules.audio_processor import process_audio
from modules.settings_processor import get_all_settings
from modules.transcription_processor import transcribe_audio
from classes.settings import Settings


def create_combined_interface():
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
            gr.update(value=current_settings['file_format']),
            gr.update(value=current_settings['whisper_model_language']),
            gr.update(value=current_settings['whisper_model_size']),
            gr.update(value=current_settings['whisper_language'])
        )

    def update_visibility(change_rate, apply_filt, remove_sil, audio_proc, transcription_selected):
        return {
            new_sample_rate: gr.update(visible=change_rate),
            filter_choices: gr.update(visible=apply_filt),
            silence_duration: gr.update(visible=remove_sil),
            silence_threshold: gr.update(visible=remove_sil),
            lambd: gr.update(visible=audio_proc != "None"),
            tau: gr.update(visible=audio_proc != "None"),
            solver: gr.update(visible=audio_proc != "None"),
            nfe: gr.update(visible=audio_proc != "None"),
            transcription_tabs: gr.update(visible=transcription_selected),
            model_language: gr.update(visible=transcription_selected),
            model_size: gr.update(visible=transcription_selected),
            language: gr.update(visible=transcription_selected)
        }
    
    def process_audio_only(*args):
        return process_audio(*args)
    def transcribe_only(audio, model_lang, model_size, lang):
        text, edited_text, timestamp_view, timestamp_table, json_output, json_raw, log = transcribe_audio(audio, model_lang, model_size, lang)
        return log, text, edited_text, timestamp_view, timestamp_table, json_output, json_raw

    def process_and_transcribe(audio, change_rate, new_rate, to_mono, apply_filt, filt_choice, remove_sil, sil_dur, sil_thresh, audio_proc, denoise_strength, cfm_temp, solver_choice, nfe_count, output_format, transcription_selected, model_lang, model_size, lang):
        processed_audio, log = process_audio(audio, change_rate, new_rate, to_mono, apply_filt, filt_choice, remove_sil, sil_dur, sil_thresh, audio_proc, denoise_strength, cfm_temp, solver_choice, nfe_count, output_format)
        
        #if transcription_selected:
        transcription_data = transcribe_audio(audio, model_lang, model_size, lang)
        return processed_audio, log, *transcription_data
        #else:
         #   return processed_audio, log, "", "", "", "", {}, {}, ""
    
    with gr.Blocks() as combined_interface:
        gr.Markdown("# Audio Processor & Transcription")
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
                
                transcription_selected = gr.Checkbox(label="Transcribe Audio", value=False)
                model_language = gr.Radio(label="Model Language", choices=["multilingual", "english-only"], visible=False)
                model_size = gr.Radio(label="Model Size", choices=["tiny", "base", "small", "medium", "large"], visible=False)
                language = gr.Radio(label="Transcription Language", choices=["original", "english"], visible=False)
                                
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
                    with gr.TabItem("Transcription Results", id="transcription_tabs", visible=True):
                        with gr.Tabs() as transcription_tabs:
                            with gr.TabItem("Edited Transcription", id="edited_transcription_tab"):
                                edited_transcription_output = gr.Textbox(label="Edited Transcription", lines=10)
                            with gr.TabItem("Raw Transcription", id="transcription_tab"):
                                transcription_output = gr.Textbox(label="Transcription", lines=10)
                            with gr.TabItem("Timestamp View", id="timestamp_view_tab"):
                                timestamp_view = gr.Textbox(label="Timestamp View", lines=10)
                            with gr.TabItem("Timestamp Table", id="timestamp_table_tab"):
                                timestamp_table = gr.Dataframe(label="Timestamp Table", headers=["Start", "End", "Text"])
                            with gr.TabItem("JSON Output", id="json_tab"):
                                json_output = gr.JSON(label="JSON Output")
                            with gr.TabItem("Raw JSON Output", id="raw_json_tab"):
                                json_raw_output = gr.JSON(label="Raw JSON Output")

        change_sample_rate.change(
            fn=update_visibility,
            inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing, transcription_selected], 
            outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe, transcription_tabs, model_language, model_size, language]
        )
        apply_filter.change(
            fn=update_visibility, 
            inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing, transcription_selected], 
            outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe, transcription_tabs, model_language, model_size, language]
        )
        remove_silence.change(
            fn=update_visibility, 
            inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing, transcription_selected], 
            outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe, transcription_tabs, model_language, model_size, language]
        )
        audio_processing.change(
            fn=update_visibility, 
            inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing, transcription_selected], 
            outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe, transcription_tabs, model_language, model_size, language]
        )
        transcription_selected.change(
            fn=update_visibility,
            inputs=[change_sample_rate, apply_filter, remove_silence, audio_processing, transcription_selected],
            outputs=[new_sample_rate, filter_choices, silence_duration, silence_threshold, lambd, tau, solver, nfe, transcription_tabs, model_language, model_size, language]
        )
        def handle_audio_processing(*args):
    # Распаковываем аргументы
            (audio_file, change_sample_rate, new_sample_rate, to_mono, apply_filter, filter_choice, 
            remove_silence, silence_duration, silence_threshold, audio_processing, lambd, tau, solver, 
            nfe, output_format, transcription_selected, model_language, model_size, language) = args

    # Лог для вывода сообщений об обработке
            log = []

    # Инициализируем выходные значения по умолчанию
            output_audio = None
            processing_log = ""
            transcription_output = ""
            edited_transcription_output = ""
            timestamp_view = ""
            timestamp_table = {}
            json_output = {}
            json_raw_output = {}

    # Обработка аудио
            if change_sample_rate or to_mono or apply_filter or audio_processing != "None" or remove_silence:
                output_audio, processing_log = process_audio(
                    audio_file, change_sample_rate, new_sample_rate, to_mono, apply_filter, filter_choice,
                    remove_silence, silence_duration, silence_threshold, audio_processing, lambd, tau, solver, nfe, output_format
                )
                log.append(processing_log)
    
    # Транскрипция
            if transcription_selected:
                transcription_output, edited_transcription_output, timestamp_view, timestamp_table, json_output, json_raw_output, transcription_log = transcribe_audio(
                    audio_file, model_language, model_size, language)
                log.append(transcription_log)
            
    
            return output_audio, "\n".join(log), transcription_output, edited_transcription_output, timestamp_view, timestamp_table, json_output, json_raw_output
        
# Вызов функции при нажатии кнопки
        
        submit_button.click(
            
            
            
            fn=handle_audio_processing,
            inputs=[file_input, change_sample_rate, new_sample_rate, to_mono, apply_filter, filter_choices, 
                remove_silence, silence_duration, silence_threshold, audio_processing, lambd, tau, solver, 
                nfe, output_format, transcription_selected, model_language, model_size, language],
            outputs=[output_audio, output_text, transcription_output, edited_transcription_output, 
                timestamp_view, timestamp_table, json_output, json_raw_output]
    )

        

        combined_interface.load(
            load_current_settings,
            outputs=[new_sample_rate, silence_duration, silence_threshold, lambd, tau, solver, nfe, output_format, model_language, model_size, language]
        )
        def update():
                return load_current_settings()

        combined_interface.update = update
        #combined_interface.update = load_current_settings

    return combined_interface

#if __name__ == "__main__":
 #   iface = create_combined_interface()
 #   iface.launch()
