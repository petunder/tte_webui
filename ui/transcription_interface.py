# ui/transcription_interface.py
import gradio as gr
from modules.transcription_processor import transcribe_audio
from modules.settings_processor import get_all_settings

def create_transcription_interface():
    def load_current_settings():
        current_settings = get_all_settings()
        return (
            current_settings['whisper_model_language'],
            current_settings['whisper_model_size'],
            current_settings['whisper_language']
        )

    with gr.Blocks() as transcription_interface:
        gr.Markdown("# Audio Transcription")
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(label="Upload Audio for Transcription")
                model_language = gr.Radio(label="Model Language", choices=["multilingual", "english-only"])
                model_size = gr.Radio(label="Model Size", choices=["tiny", "base", "small", "medium", "large"])
                language = gr.Radio(label="Transcription Language", choices=["original", "english"])
                transcribe_button = gr.Button("Transcribe Audio")
            
            with gr.Column(scale=2):
                with gr.Tabs() as tabs:
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
                    with gr.TabItem("Processing Log", id="transcribe_processing_log_tab"):
                        processing_log = gr.Textbox(label="Processing Log", lines=10)

        def process_and_transcribe(audio, model_lang, model_size, lang):
            text, edited_text, timestamp_view, timestamp_table, json_output, json_raw, log = transcribe_audio(audio, model_lang, model_size, lang)
            return log, text, edited_text, timestamp_view, timestamp_table, json_output, json_raw

        transcribe_button.click(
            fn=process_and_transcribe,
            inputs=[audio_input, model_language, model_size, language],
            outputs=[processing_log, transcription_output, edited_transcription_output, timestamp_view, timestamp_table, json_output, json_raw_output]
        )

        def update():
            return load_current_settings()

        transcription_interface.update = update
        
        # Загрузка текущих настроек при инициализации интерфейса
        transcription_interface.load(
            load_current_settings,
            outputs=[model_language, model_size, language]
        )

    return transcription_interface
