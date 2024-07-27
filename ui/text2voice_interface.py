# ui/text2voice_interface.py
# ui/text2voice_interface.py
import gradio as gr
from modules.text2voice_processor import Text2VoiceProcessor

def create_text2voice_interface():
    processor = Text2VoiceProcessor()

    with gr.Blocks() as text2voice_interface:
        gr.Markdown("# Text to Voice")
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(label="Input Text", lines=10)
                model = gr.Dropdown(choices=["aya:35b-23-q8_0"], label="Model", value="aya:35b-23-q8_0")
                language = gr.Dropdown(choices=["ru", "en", "de", "es", "fr"], label="Language", value="ru")
                speaker = gr.Dropdown(choices=processor.get_available_speakers(), label="Speaker", value=processor.get_available_speakers()[0])
            
            with gr.Column(scale=1):
                generate_button = gr.Button("Generate Audio")
                audio_output = gr.Audio(label="Generated Audio", type="filepath")

        generate_button.click(
            fn=processor.process_text_to_voice,
            inputs=[input_text, model, language, speaker],
            outputs=[audio_output]
        )

    return text2voice_interface
