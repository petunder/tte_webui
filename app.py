# app.py
import gradio as gr
from ui.audio_interface import create_combined_interface
from ui.settings_interface import create_settings_interface
from ui.text2voice_interface import create_text2voice_interface  # Добавим новый импорт

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Audio Processing App")
        
        with gr.Tabs() as tabs:
            with gr.TabItem("Audio Processing", id="audio_tab"):
                audio_interface = create_combined_interface()
            
            # with gr.TabItem("Text to Voice", id="text2voice_tab"):  # Добавим новую вкладку
            #     text2voice_interface = create_text2voice_interface()
            with gr.TabItem("Settings", id="settings_tab"):
                settings_interface = create_settings_interface()

        def on_tab_select(tab: gr.SelectData):
            if tab.value == "Audio Processing":
                return audio_interface.update()
            # elif tab.value == "Text to Voice":
            #     return text2voice_interface.update()
            return None

        tabs.select(on_tab_select, None, None)

        demo.launch(share=False, server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main()
