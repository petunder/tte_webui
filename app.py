import gradio as gr
from ui.audio_interface import create_combined_interface
from ui.settings_interface import create_settings_interface
from ui.txt2img_interface import create_text2image_interface  # Добавлен новый импорт для text-to-image

def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Audio and Image Processing App")

        with gr.Tabs() as tabs:
            with gr.TabItem("Audio Processing", id="audio_tab"):
                audio_interface = create_combined_interface()

            with gr.TabItem("Text to Image", id="text2image_tab"):  # Добавлена новая вкладка
                text2image_interface = create_text2image_interface()
            
            with gr.TabItem("Settings", id="settings_tab"):
                settings_interface = create_settings_interface()

        def on_tab_select(tab: gr.SelectData):
            if tab.value == "Audio Processing":
                return audio_interface.update()
            elif tab.value == "Text to Image":  # Обработка выбора вкладки text-to-image
                return text2image_interface.update()
            return None

        tabs.select(on_tab_select, None, None)

        demo.launch(share=False, server_name="0.0.0.0", server_port=7861)

if __name__ == "__main__":
    main()
