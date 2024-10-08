# tte_webui
# TTE (Text to Everything) WebUI

TTE (Text to Everything) WebUI is a web-based interface for generating and processing text and audio. This project leverages modern web technologies and large language models (LLMs) to provide a user-friendly and powerful platform for text and audio processing tasks.

## Features

- **Audio to Text**: Convert audio files to text using advanced speech-to-text models.
- **Audio Enhancement**: Improve the quality of audio files with built-in enhancement features.
- **Modular Architecture**: Easily extend the functionalities by adding new modules.
- **Class-based Structure**: Organized code with clear class definitions for better maintainability.
- **LLM Integration**: Seamlessly integrate large language models for text generation tasks.
- **User Interface**: Intuitive and responsive UI built with modern web technologies.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/petunder/tte_webui.git
    cd tte_webui
    ```
2. Сheck if there is ffmpeg on your computer (otherwise it will be automatically installed)
   ```sh
    python ffmpeg_check.py
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   If there is a problem with the numpy library during installation, then install it manually:
   ```sh
   pip install numpy==1.26.0 --only-binary=:all:
   ```
   and update the following:
   ```sh
   pip install --upgrade wheel & pip install --upgrade build & pip install --upgrade cython
   ```   

4. Run the application:
    ```sh
    python app.py
    ```

### Directory Structure

- `classes/`: Contains class definitions used in the project.
- `llm/`: Includes modules for large language model integrations.
- `modules/`: Additional modules for extending functionality.
- `ui/`: Frontend components and static files for the web interface.

### Usage

1. Start the application by running `python app.py`.
2. Open your web browser and navigate to `http://localhost:7861`.
3. Interact with the UI to generate text-based outputs.
* In order to use TogetherAI you need to register on https://api.together.ai/ and save your personal API key

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue in the repository.

