# SignSpeak User Guide

Welcome to SignSpeak, a real-time sign language translator that bridges communication between Deaf/hard-of-hearing individuals and hearing people.

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [Using Sign-to-Text](#using-sign-to-text)
4. [Using Text-to-Sign](#using-text-to-sign)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam for sign recognition
- Microphone for speech input (optional)
- Internet connection for LLM-based translation (optional, only if using OpenAI)

### Step-by-Step Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YourUsername/SignSpeak.git
   cd SignSpeak
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Set up API keys (for LLM-based translation):
   ```bash
   cp .env.example .env
   ```
   
   Then edit the `.env` file to add your API keys.

## Getting Started

There are two main ways to run SignSpeak:

### Desktop Application

The desktop application provides a full-featured experience with webcam integration and sign recognition:

```bash
python src/app.py
```

### Web Interface

The Streamlit web interface offers a more user-friendly experience:

```bash
streamlit run src/streamlit_app.py
```

This will open a web browser with the SignSpeak interface.

## Using Sign-to-Text

Sign-to-Text allows you to communicate using sign language and have it translated to text.

1. Start the application using one of the methods above.
2. Choose "Sign to Text" mode.
3. Click "Start Webcam" to begin capturing video.
4. Position yourself so that your upper body and hands are clearly visible.
5. Perform signs at a moderate pace.
6. The recognized signs and their translations will appear on the screen.
7. Click "Stop" to end the session.

### Tips for Better Recognition

- Ensure good lighting conditions.
- Wear clothing that contrasts with your skin color.
- Make clear, deliberate gestures.
- Face the camera directly.
- Keep a consistent distance from the camera.

## Using Text-to-Sign

Text-to-Sign allows you to enter text and have it translated to sign language.

1. Start the application using one of the methods above.
2. Choose "Text to Sign" mode.
3. Type your message in the text input box.
4. Click "Generate Sign Language."
5. The system will display the signs that correspond to your message.

In the current version, sign language is displayed as text descriptions. Future versions will include animated avatars.

## Configuration

### Environment Variables

SignSpeak uses environment variables for configuration. Create or edit the `.env` file in the project root to set these options:

- `OPENAI_API_KEY`: Your OpenAI API key (for advanced translation)
- `WEBCAM_INDEX`: The index of the webcam to use (default: 0)
- `USE_GPU`: Whether to use GPU acceleration (default: true)
- `DEBUG_MODE`: Enable debug mode for more verbose logging (default: false)

### Changing Models

To use a different sign recognition model:

1. Place your trained model file in the `src/models/` directory.
2. Update the `SIGN_RECOGNITION_MODEL_PATH` variable in your `.env` file.

## Troubleshooting

### Camera Access Issues

If the application cannot access your webcam:

1. Ensure that your webcam is properly connected.
2. Check if other applications are using the webcam.
3. Grant camera permissions to the application.
4. Try changing the `WEBCAM_INDEX` in the `.env` file if you have multiple cameras.

### Poor Recognition Accuracy

If sign recognition is not working well:

1. Ensure good lighting conditions.
2. Check that your hands and face are clearly visible.
3. Make more deliberate signs with clear pauses between them.
4. Try adjusting your position relative to the camera.

### Application Crashes

If the application crashes:

1. Check the logs for error messages.
2. Ensure all dependencies are correctly installed.
3. Verify that you have the necessary hardware resources (RAM, CPU).
4. If using GPU acceleration, ensure your GPU drivers are up to date.

## FAQ

### Does SignSpeak work with all sign languages?

The initial version is designed for American Sign Language (ASL). Support for other sign languages may be added in future versions.

### Do I need an internet connection?

SignSpeak can work offline with basic functionality, but internet access is required for LLM-based translation (using OpenAI API) and for accessing certain models.

### How many signs can SignSpeak recognize?

The current version has a limited vocabulary of common signs. Future updates will expand the sign vocabulary.

### Can I teach SignSpeak new signs?

This feature is planned for future releases but is not available in the current version.

### Is my data private?

When using local translation (rule-based or local models), all processing happens on your device. If you use OpenAI translation, your sign data will be sent to OpenAI's servers according to their privacy policy. 