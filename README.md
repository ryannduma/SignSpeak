# SignSpeak

A real-time sign language translator leveraging computer vision, neural networks, and large language models to bridge communication between Deaf/hard-of-hearing individuals and hearing people.

## Overview

SignSpeak is a bilingual communication platform that:
- **Recognizes** sign language from video input in real-time
- **Translates** signs into natural language text and speech
- **Converts** spoken/written language back into sign language via an animated avatar
- **Enables** seamless two-way communication between signing and non-signing users

## Features

- **Real-time video processing** using OpenCV and MediaPipe for skeletal tracking
- **Neural network-based sign recognition** for accurate gesture interpretation
- **Context-aware translation** via Large Language Models (LLMs)
- **Bidirectional communication:**
  - Sign language → Text/Speech
  - Speech/Text → Sign language (via animated avatar)
- **User-friendly interface** with camera feed and translation display

## Project Structure

```
SignSpeak/
├─ .gitignore              # Git ignore file
├─ LICENSE                 # Project license
├─ README.md               # This file
├─ requirements.txt        # Python dependencies
├─ .env.example            # Example environment variables
├─ docs/                   # Documentation
│  ├─ architecture.md      # System architecture
│  └─ user_guide.md        # Usage instructions
├─ notebooks/              # Jupyter notebooks
│  ├─ data_exploration.ipynb    # Data analysis
│  ├─ model_training.ipynb      # Training pipeline
│  └─ evaluation.ipynb          # Performance metrics
├─ src/                    # Source code
│  ├─ app.py               # Main application entry point
│  ├─ config.py            # Configuration settings
│  ├─ data/                # Data handling
│  │  ├─ __init__.py
│  │  ├─ dataset.py        # Dataset classes
│  │  └─ preprocessing.py  # Data preprocessing utilities
│  ├─ models/              # Model definitions and weights
│  │  ├─ __init__.py
│  │  ├─ sign_recognition.py  # Sign recognition model
│  │  └─ README.md         # Model information
│  ├─ sign_recognition/    # Sign language recognition
│  │  ├─ __init__.py
│  │  ├─ detector.py       # Pose detection
│  │  └─ classifier.py     # Sign classification
│  ├─ translation/         # Translation services
│  │  ├─ __init__.py
│  │  ├─ translator.py     # LLM-based translation
│  │  └─ speech.py         # TTS/ASR utilities
│  ├─ sign_generation/     # Sign language generation
│  │  ├─ __init__.py
│  │  ├─ generator.py      # Text-to-sign conversion
│  │  └─ avatar.py         # Avatar animation
│  └─ utils/               # Utility functions
│     ├─ __init__.py
│     └─ visualization.py  # Visualization helpers
└─ tests/                  # Unit tests
   ├─ __init__.py
   ├─ test_recognition.py
   └─ test_translation.py
```

## Getting Started

### Prerequisites

- Python 3.8+
- Webcam for sign language input
- Microphone for speech input (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YourUsername/SignSpeak.git
   cd SignSpeak
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Running SignSpeak

```bash
# Start the main application
python src/app.py

# Alternatively, run the Streamlit web interface
streamlit run src/streamlit_app.py
```

## Development Roadmap

- [x] Project setup and repository structure
- [ ] Data collection and preprocessing pipeline
- [ ] Sign recognition model development
  - [ ] Hand pose detection
  - [ ] Gesture classification
  - [ ] Continuous sign sequence recognition
- [ ] LLM integration for translation
- [ ] Speech-to-text and text-to-speech functionality
- [ ] Avatar-based sign language generation
- [ ] User interface development
- [ ] Performance optimization for real-time usage
- [ ] User testing and feedback integration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the terms of the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [OpenAI](https://openai.com/) for LLM capabilities
- Sign language datasets and resources that make this project possible
