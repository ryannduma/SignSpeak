"""
SignSpeak Streamlit Web Interface

This module provides a web-based interface for the SignSpeak application
using Streamlit.
"""

import streamlit as st
import cv2
import numpy as np
import time
import sys
import os
import logging
from PIL import Image
from io import BytesIO

# Add src to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from sign_recognition.detector import SignDetector
from translation.translator import Translator
from sign_generation.generator import SignGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="SignSpeak - Sign Language Translator",
    page_icon="👐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("SignSpeak: Sign Language Translator")
st.markdown("""
    Welcome to SignSpeak! This application translates between sign language and text/speech.
    - **Sign to Text**: Use your webcam to sign, and the app will translate to text.
    - **Text to Sign**: Type or speak, and the app will generate sign language animations.
""")

# Sidebar with options
st.sidebar.title("Settings")
translation_mode = st.sidebar.radio(
    "Translation Mode",
    ["Sign to Text", "Text to Sign"]
)

# Function to process webcam frames for sign recognition
def process_webcam():
    # Initialize detector and translator
    detector = SignDetector()
    translator = Translator(model_type="dummy")
    
    # Placeholder for webcam feed
    webcam_placeholder = st.empty()
    
    # Placeholder for translation results
    translation_placeholder = st.empty()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Button to stop webcam
    stop_button = st.button("Stop Webcam")
    
    # Last detected sign and translation
    last_sign = None
    last_translation = None
    translation_display_time = 0
    
    while cap.isOpened() and not stop_button:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read from webcam.")
            break
        
        # Process the frame for sign detection
        frame, detected_keypoints = detector.process_frame(frame)
        
        # If keypoints are detected, try to recognize sign
        current_time = time.time()
        if detected_keypoints and (last_sign is None or current_time - translation_display_time > 2):
            recognized_sign = detector.recognize_sign(detected_keypoints)
            
            if recognized_sign and recognized_sign != last_sign:
                last_sign = recognized_sign
                
                # Translate recognized sign to text
                translated_text = translator.sign_to_text(recognized_sign)
                last_translation = translated_text
                translation_display_time = current_time
                
                logger.info(f"Recognized: '{recognized_sign}' -> '{translated_text}'")
                
                # Update translation display
                translation_placeholder.markdown(f"""
                    ### Recognized Sign: {last_sign}
                    ### Translation: {last_translation}
                """)
        
        # Convert frame from BGR to RGB (for Streamlit)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        webcam_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Check if stop button was pressed
        if stop_button:
            break
        
        # Brief pause to reduce CPU usage
        time.sleep(0.01)
    
    # Release webcam
    cap.release()
    
    # Clear placeholders
    webcam_placeholder.empty()
    translation_placeholder.empty()
    
    st.info("Webcam stopped.")

# Function to handle text-to-sign conversion
def text_to_sign():
    # Initialize translator and generator
    translator = Translator(model_type="dummy")
    generator = SignGenerator()
    
    # Text input
    input_text = st.text_area("Enter text to convert to sign language:", "Hello, how are you?")
    
    # Button to generate sign language
    if st.button("Generate Sign Language"):
        with st.spinner("Generating sign language..."):
            # Convert text to sign sequence
            signs = translator.text_to_sign(input_text)
            
            # If signs were found, generate animation
            if signs:
                st.success(f"Found {len(signs)} signs in the text.")
                
                # Generate animation
                animation_data = generator.generate_animation(signs)
                result = generator.render_animation(animation_data)
                
                # Display result
                st.markdown("### Generated Sign Sequence:")
                st.markdown(f"```{result}```")
                
                # Placeholder for future animation rendering
                st.info("In a full implementation, an animated avatar would appear here.")
                
                # Display individual signs
                st.markdown("### Individual Signs:")
                for sign in signs:
                    st.markdown(f"- {sign}")
            else:
                st.warning("No recognizable signs found in the text.")

# Main app logic
if translation_mode == "Sign to Text":
    st.header("Sign to Text Translation")
    st.markdown("This mode uses your webcam to recognize sign language and translate it to text.")
    
    # Start button for webcam
    if st.button("Start Webcam"):
        process_webcam()
    
else:  # Text to Sign
    st.header("Text to Sign Translation")
    st.markdown("This mode converts text input into sign language animations.")
    
    text_to_sign()

# Footer
st.markdown("---")
st.markdown("SignSpeak - A Real-time Sign Language Translator")
st.markdown(
    "Built with ❤️ using Python, MediaPipe, OpenCV, and Streamlit. " +
    "This is a prototype for demonstration purposes."
)

# Add a GitHub link
st.sidebar.markdown("---")
st.sidebar.markdown("### Resources")
st.sidebar.markdown("[GitHub Repository](https://github.com/YourUsername/SignSpeak)")
st.sidebar.markdown("[Report an Issue](https://github.com/YourUsername/SignSpeak/issues)")

# Version information
st.sidebar.markdown("---")
st.sidebar.markdown("### Version Information")
st.sidebar.markdown("SignSpeak v0.1.0 (Alpha)")
st.sidebar.markdown(f"Python {sys.version_split()[0]}")
st.sidebar.markdown(f"OpenCV {cv2.__version__}")
st.sidebar.markdown(f"Streamlit {st.__version__}") 