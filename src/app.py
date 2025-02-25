"""
SignSpeak - Main Application

This module serves as the entry point for the SignSpeak application, 
integrating sign language recognition, translation, and generation components.
"""

import cv2
import time
import logging
import sys
import numpy as np

# Import configuration
from config import (
    WEBCAM_INDEX, 
    DEBUG_MODE, 
    LOG_LEVEL,
    MEDIAPIPE_MODEL_COMPLEXITY,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE
)

# Import components (to be implemented)
try:
    from sign_recognition.detector import SignDetector
    from translation.translator import Translator
    from sign_generation.generator import SignGenerator
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.info("Ensure you've installed all dependencies with 'pip install -r requirements.txt'")
    sys.exit(1)

# Configure logging
logging_level = getattr(logging, LOG_LEVEL)
logging.basicConfig(
    level=logging_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SignSpeakApp:
    """Main application class for SignSpeak."""
    
    def __init__(self):
        """Initialize the application components."""
        logger.info("Initializing SignSpeak application...")
        
        # Initialize sign detection component
        self.detector = SignDetector(
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
        # Initialize translation component
        self.translator = Translator()
        
        # Initialize sign generation component
        self.generator = SignGenerator()
        
        # Initialize webcam
        self.cap = None
        self.running = False
        
        logger.info("SignSpeak application initialized successfully")
    
    def start_webcam(self):
        """Start the webcam capture."""
        logger.info(f"Starting webcam with index {WEBCAM_INDEX}")
        self.cap = cv2.VideoCapture(WEBCAM_INDEX)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open webcam with index {WEBCAM_INDEX}")
            return False
        
        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        return True
    
    def stop_webcam(self):
        """Stop the webcam capture."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam stopped")
    
    def run(self):
        """Run the main application loop."""
        if not self.start_webcam():
            return
        
        self.running = True
        logger.info("Starting main application loop")
        
        # FPS calculation variables
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        # Last detected sign and translation
        last_sign = None
        last_translation = None
        translation_display_time = 0
        
        try:
            while self.running:
                # Capture frame from webcam
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to capture frame from webcam")
                    break
                
                # Process the frame (detect signs)
                frame, detected_keypoints = self.detector.process_frame(frame)
                
                # Calculate FPS
                fps_frame_count += 1
                if (time.time() - fps_start_time) > 1:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_start_time = time.time()
                    fps_frame_count = 0
                
                # Display FPS on the frame
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # If keypoints are detected, try to recognize sign
                current_time = time.time()
                if detected_keypoints and (last_sign is None or current_time - translation_display_time > 2):
                    recognized_sign = self.detector.recognize_sign(detected_keypoints)
                    
                    if recognized_sign and recognized_sign != last_sign:
                        last_sign = recognized_sign
                        
                        # Translate recognized sign to text
                        translated_text = self.translator.sign_to_text(recognized_sign)
                        last_translation = translated_text
                        translation_display_time = current_time
                        
                        logger.info(f"Recognized: '{recognized_sign}' -> '{translated_text}'")
                
                # Display the recognized sign and translation
                if last_sign and last_translation:
                    time_since_translation = current_time - translation_display_time
                    if time_since_translation < 3:  # Display for 3 seconds
                        alpha = min(1.0, 3 - time_since_translation)  # Fade out effect
                        cv2.putText(frame, f"Sign: {last_sign}", (10, frame.shape[0] - 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(frame, f"Translation: {last_translation}", (10, frame.shape[0] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                cv2.imshow('SignSpeak', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
        
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.exception(f"An error occurred: {e}")
        finally:
            self.stop_webcam()


def main():
    """Main function to run the SignSpeak application."""
    try:
        app = SignSpeakApp()
        app.run()
    except Exception as e:
        logger.exception(f"Application crashed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 