"""
Tests for the SignSpeak Recognition Component

This module contains unit tests for the sign recognition
functionality of the SignSpeak application.
"""

import unittest
import numpy as np
import cv2
import os
import sys

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the detector module
from sign_recognition.detector import SignDetector


class TestSignDetector(unittest.TestCase):
    """Test cases for the SignDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = SignDetector()
        
        # Create a simple test image (black background)
        self.test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Create a second test image with a simple shape (circle)
        self.test_image_with_shape = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(self.test_image_with_shape, (320, 240), 50, (0, 0, 255), -1)

    def test_detector_initialization(self):
        """Test that the detector initializes correctly."""
        self.assertIsNotNone(self.detector)
        self.assertIsNotNone(self.detector.holistic)
    
    def test_process_frame_empty(self):
        """Test processing an empty (None) frame."""
        result, keypoints = self.detector.process_frame(None)
        self.assertIsNone(result)
        self.assertIsNone(keypoints)
    
    def test_process_frame_black(self):
        """Test processing a black frame."""
        result, keypoints = self.detector.process_frame(self.test_image)
        
        # The frame should be processed (not None)
        self.assertIsNotNone(result)
        
        # A black image should have no detectable keypoints
        self.assertIsNone(keypoints)
    
    def test_placeholder_recognition(self):
        """Test the placeholder recognition function."""
        # Create mock keypoints with hands raised above shoulders
        mock_keypoints = {
            'right_hand': np.zeros((21, 3)),
            'left_hand': np.zeros((21, 3)),
            'pose': np.zeros((33, 3))
        }
        
        # Mock right hand, left hand, and shoulders positions
        # Right hand above right shoulder, left hand above left shoulder
        mock_keypoints['right_hand'][0] = [0.7, 0.3, 0]  # Right wrist above shoulder
        mock_keypoints['left_hand'][0] = [0.3, 0.3, 0]   # Left wrist above shoulder
        mock_keypoints['pose'][11] = [0.3, 0.4, 0]       # Left shoulder
        mock_keypoints['pose'][12] = [0.7, 0.4, 0]       # Right shoulder
        
        # Test recognition
        sign = self.detector._placeholder_recognition(mock_keypoints)
        self.assertEqual(sign, "HELLO")
        
        # Now test with only right hand raised (should be "YES")
        mock_keypoints_right_only = {
            'right_hand': np.zeros((21, 3)),
            'left_hand': np.zeros((21, 3)) * 0,  # Empty array
            'pose': np.zeros((33, 3))
        }
        mock_keypoints_right_only['right_hand'][0] = [0.7, 0.3, 0]  # Right wrist above shoulder
        mock_keypoints_right_only['pose'][12] = [0.7, 0.4, 0]       # Right shoulder
        
        sign = self.detector._placeholder_recognition(mock_keypoints_right_only)
        self.assertEqual(sign, "YES")
    
    def test_recognize_sign(self):
        """Test the main recognize_sign function."""
        # Test with None keypoints
        sign = self.detector.recognize_sign(None)
        self.assertIsNone(sign)
        
        # Test with valid keypoints
        mock_keypoints = {
            'right_hand': np.zeros((21, 3)),
            'left_hand': np.zeros((21, 3)),
            'pose': np.zeros((33, 3))
        }
        
        # Set up for "HELLO" sign
        mock_keypoints['right_hand'][0] = [0.7, 0.3, 0]
        mock_keypoints['left_hand'][0] = [0.3, 0.3, 0]
        mock_keypoints['pose'][11] = [0.3, 0.4, 0]
        mock_keypoints['pose'][12] = [0.7, 0.4, 0]
        
        sign = self.detector.recognize_sign(mock_keypoints)
        self.assertIsNotNone(sign)


if __name__ == '__main__':
    unittest.main() 