"""
Translator Module

This module handles the translation between sign language and natural language
using Large Language Models (LLMs) and other NLP techniques.
"""

import os
import logging
import json
from typing import Optional, List, Dict, Any, Union
import time

# Optional imports for different LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not available. OpenAI translation will be disabled.")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers package not available. Local LLM translation will be disabled.")

# Setup logging
logger = logging.getLogger(__name__)


class Translator:
    """
    Class for translating between sign language and natural language.
    
    Uses LLMs (like OpenAI's GPT or local models via HuggingFace Transformers)
    to generate translations between sign language and natural language.
    """
    
    def __init__(self, model_type: str = "openai", model_name: str = None, **kwargs):
        """
        Initialize the translator.
        
        Args:
            model_type: Type of model to use ('openai', 'transformers', or 'dummy')
            model_name: Name of the specific model to use (if applicable)
            **kwargs: Additional model-specific parameters
        """
        logger.info(f"Initializing Translator with model type: {model_type}")
        
        self.model_type = model_type
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.translation_history = []
        
        # Setup based on model type
        if model_type == "openai":
            self._setup_openai()
        elif model_type == "transformers":
            self._setup_transformers(model_name, **kwargs)
        else:
            logger.info("Using dummy translator (rule-based mappings)")
            self._setup_dummy()
    
    def _setup_openai(self):
        """Setup OpenAI API for translation."""
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package is not installed. Please install it with 'pip install openai'.")
        
        # Get API key from environment variable
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        
        logger.info("OpenAI translator setup complete")
    
    def _setup_transformers(self, model_name: str = None, **kwargs):
        """
        Setup HuggingFace Transformers for local translation.
        
        Args:
            model_name: Name of the pre-trained model to use
            **kwargs: Additional model parameters
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers package is not installed. Please install it with 'pip install transformers'.")
        
        # Default to a general translation model if none specified
        if not model_name:
            model_name = "t5-small"  # Small model for testing, use t5-base or larger for better results
        
        try:
            logger.info(f"Loading Transformers model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create translation pipeline
            self.pipeline = pipeline(
                "text2text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                **kwargs
            )
            
            logger.info("Transformers translator setup complete")
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {e}")
            self._setup_dummy()  # Fallback to dummy translator
    
    def _setup_dummy(self):
        """Setup a simple rule-based dummy translator."""
        self.dummy_translations = {
            "HELLO": "Hello there!",
            "THANK YOU": "Thank you very much!",
            "YES": "Yes.",
            "NO": "No.",
            "HELP": "Can you help me please?",
            "WHAT": "What is that?",
            "HOW": "How are you?",
            "WHERE": "Where is it?",
            "WANT": "I want that.",
            "NEED": "I need help."
        }
        
        # Reverse dictionary for text-to-sign
        self.text_to_sign_mapping = {
            "hello": "HELLO",
            "hi": "HELLO",
            "hey": "HELLO",
            "thanks": "THANK YOU",
            "thank you": "THANK YOU",
            "yes": "YES",
            "yeah": "YES",
            "no": "NO",
            "nope": "NO",
            "help": "HELP",
            "what": "WHAT",
            "how": "HOW",
            "where": "WHERE",
            "want": "WANT",
            "need": "NEED"
        }
        
        logger.info("Dummy translator setup complete")
    
    def sign_to_text(self, sign: str, context: List[str] = None) -> str:
        """
        Translate a sign language token to natural language text.
        
        Args:
            sign: The sign language token or sequence to translate
            context: Optional list of previous signs for context-aware translation
            
        Returns:
            Translated natural language text
        """
        if not sign:
            return ""
        
        # Add to history
        self.translation_history.append({"sign": sign, "timestamp": time.time()})
        if len(self.translation_history) > 10:  # Keep only the last 10 translations
            self.translation_history.pop(0)
        
        # Choose translation method based on model type
        if self.model_type == "openai":
            return self._translate_with_openai(sign, context)
        elif self.model_type == "transformers":
            return self._translate_with_transformers(sign, context)
        else:
            # Dummy translator (rule-based)
            return self.dummy_translations.get(sign, sign)
    
    def _translate_with_openai(self, sign: str, context: List[str] = None) -> str:
        """
        Translate sign to text using OpenAI API.
        
        Args:
            sign: The sign to translate
            context: Optional list of previous signs for context
            
        Returns:
            Translated text
        """
        try:
            # Prepare prompt with sign and context if available
            prompt = f"Translate the sign language token '{sign}' to natural language text."
            
            if context:
                context_str = ", ".join(context)
                prompt += f" Previous signs were: {context_str}."
            
            # Call OpenAI API
            response = openai.Completion.create(
                engine="text-davinci-003",  # or a more advanced model
                prompt=prompt,
                max_tokens=50,
                temperature=0.7,
                n=1,
                stop=None
            )
            
            # Extract and clean up the response
            translation = response.choices[0].text.strip()
            
            return translation
        
        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            # Fallback to dummy translator
            return self.dummy_translations.get(sign, sign)
    
    def _translate_with_transformers(self, sign: str, context: List[str] = None) -> str:
        """
        Translate sign to text using local transformers model.
        
        Args:
            sign: The sign to translate
            context: Optional list of previous signs for context
            
        Returns:
            Translated text
        """
        try:
            # Prepare input with sign and context if available
            input_text = f"translate sign to text: {sign}"
            
            if context:
                context_str = " ".join(context)
                input_text += f" context: {context_str}"
            
            # Generate translation
            result = self.pipeline(input_text, max_length=50, do_sample=True)[0]
            translation = result["generated_text"]
            
            return translation
        
        except Exception as e:
            logger.error(f"Transformers translation error: {e}")
            # Fallback to dummy translator
            return self.dummy_translations.get(sign, sign)
    
    def text_to_sign(self, text: str) -> List[str]:
        """
        Translate natural language text to a sequence of sign language tokens.
        
        Args:
            text: The natural language text to translate
            
        Returns:
            List of sign language tokens representing the text
        """
        # This is a simplified approach - a full implementation would involve:
        # 1. Analyzing the sentence structure
        # 2. Mapping to appropriate sign language grammar
        # 3. Generating a sequence of sign tokens that follows sign language grammar
        
        if not text:
            return []
        
        # Simple word-by-word mapping for demonstration
        words = text.lower().split()
        sign_tokens = []
        
        for word in words:
            clean_word = word.strip('.,!?;:()[]{}"\'-')
            if clean_word in self.text_to_sign_mapping:
                sign_tokens.append(self.text_to_sign_mapping[clean_word])
        
        # For any model-based approach, this would be similar to the sign_to_text method
        # but in reverse, likely using the same LLM with a different prompt
        
        return sign_tokens


# Example usage if this file is run directly
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the translator
    translator = Translator(model_type="dummy")
    
    # Test sign to text
    sign = "HELLO"
    text = translator.sign_to_text(sign)
    print(f"Sign: {sign} -> Text: {text}")
    
    # Test text to sign
    text = "Hello, thank you for your help"
    signs = translator.text_to_sign(text)
    print(f"Text: '{text}' -> Signs: {signs}") 