"""
Configuration module for the Islamic Text Verification System.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv
from langchain_ollama.chat_models import ChatOllama

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class."""
    
    # Ollama Configuration
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b-it-fp16")
    OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # File Paths
    INPUT_FILE = os.getenv("INPUT_FILE", "dataset/dev_top20_matches.pkl")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "final.jbl")
    
    # Text Processing Options
    REMOVE_DIACRITICS = os.getenv("REMOVE_DIACRITICS", "true").lower() == "true"
    
    @classmethod
    def get_results_folder(cls):
        """Generate results folder path based on model and diacritic settings."""
        # Clean model name for folder (replace : with _ and remove special chars)
        model_name = cls.OLLAMA_MODEL.replace(":", "_").replace("/", "_")
        diacritic_suffix = "with_diacritic" if cls.REMOVE_DIACRITICS else "without_diacritic"
        folder_name = f"{model_name}_{diacritic_suffix}"
        
        results_path = os.path.join("results", folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(results_path, exist_ok=True)
        
        return results_path
    
    @classmethod
    def get_output_file_path(cls, filename=None):
        """Get full output file path in the appropriate results subfolder."""
        if filename is None:
            filename = cls.OUTPUT_FILE
        
        results_folder = cls.get_results_folder()
        return os.path.join(results_folder, filename)
    
    @classmethod
    def get_llm(cls):
        """Create and return configured LLM instance."""
        return ChatOllama(
            model=cls.OLLAMA_MODEL,
            temperature=cls.OLLAMA_TEMPERATURE,
            base_url=cls.OLLAMA_BASE_URL
        )
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present."""
        # Try multiple possible locations for the input file
        possible_paths = [
            cls.INPUT_FILE,  # dataset/dev_top20_matches.pkl
            "dev_top20_matches.pkl",  # root directory fallback
            os.path.join("dataset", "dev_top20_matches.pkl")  # explicit dataset path
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                cls.INPUT_FILE = path  # Update to the found path
                print(f"✅ Found input file at: {path}")
                return True
        
        raise FileNotFoundError(f"Input file not found in any of these locations: {possible_paths}")
        
        return True
