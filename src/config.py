"""
Configuration file for Image Captioning Project
This file contains all the settings for our project.
"""

class Config:
    # MODEL PARAMETERS - These control how smart our model is
  
    EMBED_SIZE = 256        # How big our word representations are (bigger = more detail)
    HIDDEN_SIZE = 512       # How much memory our LSTM has (bigger = remembers more)
    NUM_LAYERS = 1          # How many LSTM layers (more = more complex)
    
    
    # VOCABULARY SETTINGS - Controls which words we understand
    VOCAB_THRESHOLD = 5     # A word must appear 5+ times to be included
    MAX_CAPTION_LENGTH = 50 # Maximum words in a caption
    
    # TRAINING PARAMETERS - Controls how we train
    BATCH_SIZE = 32         # How many images we process at once
    LEARNING_RATE = 0.001   # How fast our model learns (smaller = slower but safer)
    NUM_EPOCHS = 20         # How many times we go through all data
    
    # IMAGE PROCESSING - How we handle images
    IMAGE_SIZE = 224        # All images resized to 224x224 pixels
    
    # DEVELOPMENT vs PRODUCTION - Easy switching
    DEBUG_MODE = False      # True = use small dataset for testing
    DEBUG_SAMPLES = 1000    # Only use 1000 images when debugging
    
    # FILE PATHS 
    MODEL_SAVE_PATH = "models/"
    VOCAB_SAVE_PATH = "vocab.pkl"

    def __init__(self):
        print("🔧 Configuration loaded:")
        print(f"   Debug Mode: {self.DEBUG_MODE}")
        print(f"   Batch Size: {self.BATCH_SIZE}")
        print(f"   Model Size: {self.EMBED_SIZE}x{self.HIDDEN_SIZE}")
    def set_colab_mode(self):
        """
        Special settings for Google Colab
        Call this when running in Colab
        """
        self.DEBUG_MODE = False  
        self.BATCH_SIZE = 64    
        print("Switched to Colab mode - using full dataset!")
    
    def set_debug_mode(self, debug=True):
        """
        Easy way to switch debug mode on/off
        """
        self.DEBUG_MODE = debug
        if debug:
            print("Debug mode ON - using small dataset")
        else:
            print("Debug mode OFF - using full dataset")
