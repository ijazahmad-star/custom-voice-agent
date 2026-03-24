import pyaudio
import torch

class VoiceAgentConfig:
    """Configuration settings for the Voice Agent"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Configurations
    STT_MODEL = "openai/whisper-large-v3-turbo"
    LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    KOKORO_MODEL = "kokoro-v1.0.onnx"
    KOKORO_VOICES = "voices-v1.0.bin"
