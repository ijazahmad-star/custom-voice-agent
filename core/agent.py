import os
import wave
import pyaudio
import numpy as np
import soundfile as sf
# from ctypes import *
# from contextlib import contextmanager
from transformers import pipeline
from kokoro_onnx import Kokoro

# ALSA error handler to suppress PortAudio warnings on Linux
# ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
# def py_error_handler(filename, line, function, err, fmt):
#     pass
# c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

# @contextmanager
# def suppress_alsa_warnings():
#     try:
#         asound = cdll.LoadLibrary('libasound.so.2')
#         asound.snd_lib_error_set_handler(c_error_handler)
#         yield
#         asound.snd_lib_error_set_handler(None)
#     except OSError:
#         try:
#             asound = cdll.LoadLibrary('libasound.so')
#             asound.snd_lib_error_set_handler(c_error_handler)
#             yield
#             asound.snd_lib_error_set_handler(None)
#         except OSError:
#             yield

from core.knowledge import setup_knowledge_base

class VoiceAgent:
    """A local voice-to-voice agent that uses RAG, LLMs, and TTS."""
    
    def __init__(self, config):
        self.config = config
        self.vector_db = None
        self.stt_model = None
        self.llm_model = None
        self.tts = None
        
        self.vector_db = setup_knowledge_base(self.config)
        self._load_models()

    def _load_models(self):
        print(f"Loading Models on {self.config.DEVICE}...")
        self.stt_model = pipeline(
            "automatic-speech-recognition", 
            model=self.config.STT_MODEL, 
            device=self.config.DEVICE
        )
        self.llm_model = pipeline(
            "text-generation", 
            model=self.config.LLM_MODEL, 
            device_map="auto"
        )
        self.tts = Kokoro(self.config.KOKORO_MODEL, self.config.KOKORO_VOICES)

    def record_audio(self, output_filename="input.wav"):
        """Records audio from mic and saves to output_filename"""
        # with suppress_alsa_warnings():
            # p = pyaudio.PyAudio()
        p = pyaudio.PyAudio()
        stream = p.open(
            format=self.config.FORMAT, 
            channels=self.config.CHANNELS, 
            rate=self.config.RATE, 
            input=True, 
            frames_per_buffer=self.config.CHUNK
        )
        
        print("\n🎤 Listening... (Speak now)")
        frames = []
        silence_frames = 0
        silence_threshold = getattr(self.config, 'SILENCE_THRESHOLD', 500)
        silence_duration_frames = int(
            self.config.RATE / self.config.CHUNK * getattr(self.config, 'SILENCE_DURATION_SECONDS', 1.5)
        )
        max_frames = int(self.config.RATE / self.config.CHUNK * self.config.RECORD_SECONDS)
        
        for _ in range(max_frames):
            data = stream.read(self.config.CHUNK)
            frames.append(data)
            
            # Calculate RMS energy to detect silence
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
            
            if rms < silence_threshold:
                silence_frames += 1
            else:
                silence_frames = 0
                
            if silence_frames >= silence_duration_frames:
                print("🔇 Silence detected, processing...")
                break
        
        print("🛑 Recording finished.")
        stream.stop_stream()
        stream.close()
        p.terminate()

        with wave.open(output_filename, 'wb') as wf:
            wf.setnchannels(self.config.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.config.FORMAT))
            wf.setframerate(self.config.RATE)
            wf.writeframes(b''.join(frames))
            
        return output_filename

    def process_and_speak(self, input_filename="input.wav", output_filename="response.wav"):
        """Processes the recorded audio, gets an AI response, and plays it back."""
        # A. Speech-To-Text (STT)
        text = self.stt_model(input_filename)["text"]
        print(f"You said: {text}")

        # B. Context Retrieval (RAG)
        docs = self.vector_db.similarity_search(text, k=1)
        context = docs[0].page_content if docs else "No relevant context found."
        
        # C. LLM Generation
        prompt = (
            f"<|im_start|>system\nYou are a voice assistant. Provide ONLY the direct answer to the user's question based on the context. "
            f"Do NOT repeat the question, do not explain your reasoning, and do use conversational filler.<|im_end|>\n"
            f"<|im_start|>user\nContext: {context}\nQuestion: {text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        generated_output = self.llm_model(prompt, max_new_tokens=50, return_full_text=False)
        response = generated_output[0]['generated_text']
        print(f"AI: {response}")

        # D. Text-To-Speech (TTS)
        samples, sample_rate = self.tts.create(response, voice="af_heart", speed=1.1)
        sf.write(output_filename, samples, sample_rate)
        
        # E. Playback
        print("🔊 Speaking...")
        os.system(f"ffplay -nodisp -autoexit -loglevel quiet {output_filename}")

    def run(self):
        """Starts the interactive voice agent loop."""
        try:
            while True:
                input("\nPress Enter to start recording (or Ctrl+C to quit)...")
                self.record_audio()
                self.process_and_speak()
        except KeyboardInterrupt:
            print("\nExiting voice agent. Goodbye!")
