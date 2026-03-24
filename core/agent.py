import os
import wave
import pyaudio
import numpy as np
import soundfile as sf
from transformers import pipeline
from kokoro_onnx import Kokoro

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
        for _ in range(0, int(self.config.RATE / self.config.CHUNK * self.config.RECORD_SECONDS)):
            data = stream.read(self.config.CHUNK)
            frames.append(data)
        
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
        prompt = f"Context: {context}\nQuestion: {text}\nAnswer concisely:"
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
