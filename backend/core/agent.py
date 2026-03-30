import os
import wave
import pyaudio
import numpy as np
import soundfile as sf
from transformers import pipeline
from kokoro_onnx import Kokoro

from core.knowledge import setup_knowledge_base
from core.factory import create_agent

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
        self._setup_agent()

    def _load_models(self):
        print(f"Loading Models on {self.config.DEVICE}...")
        self.stt_model = pipeline(
            "automatic-speech-recognition", 
            model=self.config.STT_MODEL, 
            device=self.config.DEVICE
        )
        if not getattr(self.config, 'USE_GROQ', False):
            self.llm_model = pipeline(
                "text-generation", 
                model=self.config.LLM_MODEL, 
                device_map="auto",
                max_new_tokens=512,
                return_full_text=False,
                token=os.getenv("HF_TOKEN")
            )
        else:
            print(f"Using Groq Cloud Inference with model: {self.config.LLM_MODEL}")
        self.tts = Kokoro(self.config.KOKORO_MODEL, self.config.KOKORO_VOICES)

    def _setup_agent(self):
        """Initializes the underlying LangGraph agent using the factory."""
        self.agent = create_agent(
            config=self.config,
            vector_db=self.vector_db,
            llm_pipeline=self.llm_model if not getattr(self.config, 'USE_GROQ', False) else None
        )

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
        silence_frames = 0
        silence_threshold = getattr(self.config, 'SILENCE_THRESHOLD', 500)
        silence_duration_frames = int(
            self.config.RATE / self.config.CHUNK * getattr(self.config, 'SILENCE_DURATION_SECONDS', 1.5)
        )
        max_frames = int(self.config.RATE / self.config.CHUNK * self.config.RECORD_SECONDS)
        
        for _ in range(max_frames):
            try:
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
            except Exception as e:
                print(f"Error reading stream: {e}")
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
        try:
            text = self.stt_model(input_filename)["text"]
            if not text:
                print("No audio detected.")
                return
            print(f"You said: {text}")
        except Exception as e:
            print(f"STT error: {e}")
            return

        # B. Agent Thinking
        print("🧠 Agent is thinking...")
        try:
            clean_text = text.strip()
            # thread_id: allows persistence of chat history per session
            config = {"configurable": {"thread_id": "user_session_1"}}
            
            result = self.agent.invoke({"messages": [("user", clean_text)]}, config=config)
            response = result["messages"][-1].content
        except Exception as e:
            response = "I encountered an error while thinking. Let's try again."
            print(f"Agent error: {e}")
            
        print(f"AI: {response}")

        # D. Text-To-Speech (TTS)
        try:
            samples, sample_rate = self.tts.create(response, voice="af_heart", speed=1.1)
            sf.write(output_filename, samples, sample_rate)
            
            # E. Playback
            print("🔊 Speaking...")
            # Note:ffplay requires the binary to be installed on the system.
            os.system(f"ffplay -nodisp -autoexit -loglevel quiet {output_filename}")
        except Exception as e:
            print(f"TTS/Playback error: {e}")

    def run(self):
        """Starts the interactive voice agent loop."""
        try:
            while True:
                input("\nPress Enter to start recording (or Ctrl+C to quit)...")
                self.record_audio()
                self.process_and_speak()
        except KeyboardInterrupt:
            print("\nExiting voice agent. Goodbye!")
