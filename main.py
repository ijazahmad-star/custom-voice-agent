import os
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN)
load_dotenv()

from core.config import VoiceAgentConfig
from core.agent import VoiceAgent

if __name__ == "__main__":
    agent = VoiceAgent(config=VoiceAgentConfig)
    agent.run()