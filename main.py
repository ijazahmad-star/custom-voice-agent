from core.config import VoiceAgentConfig
from core.agent import VoiceAgent

if __name__ == "__main__":
    agent = VoiceAgent(config=VoiceAgentConfig)
    agent.run()