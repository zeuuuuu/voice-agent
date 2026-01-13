# server/config.py

import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ElevenLabs API
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    
    # ✅ Base URL for your region (WebSocket uses different format)
    # For EU: api.in.residency.elevenlabs.io
    # For US: api.elevenlabs.io
    ELEVENLABS_REGION = os.getenv("ELEVENLABS_REGION", "in.residency")  # or just "" for US
    
    # ElevenLabs voices
    ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    
    # Piper model path
    PIPER_MODEL = None
    
    @classmethod
    def validate(cls):
        if cls.ELEVENLABS_API_KEY:
            return True
        return False

config = Config()