# server/events.py

from dataclasses import dataclass , field
from typing import Literal , Optional

@dataclass
class VoiceAgentEvent:
    """Base event for voice pipeline"""
    type: str = field(init = False)

@dataclass
class STTChunkEvent(VoiceAgentEvent):
    """Partial transcript from STT"""
    type: Literal["stt_chunk"] = field(default="stt_chunk", init=False)
    text: str
    session_id: Optional[str] = None

@dataclass
class STTOutputEvent(VoiceAgentEvent):
    type: Literal["stt_output"] = field(default="stt_output", init=False)
    transcript: str
    session_id: Optional[str] = None

@dataclass
class AgentChunkEvent(VoiceAgentEvent):
    """Agent response chunk"""
    type: Literal["agent_chunk"] = field(default="agent_chunk", init=False)
    text: str
    session_id: Optional[str] = None

@dataclass
class TTSChunkEvent(VoiceAgentEvent):
    """TTS audio chunk"""
    type: Literal["tts_chunk"] = field(default="tts_chunk", init=False)
    audio: bytes
    session_id: Optional[str] = None

@dataclass
class UITranscriptEvent(VoiceAgentEvent):
    """For displaying user transcript in UI"""
    type: Literal["ui_transcript"] = field(default="ui_transcript", init=False)
    text: str
    session_id: Optional[str] = None

@dataclass
class UIAgentReplyEvent(VoiceAgentEvent):
    """For displaying agent reply in UI"""
    type: Literal["ui_agent_reply"] = field(default="ui_agent_reply", init=False)
    text: str
    session_id: Optional[str] = None