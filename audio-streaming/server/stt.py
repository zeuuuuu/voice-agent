# server/stt.py

from typing import AsyncIterator
import numpy as np
import wave
import tempfile
import time
import logging
from .models import whisper_model, get_speech_timestamps, vad_model
from .events import VoiceAgentEvent, STTOutputEvent, UITranscriptEvent

logger = logging.getLogger(__name__)

# Session-specific buffers
_buffers = {}
_last_speech = {}

def init_stt_session(session_id: str):
    """Initialize STT buffers for a session"""
    _buffers[session_id] = b""
    _last_speech[session_id] = time.time()
    logger.info(f"STT session {session_id} initialized")

def cleanup_stt_session(session_id: str):
    """Clean up STT session data"""
    _buffers.pop(session_id, None)
    _last_speech.pop(session_id, None)
    logger.info(f"STT session {session_id} cleaned up")

async def stt_stream(
    audio_stream: AsyncIterator[tuple[str, bytes]]
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Audio bytes → STT events
    
    Yields:
        - STTOutputEvent: Final transcript (triggers agent)
        - UITranscriptEvent: For displaying in browser
    """
    session_id = None
    chunk_count = 0
    
    logger.info("STT stream started")
    
    async for sid, audio_chunk in audio_stream:
        chunk_count += 1
        
        # Initialize session on first chunk
        if session_id is None:
            session_id = sid
            init_stt_session(session_id)
            logger.info(f"Processing audio for session: {session_id}")
        
        logger.debug(f"Received audio chunk {chunk_count}: {len(audio_chunk)} bytes")
        
        # Skip tiny chunks
        if len(audio_chunk) < 8000:
            logger.debug(f"Skipping small chunk ({len(audio_chunk)} bytes)")
            continue

        _buffers[session_id] += audio_chunk
        logger.debug(f"Buffer size: {len(_buffers[session_id])} bytes")

        # Need minimum buffer size
        if len(_buffers[session_id]) < 48000 * 2 * 0.5:
            logger.debug(f"Buffer too small, need {48000 * 2 * 0.5} bytes")
            continue

        # Convert to float for VAD
        audio_int16 = np.frombuffer(_buffers[session_id], dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Detect speech timestamps
        speech = get_speech_timestamps(
            audio_float,
            vad_model,
            sampling_rate=48000,
            min_speech_duration_ms=120
        )

        if speech:
            _last_speech[session_id] = time.time()
            logger.debug(f"Speech detected: {len(speech)} segments")
        else:
            logger.debug("No speech detected in this chunk")

        # Reset buffer if no speech for 1.5s
        if time.time() - _last_speech.get(session_id, 0) > 1.5:
            logger.debug("No speech for 1.5s, resetting buffer")
            _buffers[session_id] = b""
            continue

        # Check if speech has ended (0.5s silence after last speech)
        if speech:
            duration = len(audio_float) / 48000
            last_speech_end = speech[-1]["end"] / 48000
            silence_duration = duration - last_speech_end
            
            logger.debug(f"Silence after speech: {silence_duration:.2f}s")
            
            if silence_duration > 0.5:
                # Transcribe the complete utterance
                logger.info("Speech ended, starting transcription...")
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
                    with wave.open(f.name, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(48000)
                        wav.writeframes(_buffers[session_id])

                    result = whisper_model.transcribe(f.name, language="en", fp16=False)
                    text = result["text"].strip()

                _buffers[session_id] = b""
                logger.info(f"Transcription complete: '{text}'")

                if len(text) >= 3:
                    # Yield UI event for display
                    logger.info(f"Yielding UITranscriptEvent: {text}")
                    yield UITranscriptEvent(text=text)
                    
                    # Yield final transcript to trigger agent
                    logger.info(f"Yielding STTOutputEvent: {text}")
                    yield STTOutputEvent(transcript=text)
                else:
                    logger.warning(f"Transcript too short ({len(text)} chars), ignoring")
    
    logger.info(f"STT stream ended after {chunk_count} chunks")