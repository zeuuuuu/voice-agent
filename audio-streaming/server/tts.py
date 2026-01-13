# server/tts.py - PROPERLY ASYNC WITH EXECUTOR

from typing import AsyncIterator
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .events import VoiceAgentEvent, TTSChunkEvent, UIAgentReplyEvent
from .piper_stream import generate_speech_piper
from .elevenlabs_realtime_tts import generate_speech_elevenlabs
from .config import config

logger = logging.getLogger(__name__)

# Thread pool for blocking Piper calls
_piper_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="piper-tts")

async def tts_stream(
    event_stream: AsyncIterator[VoiceAgentEvent],
    audio_track,
    piper_model_path: str,
    use_elevenlabs: bool = False,
) -> AsyncIterator[VoiceAgentEvent]:
    """
    Transform stream: Voice Events → Voice Events (with TTS audio)
    Routes to Piper or ElevenLabs based on flag
    """ 
    full_reply = ""
    word_buffer = []
    
    logger.info(f"🎵 TTS stream started ({'ElevenLabs' if use_elevenlabs else 'Piper'})")

    # ✅ CRITICAL: Define callback that adds audio (will run in thread pool)
    def piper_callback(chunk: bytes) -> bool:
        """Callback for Piper TTS - sends audio to WebRTC track"""
        try:
            if chunk:
                audio_track.add_audio(chunk)
            return True  # ✅ ALWAYS return True to continue streaming
        except Exception as e:
            logger.error(f"Piper callback error: {e}")
            return False

    async for event in event_stream:
        yield event

        if event.type == "agent_chunk":
            full_reply += event.text
            word_buffer.append(event.text)
            
            buffer_text = "".join(word_buffer).strip()
            word_count = len(buffer_text.split())
            ends_sentence = any(p in event.text for p in ".!?")
            
            # ✅ CRITICAL: More aggressive batching to reduce gaps
            # Only flush on sentence boundaries or when buffer is very full
            should_flush = (
                (ends_sentence and word_count >= 5) or  # Sentence end with 5+ words
                (word_count >= 15)  # Force flush at 15 words
            )
            
            if should_flush and buffer_text:
                logger.info(f"🎵 Generating TTS: '{buffer_text}'")
                
                try:
                    if use_elevenlabs:
                        success = generate_speech_elevenlabs(
                            text=buffer_text,
                            audio_track=audio_track,
                        )
                    else:
                        # ✅ CRITICAL: Run blocking Piper in thread pool
                        loop = asyncio.get_event_loop()
                        success = await loop.run_in_executor(
                            _piper_executor,
                            generate_speech_piper,
                            buffer_text,
                            piper_model_path,
                            piper_callback
                        )
                    
                    if not success:
                        logger.error(f"❌ TTS generation failed for: '{buffer_text}'")
                    else:
                        logger.info(f"✅ TTS generated for: '{buffer_text[:30]}...'")
                
                except Exception as e:
                    logger.error(f"❌ TTS exception: {e}")
                    import traceback
                    traceback.print_exc()
                
                word_buffer = []

    # Final flush
    if word_buffer:
        buffer_text = "".join(word_buffer).strip()
        if buffer_text:
            logger.info(f"🎵 Generating final TTS: '{buffer_text}'")
            
            try:
                if use_elevenlabs:
                    success = generate_speech_elevenlabs(
                        text=buffer_text,
                        audio_track=audio_track,
                    )
                else:
                    # ✅ CRITICAL: Run blocking Piper in thread pool
                    loop = asyncio.get_event_loop()
                    success = await loop.run_in_executor(
                        _piper_executor,
                        generate_speech_piper,
                        buffer_text,
                        piper_model_path,
                        piper_callback
                    )
                
                if not success:
                    logger.error(f"❌ Final TTS failed for: '{buffer_text}'")
                else:
                    logger.info(f"✅ Final TTS generated: '{buffer_text[:30]}...'")
                    
            except Exception as e:
                logger.error(f"❌ Final TTS exception: {e}")
                import traceback
                traceback.print_exc()

    if full_reply.strip():
        yield UIAgentReplyEvent(text=full_reply.strip())
    
    logger.info(f"✅ TTS stream ended ({len(full_reply)} chars)")