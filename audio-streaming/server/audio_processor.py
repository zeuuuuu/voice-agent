# server/audio_processor.py

from typing import AsyncIterator
import numpy as np
import wave
import tempfile
import os
import time
import logging
import asyncio
import torch
import torchaudio.functional as F
from concurrent.futures import ThreadPoolExecutor
from .models import whisper_model, get_speech_timestamps, vad_model
from .events import STTOutputEvent
from .config import config
from .elevenlabs_realtime_stt import transcribe_elevenlabs

logger = logging.getLogger(__name__)

_buffers = {}
_last_speech = {}

# Thread pool for blocking operations
_executor = ThreadPoolExecutor(max_workers=2)

def init_session(session_id: str):
    _buffers[session_id] = b""
    _last_speech[session_id] = time.time()
    logger.info(f"Session {session_id} initialized")

def cleanup_session(session_id: str):
    _buffers.pop(session_id, None)
    _last_speech.pop(session_id, None)
    logger.info(f"Session {session_id} cleaned up")

def _transcribe_sync_whisper(wav_path: str) -> str:
    """Blocking Whisper transcription (runs in thread pool)"""
    try:
        if not os.path.exists(wav_path):
            logger.error(f"WAV file does not exist: {wav_path}")
            return ""
        
        file_size = os.path.getsize(wav_path)
        logger.info(f"Transcribing WAV file with Whisper: {file_size} bytes")
        
        if file_size < 1000:
            logger.warning(f"WAV file too small: {file_size} bytes")
            return ""
        
        # Verify WAV file
        try:
            with wave.open(wav_path, 'rb') as wf:
                logger.debug(f"WAV: {wf.getnchannels()}ch, {wf.getframerate()}Hz, {wf.getnframes()} frames")
        except Exception as e:
            logger.error(f"Invalid WAV file: {e}")
            return ""
        
        result = whisper_model.transcribe(wav_path, language="en", fp16=False)
        text = result["text"].strip()
        
        logger.info(f"Whisper complete: '{text}'")
        return text
        
    except Exception as e:
        logger.error(f"Whisper error: {e}")
        import traceback
        traceback.print_exc()
        return ""

def _transcribe_sync_elevenlabs(wav_path: str) -> str:
    """Blocking ElevenLabs transcription (runs in thread pool)"""
    try:
        if not os.path.exists(wav_path):
            logger.error(f"WAV file does not exist: {wav_path}")
            return ""
        
        file_size = os.path.getsize(wav_path)
        logger.info(f"Transcribing WAV file with ElevenLabs: {file_size} bytes")
        
        if file_size < 1000:
            logger.warning(f"WAV file too small: {file_size} bytes")
            return ""
        
        # Read WAV as bytes
        with open(wav_path, 'rb') as f:
            audio_bytes = f.read()
        
        # Run async ElevenLabs in new event loop (we're in thread pool)
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        text = loop.run_until_complete(transcribe_elevenlabs(audio_bytes))
        loop.close()
        
        return text
        
    except Exception as e:
        logger.error(f"ElevenLabs STT error: {e}")
        import traceback
        traceback.print_exc()
        return ""

async def process_audio_chunk(
    session_id: str, 
    audio_data: bytes, 
    pipeline,
    use_elevenlabs: bool = False
):
    """WebRTC-aware streaming STT processor with provider selection"""
    result = {"transcript": "", "is_final": False, "ai_reply": ""}

    # Accumulate audio
    _buffers[session_id] += audio_data

    # Need at least 0.5s of audio
    if len(_buffers[session_id]) < 48000 * 2 * 0.5:
        return result

    # ✅ CRITICAL FIX: Correct int16 → float conversion
    audio_int16 = np.frombuffer(_buffers[session_id], dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32767.0  # Use 32767.0 for proper range
    
    # ✅ Check audio quality BEFORE processing
    audio_max = np.abs(audio_float).max()
    audio_rms = np.sqrt(np.mean(audio_float ** 2))
    
    logger.debug(f"Buffer audio stats: max={audio_max:.4f}, rms={audio_rms:.4f}, bytes={len(_buffers[session_id])}")
    
    # ✅ NO GAIN APPLIED - keep original audio levels
    # Only log if audio seems problematic
    if audio_max < 0.01:
        # logger.warning(f"⚠️ Audio is nearly silent! max={audio_max:.4f}, rms={audio_rms:.4f}")
        # Don't process silent audio
        return result
    
    if audio_max > 0.95:
        logger.warning(f"⚠️ Audio is clipping! max={audio_max:.4f}")
    
    # ✅ Resample 48kHz → 16kHz for VAD
    audio_tensor = torch.from_numpy(audio_float)
    audio_16k = F.resample(audio_tensor, orig_freq=48000, new_freq=16000)
    audio_16k_np = audio_16k.numpy()

    # Run VAD
    speech = get_speech_timestamps(
        audio_16k_np,
        vad_model,
        sampling_rate=16000,
        min_speech_duration_ms=250,
    )

    if speech:
        _last_speech[session_id] = time.time()
        logger.debug(f"Speech detected: {len(speech)} segments")
    else:
        logger.debug(f"No speech detected in this chunk")

    # Reset buffer if silence too long
    if time.time() - _last_speech.get(session_id, 0) > 1.5:
        logger.debug(f"No speech for 1.5s, resetting {len(_buffers[session_id])} bytes")
        _buffers[session_id] = b""
        return result

    # Check if speech ended
    if speech:
        duration = len(audio_16k_np) / 16000
        last_speech_end = speech[-1]["end"] / 16000
        silence = duration - last_speech_end

        if silence > 0.5:
            provider = "ElevenLabs" if use_elevenlabs else "Whisper"
            logger.info(f"Speech ended → transcribing with {provider} ({len(_buffers[session_id])} bytes)")

            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            
            # Save debug copy BEFORE clearing buffer
            debug_wav = f"debug_{session_id}_{int(time.time())}.wav"

            try:
                # ✅ CRITICAL: Save the ORIGINAL audio_float, not modified
                # Convert back to int16 for WAV file
                audio_to_save_int16 = np.frombuffer(_buffers[session_id], dtype=np.int16)
                
                logger.info(f"📊 Audio to save stats:")
                logger.info(f"   Samples: {len(audio_to_save_int16)}")
                logger.info(f"   Int16 range: [{audio_to_save_int16.min()}, {audio_to_save_int16.max()}]")
                logger.info(f"   Duration: {len(audio_to_save_int16) / 48000:.2f}s")
                
                # Save WAV file
                with wave.open(wav_path, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(48000)
                    wav.writeframes(_buffers[session_id])
                
                # Save debug copy
                try:
                    with wave.open(debug_wav, "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(48000)
                        wav.writeframes(_buffers[session_id])
                    logger.info(f"✅ Saved debug WAV: {debug_wav}")
                    
                    # ✅ VERIFY the saved WAV
                    with wave.open(debug_wav, 'rb') as wf:
                        logger.info(f"✅ WAV verification: {wf.getnchannels()}ch, {wf.getframerate()}Hz, {wf.getnframes()} frames")
                        
                except Exception as e:
                    logger.warning(f"Could not save debug WAV: {e}")

                # Clear buffer AFTER saving
                _buffers[session_id] = b""

                # Run transcription in thread pool
                loop = asyncio.get_event_loop()
                
                if use_elevenlabs:
                    text = await loop.run_in_executor(
                        _executor,
                        _transcribe_sync_elevenlabs,
                        wav_path
                    )
                else:
                    text = await loop.run_in_executor(
                        _executor,
                        _transcribe_sync_whisper,
                        wav_path
                    )

                if len(text) >= 3:
                    result["transcript"] = text
                    result["is_final"] = True
                    logger.info(f"📝 Transcript ({provider}): '{text}'")

                    # Run through pipeline
                    async def one_event():
                        yield STTOutputEvent(
                            transcript=text,
                            session_id=session_id
                        )

                    try:
                        async for event in pipeline.atransform(one_event()):
                            if event.type == "ui_agent_reply":
                                result["ai_reply"] = event.text
                                logger.info(f"🤖 AI: {event.text}")
                    except Exception as e:
                        logger.error(f"Pipeline error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    logger.warning(f"Transcript too short: '{text}'")

            except Exception as e:
                logger.error(f"Transcription error: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Clean up temp file
                if os.path.exists(wav_path):
                    try:
                        os.remove(wav_path)
                    except:
                        pass

    return result