# server/elevenlabs_realtime_stt.py

import asyncio
import websockets
import json
import base64
import logging
import numpy as np
import threading

logger = logging.getLogger(__name__)

class ElevenLabsRealtimeSTT:
    """
    ElevenLabs real-time transcription using WebSocket API
    Direct callback mode - no queue polling
    """
    
    def __init__(self, api_key, language_code="en", sample_rate=16000):
        self.api_key = api_key
        self.language_code = language_code
        self.sample_rate = sample_rate
        self.audio_format = f"pcm_{sample_rate}"
        
        # WebSocket configuration - USE YOUR REGION URL
        self.websocket_url = "wss://api.in.residency.elevenlabs.io/v1/speech-to-text/realtime"
        self.model_id = "scribe_v2_realtime"
        
        # WebSocket connection
        self.websocket = None
        self.loop = None
        self.thread = None
        self.running = False
        self.connected = False
        
        # Callback for transcripts
        self.transcript_callback = None
        
        # Stats
        self.audio_chunks_sent = 0
        self.transcripts_received = 0
        self.session_id = None
        
        logger.info(f"✅ ElevenLabs Realtime STT initialized")
        logger.info(f"   URL: {self.websocket_url}")
        logger.info(f"   Language: {language_code}")
        logger.info(f"   Sample Rate: {sample_rate}")
    
    def set_transcript_callback(self, callback):
        """Set callback for receiving transcripts"""
        self.transcript_callback = callback
        logger.info(f"✅ Transcript callback set")
    
    def start(self):
        """Start the transcriber in a background thread"""
        if self.running:
            logger.warning("⚠️ Transcriber already running")
            return
        
        logger.info(f"🚀 Starting ElevenLabs realtime STT...")
        self.running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        # Wait for connection
        import time
        time.sleep(0.5)
        logger.info(f"✅ ElevenLabs STT thread started")
    
    def _run_event_loop(self):
        """Run asyncio event loop in background thread"""
        logger.info(f"🔄 STT event loop starting...")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            self.loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            logger.error(f"❌ Event loop error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            if not self.loop.is_closed():
                self.loop.close()
    
    async def _connect_and_listen(self):
        """Connect to ElevenLabs WebSocket and listen"""
        url = (
            f"{self.websocket_url}"
            f"?model_id={self.model_id}"
            f"&include_timestamps=true"
            f"&commit_strategy=vad"
            f"&language_code={self.language_code}"
            f"&audio_format={self.audio_format}"
        )
        
        try:
            headers = {"xi-api-key": self.api_key}
            
            logger.info(f"🔌 Connecting to ElevenLabs STT WebSocket...")
            
            async with websockets.connect(
                url,
                extra_headers=headers,
                ping_interval=20,
                ping_timeout=20
            ) as ws:
                self.websocket = ws
                self.connected = True
                
                logger.info(f"✅ CONNECTED TO ELEVENLABS STT WEBSOCKET")
                
                # Listen for messages
                await self._receive_loop(ws)
                        
        except Exception as e:
            logger.error(f"❌ Connection error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            self.connected = False
            logger.info(f"🔌 WebSocket connection closed")
    
    async def _receive_loop(self, websocket):
        """Receive messages and fire callbacks"""
        try:
            async for message in websocket:
                data = json.loads(message)
                message_type = data.get("message_type")
                
                if message_type == "session_started":
                    self.session_id = data.get("session_id")
                    logger.info(f"✅ Session started: {self.session_id}")
                
                elif message_type == "partial_transcript":
                    text = data.get("text", "")
                    if text:
                        logger.debug(f"📝 Partial: {text}")
                        if self.transcript_callback:
                            await self._fire_callback(text, is_final=False)
                
                elif message_type == "committed_transcript_with_timestamps":
                    # FINAL transcript
                    text = data.get("text", "")
                    
                    if text:
                        self.transcripts_received += 1
                        logger.info(f"📝 FINAL TRANSCRIPT: {text}")
                        
                        if self.transcript_callback:
                            await self._fire_callback(text, is_final=True)
                
                elif message_type == "error":
                    error_msg = data.get("message", "Unknown error")
                    logger.error(f"❌ Transcription error: {error_msg}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"⚠️ WebSocket closed by server")
        except Exception as e:
            logger.error(f"❌ Error in receive loop: {e}")
    
    async def _fire_callback(self, text, is_final):
        """Fire the transcript callback"""
        if not self.transcript_callback:
            return
        
        try:
            if asyncio.iscoroutinefunction(self.transcript_callback):
                await self.transcript_callback(text, is_final)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.transcript_callback, text, is_final)
        except Exception as e:
            logger.error(f"❌ Callback error: {e}")
    
    def stream_audio(self, audio_chunk):
        """
        Stream audio to ElevenLabs
        Args:
            audio_chunk: numpy array or bytes (int16 PCM)
        """
        if not self.running or not self.connected:
            return
        
        if not self.websocket or not self.loop:
            return
        
        # Convert to int16 PCM bytes if needed
        if isinstance(audio_chunk, np.ndarray):
            if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                audio_int16 = (audio_chunk * 32767).astype(np.int16)
            else:
                audio_int16 = audio_chunk.astype(np.int16)
            audio_bytes = audio_int16.tobytes()
        else:
            audio_bytes = audio_chunk
        
        # Schedule sending
        try:
            asyncio.run_coroutine_threadsafe(
                self._send_audio_async(audio_bytes),
                self.loop
            )
            self.audio_chunks_sent += 1
        except Exception as e:
            logger.error(f"❌ Error scheduling audio: {e}")
    
    async def _send_audio_async(self, audio_bytes):
        """Send audio to WebSocket"""
        if not self.websocket:
            return
        
        try:
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            message = {
                "message_type": "input_audio_chunk",
                "audio_base_64": audio_base64,
                "commit": False,
                "sample_rate": self.sample_rate
            }
            
            await self.websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"❌ Error sending audio: {e}")
            self.connected = False
    
    def close_sync(self):
        """Close the transcriber"""
        logger.info(f"🔌 Closing ElevenLabs STT...")
        self.running = False
        
        if self.loop and not self.loop.is_closed():
            try:
                future = asyncio.run_coroutine_threadsafe(self._close_async(), self.loop)
                future.result(timeout=2)
            except:
                pass
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)
        
        logger.info(f"✅ ElevenLabs STT closed")
    
    async def _close_async(self):
        """Async close"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass

import aiohttp
import tempfile
import os

async def transcribe_elevenlabs(audio_bytes: bytes) -> str:
    """
    One-shot ElevenLabs transcription (WAV bytes → text)
    Used by audio_processor.py
    """
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY not set")

    url = "https://api.in.residency.elevenlabs.io/v1/speech-to-text"

    headers = {
        "xi-api-key": api_key
    }

    async with aiohttp.ClientSession() as session:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            wav_path = f.name

        try:
            with open(wav_path, "rb") as audio_file:
                data = aiohttp.FormData()
                data.add_field(
                    "file",
                    audio_file,
                    filename="audio.wav",
                    content_type="audio/wav"
                )

                data.add_field("model_id", "scribe_v1")

                async with session.post(url, headers=headers, data=data) as resp:
                    if resp.status != 200:
                        raise RuntimeError(
                            f"ElevenLabs STT failed: {resp.status} {await resp.text()}"
                        )

                    result = await resp.json()
                    return result.get("text", "").strip()

        finally:
            os.remove(wav_path)
