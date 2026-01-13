# # server/elevenlabs_realtime_tts.py

# import logging
# import asyncio
# import websockets
# import json
# import base64
# import time
# from typing import Callable, Optional

# logger = logging.getLogger(__name__)

# class ElevenLabsRealtimeTTS:
#     """
#     Persistent WebSocket connection to ElevenLabs for streaming TTS
#     Connection stays open across multiple generations
#     """
    
#     def __init__(
#         self,
#         api_key: str,
#         voice_id: str,
#         sample_rate: int,
#         audio_track,  # AudioOutputTrack
#     ):
#         self.api_key = api_key
#         self.voice_id = voice_id
#         self.sample_rate = sample_rate
#         self.audio_track = audio_track
        
#         # WebSocket URL - USE YOUR REGION
#         self.base_url = "wss://api.in.residency.elevenlabs.io"
        
#         self.websocket: Optional[websockets.WebSocketClientProtocol] = None
#         self.is_connected = False
#         self.chunk_count = 0
#         self.total_bytes = 0
#         self.receive_task: Optional[asyncio.Task] = None
#         self.keepalive_task: Optional[asyncio.Task] = None
#         self._should_run = False
#         self._token_count = 0
#         self._generation_count = 0
#         self.last_send_time = None
#         self._is_generating = False
        
#         # Output format
#         if sample_rate == 22050:
#             self.output_format = "pcm_22050"
#         elif sample_rate == 24000:
#             self.output_format = "pcm_24000"
#         else:
#             self.output_format = "pcm_22050"
#             self.sample_rate = 22050
    
#     async def connect(self) -> bool:
#         """Establish persistent WebSocket connection"""
#         try:
#             url = (
#                 f"{self.base_url}/v1/text-to-speech/{self.voice_id}/stream-input"
#                 f"?model_id=eleven_turbo_v2_5"
#                 f"&output_format={self.output_format}"
#             )
            
#             logger.info(f"🔌 Connecting to ElevenLabs TTS WebSocket...")
            
#             self.websocket = await asyncio.wait_for(
#             websockets.connect(
#                 url,
#                 additional_headers={"xi-api-key": self.api_key},
#                 ping_interval=None,
#                 ping_timeout=None,
#                 open_timeout=10
#             ),

#                 timeout=15
#             )
            
#             logger.info(f"✅ Connected to ElevenLabs TTS WebSocket")
            
#             # Send BOS (Beginning of Stream)
#             bos_message = {
#                 "text": " ",
#                 "voice_settings": {
#                     "stability": 0.5,
#                     "similarity_boost": 0.8,
#                     "style": 0.0,
#                     "use_speaker_boost": True
#                 },
#                 "xi_api_key": self.api_key
#             }
#             await self.websocket.send(json.dumps(bos_message))
#             self.last_send_time = time.time()
#             logger.info(f"📤 Sent BOS message")
            
#             self.is_connected = True
#             self._should_run = True
            
#             # Start receiving audio
#             self.receive_task = asyncio.create_task(self._receive_audio_loop())
            
#             # Start keepalive
#             self.keepalive_task = asyncio.create_task(self._keepalive_loop())
            
#             return True
            
#         except Exception as e:
#             logger.error(f"❌ Failed to connect to TTS WebSocket: {e}")
#             return False
    
#     async def _keepalive_loop(self):
#         """Send keepalive messages every 15 seconds"""
#         logger.info("💓 Keepalive loop started")
        
#         while self._should_run and self.is_connected:
#             try:
#                 await asyncio.sleep(15.0)
                
#                 if not self.is_connected or not self.websocket:
#                     break
                
#                 time_since_last_send = time.time() - (self.last_send_time or 0)
                
#                 if time_since_last_send >= 15.0 and not self._is_generating:
#                     keepalive_msg = {
#                         "text": " ",
#                         "try_trigger_generation": False
#                     }
#                     await self.websocket.send(json.dumps(keepalive_msg))
#                     self.last_send_time = time.time()
#                     logger.debug("💓 Keepalive sent")
                
#             except asyncio.CancelledError:
#                 break
#             except Exception as e:
#                 logger.error(f"❌ Keepalive error: {e}")
#                 self.is_connected = False
#                 break
    
#     async def send_token(self, token: str):
#         """Send a single token for immediate TTS processing"""
#         if not self.is_connected or not self.websocket:
#             logger.warning("⚠️ Cannot send token: not connected")
#             return
        
#         try:
#             self._token_count += 1
#             self._is_generating = True
            
#             message = {
#                 "text": token,
#                 "try_trigger_generation": True
#             }
            
#             await self.websocket.send(json.dumps(message))
#             self.last_send_time = time.time()
            
#             if self._token_count <= 10:
#                 logger.debug(f"✅ Token #{self._token_count}: '{token}'")
            
#         except Exception as e:
#             logger.error(f"❌ Error sending token: {e}")
#             self.is_connected = False
    
#     async def send_eos(self):
#         """Send End of Stream for current generation"""
#         if not self.is_connected or not self.websocket:
#             return
        
#         try:
#             self._generation_count += 1
#             logger.info(f"🏁 Sending EOS for generation #{self._generation_count}")
            
#             self._token_count = 0
#             self._is_generating = False
            
#             eos_message = {
#                 "text": " ",
#                 "flush": True
#             }
            
#             await self.websocket.send(json.dumps(eos_message))
#             self.last_send_time = time.time()
#             logger.info(f"📤 EOS sent - connection stays open")
            
#         except Exception as e:
#             logger.error(f"❌ Error in send_eos: {e}")
#             self.is_connected = False
    
#     async def _receive_audio_loop(self):
#         """Background task to receive and process audio chunks"""
#         try:
#             logger.info(f"👂 Audio receive loop started")
            
#             async for message in self.websocket:
#                 if not self._should_run:
#                     break
                
#                 try:
#                     data = json.loads(message)
                    
#                     if "error" in data:
#                         error_msg = data.get("error", "Unknown error")
#                         logger.error(f"❌ ElevenLabs error: {error_msg}")
#                         continue
                    
#                     if "audio" in data:
#                         audio_data = data["audio"]
                        
#                         if not audio_data or not isinstance(audio_data, str):
#                             continue
                        
#                         try:
#                             audio_bytes = base64.b64decode(audio_data)
                            
#                             if audio_bytes and len(audio_bytes) > 0:
#                                 self.chunk_count += 1
#                                 self.total_bytes += len(audio_bytes)
                                
#                                 # Send to WebRTC audio track
#                                 self.audio_track.add_audio(audio_bytes)
                                
#                                 if self.chunk_count == 1:
#                                     logger.info(f"🎵 First audio chunk received!")
#                                 elif self.chunk_count % 50 == 0:
#                                     logger.debug(f"📤 Sent {self.chunk_count} chunks")
                        
#                         except Exception as e:
#                             logger.error(f"❌ Error processing audio: {e}")
#                             continue
                    
#                     if data.get("isFinal", False):
#                         logger.info(f"✅ Generation completed: {self.chunk_count} chunks")
#                         self.chunk_count = 0
                
#                 except json.JSONDecodeError:
#                     continue
#                 except Exception as e:
#                     logger.error(f"❌ Error in receive loop: {e}")
#                     continue
            
#         except websockets.exceptions.ConnectionClosed:
#             logger.warning(f"⚠️ TTS WebSocket closed")
#             self.is_connected = False
#         except asyncio.CancelledError:
#             logger.info(f"🛑 Receive task cancelled")
#         except Exception as e:
#             logger.error(f"❌ Error in receive loop: {e}")
#             self.is_connected = False
    
#     async def close(self):
#         """Close the WebSocket connection"""
#         logger.info("🔌 Closing TTS connection")
#         self._should_run = False
        
#         if self.keepalive_task and not self.keepalive_task.done():
#             self.keepalive_task.cancel()
#             try:
#                 await self.keepalive_task
#             except asyncio.CancelledError:
#                 pass
        
#         if self.receive_task and not self.receive_task.done():
#             self.receive_task.cancel()
#             try:
#                 await self.receive_task
#             except asyncio.CancelledError:
#                 pass
        
#         if self.websocket:
#             try:
#                 await self.websocket.send(json.dumps({"text": ""}))
#                 await self.websocket.close()
#                 logger.info(f"✅ TTS WebSocket closed")
#             except:
#                 pass
        
#         self.is_connected = False

# # =========================================================
# # Stateless helper for tts.py
# # =========================================================

# _elevenlabs_instances = {}

# def generate_speech_elevenlabs(text: str, audio_track) -> bool:
#     """
#     Wrapper used by tts_stream
#     """
#     from .config import config

#     if not config.ELEVENLABS_API_KEY:
#         logger.warning("ElevenLabs API key missing")
#         return False

#     session_key = id(audio_track)

#     if session_key not in _elevenlabs_instances:
#         tts = ElevenLabsRealtimeTTS(
#             api_key=config.ELEVENLABS_API_KEY,
#             voice_id=config.ELEVENLABS_VOICE_ID,
#             sample_rate=22050,
#             audio_track=audio_track,
#         )

#         loop = asyncio.get_event_loop()
#         loop.create_task(tts.connect())

#         _elevenlabs_instances[session_key] = tts

#     tts = _elevenlabs_instances[session_key]

#     # fire-and-forget streaming
#     asyncio.create_task(tts.send_token(text))
#     asyncio.create_task(tts.send_eos())

#     return True
# server/elevenlabs_realtime_tts.py - FIXED VERSION

import logging
import asyncio
import websockets
import json
import base64
import time
from typing import Optional

logger = logging.getLogger(__name__)

class ElevenLabsRealtimeTTS:
    """
    Persistent WebSocket connection to ElevenLabs for streaming TTS
    Connection stays open across multiple generations
    """
    
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        sample_rate: int,
        audio_track,  # AudioOutputTrack
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.audio_track = audio_track
        
        # WebSocket URL - USE YOUR REGION
        self.base_url = "wss://api.in.residency.elevenlabs.io"
        
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.chunk_count = 0
        self.total_bytes = 0
        self.receive_task: Optional[asyncio.Task] = None
        self.keepalive_task: Optional[asyncio.Task] = None
        self._should_run = False
        self._token_count = 0
        self._generation_count = 0
        self.last_send_time = None
        self._is_generating = False
        self._connection_ready = asyncio.Event()  # ✅ NEW: Wait for connection
        
        # Output format
        if sample_rate == 22050:
            self.output_format = "pcm_22050"
        elif sample_rate == 24000:
            self.output_format = "pcm_24000"
        else:
            self.output_format = "pcm_22050"
            self.sample_rate = 22050
    
    async def connect(self) -> bool:
        """Establish persistent WebSocket connection"""
        try:
            url = (
                f"{self.base_url}/v1/text-to-speech/{self.voice_id}/stream-input"
                f"?model_id=eleven_turbo_v2_5"
                f"&output_format={self.output_format}"
            )
            
            logger.info(f"🔌 Connecting to ElevenLabs TTS WebSocket...")
            
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    url,
                    additional_headers={"xi-api-key": self.api_key},
                    ping_interval=None,
                    ping_timeout=None,
                    open_timeout=10
                ),
                timeout=15
            )
            
            logger.info(f"✅ Connected to ElevenLabs TTS WebSocket")
            
            # Send BOS (Beginning of Stream)
            bos_message = {
                "text": " ",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "style": 0.0,
                    "use_speaker_boost": True
                },
                "xi_api_key": self.api_key
            }
            await self.websocket.send(json.dumps(bos_message))
            self.last_send_time = time.time()
            logger.info(f"📤 Sent BOS message")
            
            self.is_connected = True
            self._should_run = True
            
            # Start receiving audio
            self.receive_task = asyncio.create_task(self._receive_audio_loop())
            
            # Start keepalive
            self.keepalive_task = asyncio.create_task(self._keepalive_loop())
            
            # ✅ CRITICAL FIX: Signal that connection is ready
            self._connection_ready.set()
            logger.info("✅ ElevenLabs TTS ready for audio generation")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to TTS WebSocket: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def _keepalive_loop(self):
        """Send keepalive messages every 15 seconds"""
        logger.info("💓 Keepalive loop started")
        
        while self._should_run and self.is_connected:
            try:
                await asyncio.sleep(15.0)
                
                if not self.is_connected or not self.websocket:
                    break
                
                time_since_last_send = time.time() - (self.last_send_time or 0)
                
                if time_since_last_send >= 15.0 and not self._is_generating:
                    keepalive_msg = {
                        "text": " ",
                        "try_trigger_generation": False
                    }
                    await self.websocket.send(json.dumps(keepalive_msg))
                    self.last_send_time = time.time()
                    logger.debug("💓 Keepalive sent")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Keepalive error: {e}")
                self.is_connected = False
                break
    
    async def send_token(self, token: str):
        """Send a single token for immediate TTS processing"""
        # ✅ CRITICAL FIX: Wait for connection before sending
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("⏱️ Timeout waiting for ElevenLabs connection")
            return
        
        if not self.is_connected or not self.websocket:
            logger.warning("⚠️ Cannot send token: not connected")
            return
        
        try:
            self._token_count += 1
            self._is_generating = True
            
            message = {
                "text": token,
                "try_trigger_generation": True
            }
            
            await self.websocket.send(json.dumps(message))
            self.last_send_time = time.time()
            
            if self._token_count <= 10:
                logger.debug(f"✅ Token #{self._token_count}: '{token}'")
            
        except Exception as e:
            logger.error(f"❌ Error sending token: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
    
    async def send_eos(self):
        """Send End of Stream for current generation"""
        # ✅ Wait for connection
        try:
            await asyncio.wait_for(self._connection_ready.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("⏱️ Timeout waiting for ElevenLabs connection")
            return
        
        if not self.is_connected or not self.websocket:
            return
        
        try:
            self._generation_count += 1
            logger.info(f"🏁 Sending EOS for generation #{self._generation_count}")
            
            self._token_count = 0
            self._is_generating = False
            
            eos_message = {
                "text": " ",
                "flush": True
            }
            
            await self.websocket.send(json.dumps(eos_message))
            self.last_send_time = time.time()
            logger.info(f"📤 EOS sent - connection stays open")
            
        except Exception as e:
            logger.error(f"❌ Error in send_eos: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
    
    async def _receive_audio_loop(self):
        """Background task to receive and process audio chunks"""
        try:
            logger.info(f"👂 Audio receive loop started")
            
            async for message in self.websocket:
                if not self._should_run:
                    break
                
                try:
                    data = json.loads(message)
                    
                    if "error" in data:
                        error_msg = data.get("error", "Unknown error")
                        logger.error(f"❌ ElevenLabs error: {error_msg}")
                        continue
                    
                    if "audio" in data:
                        audio_data = data["audio"]
                        
                        if not audio_data or not isinstance(audio_data, str):
                            continue
                        
                        try:
                            audio_bytes = base64.b64decode(audio_data)
                            
                            if audio_bytes and len(audio_bytes) > 0:
                                self.chunk_count += 1
                                self.total_bytes += len(audio_bytes)
                                
                                # Send to WebRTC audio track
                                self.audio_track.add_audio(audio_bytes)
                                
                                if self.chunk_count == 1:
                                    logger.info(f"🎵 First audio chunk received!")
                                elif self.chunk_count % 50 == 0:
                                    logger.debug(f"📤 Sent {self.chunk_count} chunks")
                        
                        except Exception as e:
                            logger.error(f"❌ Error processing audio: {e}")
                            continue
                    
                    if data.get("isFinal", False):
                        logger.info(f"✅ Generation completed: {self.chunk_count} chunks")
                        self.chunk_count = 0
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in receive loop: {e}")
                    continue
            
        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"⚠️ TTS WebSocket closed")
            self.is_connected = False
        except asyncio.CancelledError:
            logger.info(f"🛑 Receive task cancelled")
        except Exception as e:
            logger.error(f"❌ Error in receive loop: {e}")
            import traceback
            traceback.print_exc()
            self.is_connected = False
    
    async def close(self):
        """Close the WebSocket connection"""
        logger.info("🔌 Closing TTS connection")
        self._should_run = False
        
        if self.keepalive_task and not self.keepalive_task.done():
            self.keepalive_task.cancel()
            try:
                await self.keepalive_task
            except asyncio.CancelledError:
                pass
        
        if self.receive_task and not self.receive_task.done():
            self.receive_task.cancel()
            try:
                await self.receive_task
            except asyncio.CancelledError:
                pass
        
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"text": ""}))
                await self.websocket.close()
                logger.info(f"✅ TTS WebSocket closed")
            except:
                pass
        
        self.is_connected = False

# =========================================================
# Stateless helper for tts.py
# =========================================================

_elevenlabs_instances = {}

def generate_speech_elevenlabs(text: str, audio_track) -> bool:
    """
    Wrapper used by tts_stream - FIXED to wait for connection
    """
    from .config import config

    if not config.ELEVENLABS_API_KEY:
        logger.warning("ElevenLabs API key missing")
        return False

    session_key = id(audio_track)

    # Create instance if needed
    if session_key not in _elevenlabs_instances:
        tts = ElevenLabsRealtimeTTS(
            api_key=config.ELEVENLABS_API_KEY,
            voice_id=config.ELEVENLABS_VOICE_ID,
            sample_rate=22050,
            audio_track=audio_track,
        )

        # ✅ CRITICAL FIX: Start connection task immediately
        loop = asyncio.get_event_loop()
        loop.create_task(tts.connect())

        _elevenlabs_instances[session_key] = tts
        
        logger.info(f"✅ Created new ElevenLabs TTS instance for session {session_key}")

    tts = _elevenlabs_instances[session_key]

    # ✅ CRITICAL FIX: Schedule async tasks properly
    loop = asyncio.get_event_loop()
    loop.create_task(tts.send_token(text))
    loop.create_task(tts.send_eos())

    return True