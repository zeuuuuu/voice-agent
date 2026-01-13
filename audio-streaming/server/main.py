# # server/main.py

# import os
# import logging
# import asyncio
# from pathlib import Path

# from fastapi import FastAPI
# from fastapi.responses import FileResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
# from aiortc.mediastreams import MediaStreamError

# from dotenv import load_dotenv
# import shutil
# import numpy as np
# from .call_recorder import CallRecorder
# from .audio_processor import init_session, cleanup_session, process_audio_chunk
# from .webrtc import sessions
# from .output_track import AudioOutputTrack
# from .pipeline import create_voice_pipeline
# from .config import config

# # -------------------------------------------------
# # Logging
# # -------------------------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# # -------------------------------------------------
# # Piper TTS Model Setup
# # -------------------------------------------------
# PIPER_MODEL_LOCATIONS = [
#     Path.home() / ".local/share/piper/models/en_US-lessac-medium.onnx",
#     Path("/usr/share/piper/models/en_US-lessac-medium.onnx"),
#     Path("./models/en_US-lessac-medium.onnx"),
# ]

# PIPER_MODEL = None
# piper_path = shutil.which("piper")

# for model_path in PIPER_MODEL_LOCATIONS:
#     if model_path.exists() and piper_path:
#         PIPER_MODEL = str(model_path)
#         logger.info(f"Found Piper model at: {PIPER_MODEL}")
#         break

# if not PIPER_MODEL:
#     logger.warning("⚠️ Piper TTS not available")

# # Store in config for access by other modules
# config.PIPER_MODEL = PIPER_MODEL

# # -------------------------------------------------
# # ElevenLabs Validation
# # -------------------------------------------------
# if config.ELEVENLABS_API_KEY:
#     logger.info("✅ ElevenLabs API key configured")
# else:
#     logger.warning("⚠️ ElevenLabs API key not found in environment")

# # -------------------------------------------------
# # FastAPI App Setup
# # -------------------------------------------------
# app = FastAPI(title="Real-Time Voice AI Assistant")

# BASE_DIR = Path(__file__).parent.parent
# STATIC_DIR = BASE_DIR / "static"

# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def root():
#     return FileResponse(STATIC_DIR / "index.html")

# # -------------------------------------------------
# # Session Preferences Storage
# # -------------------------------------------------
# session_preferences = {}  # {session_id: {"use_elevenlabs": bool}}

# # -------------------------------------------------
# # WebRTC Offer / Answer
# # -------------------------------------------------
# class Offer(BaseModel):
#     sdp: str
#     type: str
#     use_elevenlabs: bool = False  # ✅ NEW

# @app.post("/offer")
# async def offer(offer: Offer):
#     """
#     Full-duplex WebRTC:
#     Browser → mic audio
#     Server → TTS audio
#     """
#     session_id = os.urandom(16).hex()
#     pc = RTCPeerConnection()
#     sessions[session_id] = pc

#     # ✅ Store user preference
#     use_elevenlabs = offer.use_elevenlabs
#     session_preferences[session_id] = {"use_elevenlabs": use_elevenlabs}
    
#     provider = "ElevenLabs" if use_elevenlabs else "Whisper/Piper"
#     logger.info(f"Creating session {session_id} (Provider: {provider})")

#     # Init STT buffers
#     init_session(session_id)

#     # Audio output (Server → Browser)
#     audio_output_track = AudioOutputTrack()
#     pc.addTrack(audio_output_track)
#     sessions[f"{session_id}_tts"] = audio_output_track

#     # ✅ AI Pipeline with provider selection
#     pipeline = create_voice_pipeline(
#         audio_output_track, 
#         PIPER_MODEL,
#         use_elevenlabs=use_elevenlabs
#     )
#     sessions[f"{session_id}_pipeline"] = pipeline

#     # Debug: Connection state monitoring
#     @pc.on("connectionstatechange")
#     async def on_connection_state_change():
#         logger.info(f"[{session_id}] Connection state: {pc.connectionState}")

#     @pc.on("iceconnectionstatechange")
#     async def on_ice_state_change():
#         logger.info(f"[{session_id}] ICE state: {pc.iceConnectionState}")

#     # Audio input (Browser → Server)
#     @pc.on("track")
#     def on_track(track: MediaStreamTrack):
#         logger.info(f"[{session_id}] Received track: kind={track.kind}, id={track.id}")
        
#         if track.kind != "audio":
#             logger.warning(f"[{session_id}] Ignoring non-audio track")
#             return

#         async def consume_audio():
#             frame_count = 0
#             try:
#                 logger.info(f"[{session_id}] 🎤 Audio consumer started")
                
#                 while True:
#                     try:
#                         frame = await track.recv()
#                         frame_count += 1
                        
#                         if frame_count == 1:
#                             logger.info(f"First frame: format={frame.format}, "
#                                         f"sample_rate={frame.sample_rate}, "
#                                         f"samples={frame.samples}, "
#                                         f"layout={frame.layout}")

#                     except MediaStreamError:
#                         logger.info(f"[{session_id}] MediaStream ended → flushing audio")
                        
#                         try:
#                             await process_audio_chunk(
#                                 session_id=session_id,
#                                 audio_data=b"",
#                                 pipeline=pipeline,
#                                 use_elevenlabs=use_elevenlabs
#                             )
#                         except Exception as e:
#                             logger.error(f"Flush error: {e}")
                        
#                         break
                        
#                     except Exception as e:
#                         logger.error(f"[{session_id}] Frame recv error: {e}")
#                         import traceback
#                         traceback.print_exc()
#                         break

#                     try:
#                         # Get raw audio array
#                         audio_array = frame.to_ndarray()
                        
#                         if frame_count == 1:
#                             logger.info(f"🔍 Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
#                             logger.info(f"🔍 Frame.samples: {frame.samples}, Frame.layout: {frame.layout}")
                        
#                         # ✅ CRITICAL FIX: Handle PyAV's interleaved stereo format
#                         # PyAV with stereo layout returns shape (1, samples*channels) where data is interleaved
#                         # For stereo 960 samples: shape is (1, 1920) with [L,R,L,R,L,R,...]
                        
#                         if len(audio_array.shape) == 2 and audio_array.shape[0] == 1:
#                             # Extract the audio data
#                             audio_data = audio_array[0]
                            
#                             if frame_count == 1:
#                                 logger.info(f"🔍 Extracted audio_data shape: {audio_data.shape}")
                            
#                             # Check if this is interleaved stereo
#                             expected_samples = frame.samples
#                             actual_samples = audio_data.shape[0]
                            
#                             if actual_samples == expected_samples * 2:
#                                 # ✅ This is interleaved stereo: [L, R, L, R, ...]
#                                 # Separate into left and right channels
#                                 left_channel = audio_data[0::2]   # Every even index
#                                 right_channel = audio_data[1::2]  # Every odd index
                                
#                                 # Average to mono
#                                 if audio_data.dtype == np.int16:
#                                     # Convert to int32 first to avoid overflow
#                                     audio_mono = ((left_channel.astype(np.int32) + right_channel.astype(np.int32)) // 2).astype(np.int16)
#                                 else:
#                                     audio_mono = (left_channel + right_channel) / 2.0
                                
#                                 if frame_count == 1:
#                                     logger.info(f"✅ De-interleaved stereo → mono: {len(audio_mono)} samples")
#                                     logger.info(f"   Left channel range: [{left_channel.min()}, {left_channel.max()}]")
#                                     logger.info(f"   Right channel range: [{right_channel.min()}, {right_channel.max()}]")
#                                     logger.info(f"   Mono range: [{audio_mono.min()}, {audio_mono.max()}]")
                            
#                             elif actual_samples == expected_samples:
#                                 # Already mono
#                                 audio_mono = audio_data
#                                 if frame_count == 1:
#                                     logger.info(f"✅ Already mono: {len(audio_mono)} samples")
                            
#                             else:
#                                 logger.error(f"Unexpected sample count: expected {expected_samples}, got {actual_samples}")
#                                 continue
                        
#                         elif len(audio_array.shape) == 2:
#                             # Planar format: (channels, samples)
#                             if audio_array.shape[0] == 2:
#                                 # Stereo planar
#                                 audio_mono = (audio_array[0] + audio_array[1]) / 2.0
#                                 if frame_count == 1:
#                                     logger.info(f"✅ Planar stereo → mono")
#                             else:
#                                 audio_mono = audio_array[0]
#                                 if frame_count == 1:
#                                     logger.info(f"✅ Planar mono")
                        
#                         elif len(audio_array.shape) == 1:
#                             # Already 1D
#                             audio_mono = audio_array
#                             if frame_count == 1:
#                                 logger.info(f"✅ 1D mono")
                        
#                         else:
#                             logger.error(f"Unexpected audio shape: {audio_array.shape}")
#                             continue
                        
#                         # ✅ Convert to PCM bytes
#                         if audio_mono.dtype == np.int16:
#                             pcm = audio_mono.tobytes()
#                         elif audio_mono.dtype in [np.float32, np.float64]:
#                             audio_clipped = np.clip(audio_mono, -1.0, 1.0)
#                             audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
#                             pcm = audio_int16.tobytes()
#                         else:
#                             logger.error(f"Unsupported dtype: {audio_mono.dtype}")
#                             continue
                        
#                         if frame_count == 1:
#                             logger.info(f"✅ PCM: {len(pcm)} bytes")
                            
#                             # Verify the PCM data
#                             test_audio_int = np.frombuffer(pcm, dtype=np.int16)
#                             test_audio_float = test_audio_int.astype(np.float32) / 32767.0
                            
#                             logger.info(f"✅ Audio verification:")
#                             logger.info(f"   Samples: {len(test_audio_int)}")
#                             logger.info(f"   Int16 range: [{test_audio_int.min()}, {test_audio_int.max()}]")
#                             logger.info(f"   Float range: [{test_audio_float.min():.3f}, {test_audio_float.max():.3f}]")
#                             logger.info(f"   Max amplitude: {np.abs(test_audio_float).max():.3f}")
#                             logger.info(f"   RMS: {np.sqrt(np.mean(test_audio_float**2)):.3f}")
#                             logger.info(f"   Non-zero: {np.count_nonzero(test_audio_int)}/{len(test_audio_int)}")
                            
#                             # Sample the first few values
#                             logger.info(f"   First 20 samples: {test_audio_int[:20].tolist()}")
                        
#                         # Log periodically
#                         if frame_count % 100 == 0:
#                             test_audio_int = np.frombuffer(pcm, dtype=np.int16)
#                             test_audio_float = test_audio_int.astype(np.float32) / 32767.0
#                             logger.info(f"Frame {frame_count}: max={np.abs(test_audio_float).max():.3f}, "
#                                     f"rms={np.sqrt(np.mean(test_audio_float**2)):.3f}")

#                         # Get session preference
#                         use_elevenlabs = session_preferences.get(session_id, {}).get("use_elevenlabs", False)

#                         await process_audio_chunk(
#                             session_id=session_id,
#                             audio_data=pcm,
#                             pipeline=pipeline,
#                             use_elevenlabs=use_elevenlabs
#                         )

#                     except Exception as e:
#                         logger.error(f"[{session_id}] Processing error: {e}")
#                         import traceback
#                         traceback.print_exc()

#             except asyncio.CancelledError:
#                 logger.info(f"[{session_id}] Audio consumer cancelled")

#             except Exception as e:
#                 logger.exception(f"[{session_id}] Fatal audio error")

#             finally:
#                 logger.info(f"[{session_id}] 🛑 Audio consumer stopped ({frame_count} frames)")

#         # Start consumer
#         asyncio.create_task(consume_audio())
#     # SDP Exchange
#     await pc.setRemoteDescription(
#         RTCSessionDescription(sdp=offer.sdp, type=offer.type)
#     )

#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     logger.info(f"[{session_id}] Session ready")

#     return {
#         "session_id": session_id,
#         "sdp": pc.localDescription.sdp,
#         "type": pc.localDescription.type,
#     }

# # -------------------------------------------------
# # End Session
# # -------------------------------------------------
# @app.post("/end/{session_id}")
# async def end(session_id: str):
#     pc = sessions.pop(session_id, None)
#     tts_track = sessions.pop(f"{session_id}_tts", None)
#     sessions.pop(f"{session_id}_pipeline", None)
#     session_preferences.pop(session_id, None)  # ✅ Clean up preference

#     if tts_track:
#         tts_track.reset()

#     if pc:
#         await pc.close()

#     cleanup_session(session_id)
#     logger.info(f"Session {session_id} ended")

#     return {"status": "closed"}

# server/main.py

# import os
# import logging
# import asyncio
# from pathlib import Path

# from fastapi import FastAPI
# from fastapi.responses import FileResponse, HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
# from aiortc.mediastreams import MediaStreamError

# from dotenv import load_dotenv
# import shutil
# import numpy as np
# from .call_recorder import CallRecorder
# from .audio_processor import init_session, cleanup_session, process_audio_chunk
# from .webrtc import sessions
# from .output_track import AudioOutputTrack
# from .pipeline import create_voice_pipeline
# from .config import config

# # -------------------------------------------------
# # Logging
# # -------------------------------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# # -------------------------------------------------
# # Piper TTS Model Setup
# # -------------------------------------------------
# PIPER_MODEL_LOCATIONS = [
#     Path.home() / ".local/share/piper/models/en_US-lessac-medium.onnx",
#     Path("/usr/share/piper/models/en_US-lessac-medium.onnx"),
#     Path("./models/en_US-lessac-medium.onnx"),
# ]

# PIPER_MODEL = None
# piper_path = shutil.which("piper")

# for model_path in PIPER_MODEL_LOCATIONS:
#     if model_path.exists() and piper_path:
#         PIPER_MODEL = str(model_path)
#         logger.info(f"Found Piper model at: {PIPER_MODEL}")
#         break

# if not PIPER_MODEL:
#     logger.warning("⚠️ Piper TTS not available")

# # Store in config for access by other modules
# config.PIPER_MODEL = PIPER_MODEL

# # -------------------------------------------------
# # ElevenLabs Validation
# # -------------------------------------------------
# if config.ELEVENLABS_API_KEY:
#     logger.info("✅ ElevenLabs API key configured")
# else:
#     logger.warning("⚠️ ElevenLabs API key not found in environment")

# # -------------------------------------------------
# # FastAPI App Setup
# # -------------------------------------------------
# app = FastAPI(title="Real-Time Voice AI Assistant")

# BASE_DIR = Path(__file__).parent.parent
# STATIC_DIR = BASE_DIR / "static"

# app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def root():
#     return FileResponse(STATIC_DIR / "index.html")

# # -------------------------------------------------
# # Session Preferences Storage
# # -------------------------------------------------
# session_preferences = {}  # {session_id: {"use_elevenlabs": bool}}

# # -------------------------------------------------
# # WebRTC Offer / Answer
# # -------------------------------------------------
# class Offer(BaseModel):
#     sdp: str
#     type: str
#     use_elevenlabs: bool = False  # ✅ NEW

# @app.post("/offer")
# async def offer(offer: Offer):
#     """
#     Full-duplex WebRTC:
#     Browser → mic audio
#     Server → TTS audio
#     """
#     session_id = os.urandom(16).hex()
#     pc = RTCPeerConnection()
#     sessions[session_id] = pc

#     # ✅ Store user preference
#     use_elevenlabs = offer.use_elevenlabs
#     session_preferences[session_id] = {"use_elevenlabs": use_elevenlabs}
    
#     provider = "ElevenLabs" if use_elevenlabs else "Whisper/Piper"
#     logger.info(f"Creating session {session_id} (Provider: {provider})")

#     # Init STT buffers
#     init_session(session_id)

#     # ✅ NEW: Initialize call recorder for this session
#     recorder = CallRecorder(session_id=session_id, sample_rate=48000)
#     sessions[f"{session_id}_recorder"] = recorder

#     # Audio output (Server → Browser)
#     audio_output_track = AudioOutputTrack()
#     pc.addTrack(audio_output_track)
#     sessions[f"{session_id}_tts"] = audio_output_track

#     # ✅ AI Pipeline with provider selection
#     pipeline = create_voice_pipeline(
#         audio_output_track, 
#         PIPER_MODEL,
#         use_elevenlabs=use_elevenlabs
#     )
#     sessions[f"{session_id}_pipeline"] = pipeline

#     # Debug: Connection state monitoring
#     @pc.on("connectionstatechange")
#     async def on_connection_state_change():
#         logger.info(f"[{session_id}] Connection state: {pc.connectionState}")

#     @pc.on("iceconnectionstatechange")
#     async def on_ice_state_change():
#         logger.info(f"[{session_id}] ICE state: {pc.iceConnectionState}")

#     # Audio input (Browser → Server)
#     @pc.on("track")
#     def on_track(track: MediaStreamTrack):
#         logger.info(f"[{session_id}] Received track: kind={track.kind}, id={track.id}")
        
#         if track.kind != "audio":
#             logger.warning(f"[{session_id}] Ignoring non-audio track")
#             return

#         async def consume_audio():
#             frame_count = 0
#             try:
#                 logger.info(f"[{session_id}] 🎤 Audio consumer started")
                
#                 while True:
#                     try:
#                         frame = await track.recv()
#                         frame_count += 1
                        
#                         if frame_count == 1:
#                             logger.info(f"First frame: format={frame.format}, "
#                                         f"sample_rate={frame.sample_rate}, "
#                                         f"samples={frame.samples}, "
#                                         f"layout={frame.layout}")

#                     except MediaStreamError:
#                         logger.info(f"[{session_id}] MediaStream ended → flushing audio")
                        
#                         try:
#                             await process_audio_chunk(
#                                 session_id=session_id,
#                                 audio_data=b"",
#                                 pipeline=pipeline,
#                                 use_elevenlabs=use_elevenlabs
#                             )
#                         except Exception as e:
#                             logger.error(f"Flush error: {e}")
                        
#                         break
                        
#                     except Exception as e:
#                         logger.error(f"[{session_id}] Frame recv error: {e}")
#                         import traceback
#                         traceback.print_exc()
#                         break

#                     try:
#                         # Get raw audio array
#                         audio_array = frame.to_ndarray()
                        
#                         if frame_count == 1:
#                             logger.info(f"🔍 Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
#                             logger.info(f"🔍 Frame.samples: {frame.samples}, Frame.layout: {frame.layout}")
                        
#                         # ✅ CRITICAL FIX: Handle PyAV's interleaved stereo format
#                         if len(audio_array.shape) == 2 and audio_array.shape[0] == 1:
#                             audio_data = audio_array[0]
                            
#                             if frame_count == 1:
#                                 logger.info(f"🔍 Extracted audio_data shape: {audio_data.shape}")
                            
#                             expected_samples = frame.samples
#                             actual_samples = audio_data.shape[0]
                            
#                             if actual_samples == expected_samples * 2:
#                                 left_channel = audio_data[0::2]
#                                 right_channel = audio_data[1::2]
                                
#                                 if audio_data.dtype == np.int16:
#                                     audio_mono = ((left_channel.astype(np.int32) + right_channel.astype(np.int32)) // 2).astype(np.int16)
#                                 else:
#                                     audio_mono = (left_channel + right_channel) / 2.0
                                
#                                 if frame_count == 1:
#                                     logger.info(f"✅ De-interleaved stereo → mono: {len(audio_mono)} samples")
                            
#                             elif actual_samples == expected_samples:
#                                 audio_mono = audio_data
#                                 if frame_count == 1:
#                                     logger.info(f"✅ Already mono: {len(audio_mono)} samples")
                            
#                             else:
#                                 logger.error(f"Unexpected sample count: expected {expected_samples}, got {actual_samples}")
#                                 continue
                        
#                         elif len(audio_array.shape) == 2:
#                             if audio_array.shape[0] == 2:
#                                 audio_mono = (audio_array[0] + audio_array[1]) / 2.0
#                                 if frame_count == 1:
#                                     logger.info(f"✅ Planar stereo → mono")
#                             else:
#                                 audio_mono = audio_array[0]
#                                 if frame_count == 1:
#                                     logger.info(f"✅ Planar mono")
                        
#                         elif len(audio_array.shape) == 1:
#                             audio_mono = audio_array
#                             if frame_count == 1:
#                                 logger.info(f"✅ 1D mono")
                        
#                         else:
#                             logger.error(f"Unexpected audio shape: {audio_array.shape}")
#                             continue
                        
#                         # ✅ Convert to PCM bytes (int16)
#                         if audio_mono.dtype == np.int16:
#                             pcm = audio_mono.tobytes()
#                         elif audio_mono.dtype in [np.float32, np.float64]:
#                             audio_clipped = np.clip(audio_mono, -1.0, 1.0)
#                             audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
#                             pcm = audio_int16.tobytes()
#                         else:
#                             logger.error(f"Unsupported dtype: {audio_mono.dtype}")
#                             continue
                        
#                         # ✅ RECORD THE USER AUDIO
#                         recorder.add_audio(pcm)

#                         if frame_count == 1:
#                             logger.info(f"✅ PCM: {len(pcm)} bytes")
                            
#                             test_audio_int = np.frombuffer(pcm, dtype=np.int16)
#                             test_audio_float = test_audio_int.astype(np.float32) / 32767.0
                            
#                             logger.info(f"✅ Audio verification:")
#                             logger.info(f"   Samples: {len(test_audio_int)}")
#                             logger.info(f"   Int16 range: [{test_audio_int.min()}, {test_audio_int.max()}]")
#                             logger.info(f"   Float range: [{test_audio_float.min():.3f}, {test_audio_float.max():.3f}]")
#                             logger.info(f"   Max amplitude: {np.abs(test_audio_float).max():.3f}")
#                             logger.info(f"   RMS: {np.sqrt(np.mean(test_audio_float**2)):.3f}")
#                             logger.info(f"   Non-zero: {np.count_nonzero(test_audio_int)}/{len(test_audio_int)}")
#                             logger.info(f"   First 20 samples: {test_audio_int[:20].tolist()}")
                        
#                         if frame_count % 100 == 0:
#                             test_audio_int = np.frombuffer(pcm, dtype=np.int16)
#                             test_audio_float = test_audio_int.astype(np.float32) / 32767.0
#                             logger.info(f"Frame {frame_count}: max={np.abs(test_audio_float).max():.3f}, "
#                                     f"rms={np.sqrt(np.mean(test_audio_float**2)):.3f}")

#                         # Get session preference (in case it changed, though unlikely)
#                         use_elevenlabs = session_preferences.get(session_id, {}).get("use_elevenlabs", False)

#                         await process_audio_chunk(
#                             session_id=session_id,
#                             audio_data=pcm,
#                             pipeline=pipeline,
#                             use_elevenlabs=use_elevenlabs
#                         )

#                     except Exception as e:
#                         logger.error(f"[{session_id}] Processing error: {e}")
#                         import traceback
#                         traceback.print_exc()

#             except asyncio.CancelledError:
#                 logger.info(f"[{session_id}] Audio consumer cancelled")

#             except Exception as e:
#                 logger.exception(f"[{session_id}] Fatal audio error")

#             finally:
#                 logger.info(f"[{session_id}] 🛑 Audio consumer stopped ({frame_count} frames)")

#         # Start consumer
#         asyncio.create_task(consume_audio())

#     # SDP Exchange
#     await pc.setRemoteDescription(
#         RTCSessionDescription(sdp=offer.sdp, type=offer.type)
#     )

#     answer = await pc.createAnswer()
#     await pc.setLocalDescription(answer)

#     logger.info(f"[{session_id}] Session ready")

#     return {
#         "session_id": session_id,
#         "sdp": pc.localDescription.sdp,
#         "type": pc.localDescription.type,
#     }

# # -------------------------------------------------
# # End Session
# # -------------------------------------------------
# @app.post("/end/{session_id}")
# async def end(session_id: str):
#     pc = sessions.pop(session_id, None)
#     tts_track = sessions.pop(f"{session_id}_tts", None)
#     pipeline = sessions.pop(f"{session_id}_pipeline", None)
    
#     # ✅ CRITICAL: Pop recorder BEFORE anything else that might fail
#     recorder = sessions.pop(f"{session_id}_recorder", None)
    
#     session_preferences.pop(session_id, None)

#     # Finalize and save recording
#     if recorder:
#         saved_path = recorder.finalize()
#         if saved_path:
#             logger.info(f"📼 Call recording saved: {saved_path}")
#         else:
#             logger.warning("No audio frames were recorded (empty call)")
#     else:
#         logger.warning("Recorder not found – possible early cleanup bug")

#     if tts_track:
#         tts_track.reset()

#     if pc:
#         await pc.close()

#     cleanup_session(session_id)
#     logger.info(f"Session {session_id} ended")

#     return {"status": "closed"}
# server/main.py

import os
import logging
import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from dotenv import load_dotenv
import shutil
import numpy as np
from .call_recorder import CallRecorder
from .audio_processor import init_session, cleanup_session, process_audio_chunk
from .webrtc import sessions
from .output_track import AudioOutputTrack
from .pipeline import create_voice_pipeline
from .config import config

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# -------------------------------------------------
# Piper TTS Model Setup
# -------------------------------------------------
PIPER_MODEL_LOCATIONS = [
    Path.home() / ".local/share/piper/models/en_US-lessac-medium.onnx",
    Path("/usr/share/piper/models/en_US-lessac-medium.onnx"),
    Path("./models/en_US-lessac-medium.onnx"),
]

PIPER_MODEL = None
piper_path = shutil.which("piper")

for model_path in PIPER_MODEL_LOCATIONS:
    if model_path.exists() and piper_path:
        PIPER_MODEL = str(model_path)
        logger.info(f"Found Piper model at: {PIPER_MODEL}")
        break

if not PIPER_MODEL:
    logger.warning("⚠️ Piper TTS not available")

config.PIPER_MODEL = PIPER_MODEL

# -------------------------------------------------
# ElevenLabs Validation
# -------------------------------------------------
if config.ELEVENLABS_API_KEY:
    logger.info("✅ ElevenLabs API key configured")
else:
    logger.warning("⚠️ ElevenLabs API key not found in environment")

# -------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------
app = FastAPI(title="Real-Time Voice AI Assistant")

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(STATIC_DIR / "index.html")

# -------------------------------------------------
# Session Preferences Storage
# -------------------------------------------------
session_preferences = {}

# -------------------------------------------------
# WebRTC Offer / Answer
# -------------------------------------------------
class Offer(BaseModel):
    sdp: str
    type: str
    use_elevenlabs: bool = False

@app.post("/offer")
async def offer(offer: Offer):
    session_id = os.urandom(16).hex()
    pc = RTCPeerConnection()
    sessions[session_id] = pc

    use_elevenlabs = offer.use_elevenlabs
    session_preferences[session_id] = {"use_elevenlabs": use_elevenlabs}
    
    provider = "ElevenLabs" if use_elevenlabs else "Whisper/Piper"
    logger.info(f"Creating session {session_id} (Provider: {provider})")

    init_session(session_id)

    recorder = CallRecorder(session_id=session_id, sample_rate=48000)
    sessions[f"{session_id}_recorder"] = recorder

    audio_output_track = AudioOutputTrack(session_id=session_id)
    pc.addTrack(audio_output_track)
    sessions[f"{session_id}_tts"] = audio_output_track

    pipeline = create_voice_pipeline(
        audio_output_track, 
        PIPER_MODEL,
        use_elevenlabs=use_elevenlabs
    )
    sessions[f"{session_id}_pipeline"] = pipeline

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        logger.info(f"[{session_id}] Connection state: {pc.connectionState}")

    @pc.on("iceconnectionstatechange")
    async def on_ice_state_change():
        logger.info(f"[{session_id}] ICE state: {pc.iceConnectionState}")

    @pc.on("track")
    def on_track(track: MediaStreamTrack):
        logger.info(f"[{session_id}] Received track: kind={track.kind}, id={track.id}")
        
        if track.kind != "audio":
            logger.warning(f"[{session_id}] Ignoring non-audio track")
            return

        async def consume_audio():
            frame_count = 0
            # Capture variables from outer scope safely
            recorder = sessions.get(f"{session_id}_recorder")
            current_use_elevenlabs = use_elevenlabs

            try:
                logger.info(f"[{session_id}] 🎤 Audio consumer started")
                
                while True:
                    try:
                        frame = await track.recv()
                        frame_count += 1
                        
                        if frame_count == 1:
                            logger.info(f"First frame: format={frame.format}, "
                                        f"sample_rate={frame.sample_rate}, "
                                        f"samples={frame.samples}, "
                                        f"layout={frame.layout}")

                    except MediaStreamError:
                        logger.info(f"[{session_id}] MediaStream ended → flushing audio")
                        try:
                            await process_audio_chunk(
                                session_id=session_id,
                                audio_data=b"",
                                pipeline=pipeline,
                                use_elevenlabs=current_use_elevenlabs
                            )
                        except Exception as e:
                            logger.error(f"Flush error: {e}")
                        break
                        
                    except Exception as e:
                        logger.error(f"[{session_id}] Frame recv error: {e}")
                        import traceback
                        traceback.print_exc()
                        break

                    try:
                        # Safety: stop if session was cleaned up
                        if session_id not in sessions:
                            logger.info(f"[{session_id}] Session ended externally — stopping processing")
                            break

                        if not recorder:
                            logger.warning(f"[{session_id}] Recorder missing — cannot save audio")
                            break

                        audio_array = frame.to_ndarray()
                        
                        if frame_count == 1:
                            logger.info(f"🔍 Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
                            logger.info(f"🔍 Frame.samples: {frame.samples}, Frame.layout: {frame.layout}")
                        
                        # Stereo → Mono conversion (unchanged)
                        if len(audio_array.shape) == 2 and audio_array.shape[0] == 1:
                            audio_data = audio_array[0]
                            expected_samples = frame.samples
                            actual_samples = audio_data.shape[0]
                            
                            if actual_samples == expected_samples * 2:
                                left = audio_data[0::2]
                                right = audio_data[1::2]
                                if audio_data.dtype == np.int16:
                                    audio_mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
                                else:
                                    audio_mono = (left + right) / 2.0
                            elif actual_samples == expected_samples:
                                audio_mono = audio_data
                            else:
                                logger.error(f"Unexpected sample count: expected {expected_samples}, got {actual_samples}")
                                continue
                        elif len(audio_array.shape) == 2:
                            audio_mono = (audio_array[0] + audio_array[1]) / 2.0 if audio_array.shape[0] == 2 else audio_array[0]
                        elif len(audio_array.shape) == 1:
                            audio_mono = audio_array
                        else:
                            logger.error(f"Unexpected audio shape: {audio_array.shape}")
                            continue
                        
                        # Convert to PCM int16
                        if audio_mono.dtype == np.int16:
                            pcm = audio_mono.tobytes()
                        elif audio_mono.dtype in [np.float32, np.float64]:
                            audio_clipped = np.clip(audio_mono, -1.0, 1.0)
                            audio_int16 = (audio_clipped * 32767.0).astype(np.int16)
                            pcm = audio_int16.tobytes()
                        else:
                            logger.error(f"Unsupported dtype: {audio_mono.dtype}")
                            continue
                        
                        # Record user audio
                        recorder.add_user_audio(pcm)

                        # Debug logging (first frame + every 100)
                        if frame_count == 1:
                            logger.info(f"✅ PCM: {len(pcm)} bytes")
                            test_int = np.frombuffer(pcm, dtype=np.int16)
                            test_float = test_int.astype(np.float32) / 32767.0
                            logger.info(f"✅ Audio verification: samples={len(test_int)}, "
                                        f"range=[{test_int.min()}, {test_int.max()}], "
                                        f"max_amp={np.abs(test_float).max():.3f}, rms={np.sqrt(np.mean(test_float**2)):.3f}")

                        if frame_count % 100 == 0:
                            test_float = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                            logger.info(f"Frame {frame_count}: max={np.abs(test_float).max():.3f}, rms={np.sqrt(np.mean(test_float**2)):.3f}")

                        await process_audio_chunk(
                            session_id=session_id,
                            audio_data=pcm,
                            pipeline=pipeline,
                            use_elevenlabs=current_use_elevenlabs
                        )

                    except Exception as e:
                        logger.error(f"[{session_id}] Processing error: {e}")
                        import traceback
                        traceback.print_exc()

            except asyncio.CancelledError:
                logger.info(f"[{session_id}] Audio consumer cancelled")

            except Exception as e:
                logger.exception(f"[{session_id}] Fatal audio error")

            finally:
                logger.info(f"[{session_id}] 🛑 Audio consumer stopped ({frame_count} frames)")

                fallback_recorder = sessions.get(f"{session_id}_recorder")
                if fallback_recorder:
                    has_user = fallback_recorder.user_frames if hasattr(fallback_recorder, 'user_frames') else []
                    has_ai = fallback_recorder.ai_frames if hasattr(fallback_recorder, 'ai_frames') else []
                    if has_user or has_ai:
                        saved_path = fallback_recorder.finalize()
                        if saved_path:
                            logger.info(f"📼 Auto-saved full recording (fallback): {saved_path}")
                    else:
                        logger.info("No audio (user or AI) to save in fallback")
                    sessions.pop(f"{session_id}_recorder", None)

                # Fallback auto-save if /end wasn't called properly
                # fallback_recorder = sessions.get(f"{session_id}_recorder")
                # if fallback_recorder:
                #     # Safe check: only if it has recorded data
                #     if hasattr(fallback_recorder, 'frames') and fallback_recorder.frames:
                #         saved_path = fallback_recorder.finalize()
                #         if saved_path:
                #             logger.info(f"📼 Auto-saved recording (fallback): {saved_path}")
                #     elif not hasattr(fallback_recorder, 'frames') or len(fallback_recorder.frames) == 0:
                #         logger.info("No audio to save in fallback")
                #     sessions.pop(f"{session_id}_recorder", None)

        asyncio.create_task(consume_audio())

    await pc.setRemoteDescription(RTCSessionDescription(sdp=offer.sdp, type=offer.type))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info(f"[{session_id}] Session ready")

    return {
        "session_id": session_id,
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
    }

# -------------------------------------------------
# End Session
# -------------------------------------------------
@app.post("/end/{session_id}")
async def end(session_id: str):
    pc = sessions.pop(session_id, None)
    tts_track = sessions.pop(f"{session_id}_tts", None)
    pipeline = sessions.pop(f"{session_id}_pipeline", None)
    recorder = sessions.pop(f"{session_id}_recorder", None)
    session_preferences.pop(session_id, None)

    if recorder:
        saved_path = recorder.finalize()
        if saved_path:
            logger.info(f"📼 Call recording saved: {saved_path}")
        else:
            logger.warning("No audio recorded or finalize returned None")

    if tts_track:
        tts_track.reset()

    if pc:
        await pc.close()

    cleanup_session(session_id)
    logger.info(f"Session {session_id} ended")

    return {"status": "closed"}