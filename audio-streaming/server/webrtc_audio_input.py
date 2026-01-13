# server/webrtc_audio_input.py

import asyncio
import numpy as np
from numpy import ndarray
import logging
from aiortc import MediaStreamTrack
from .audio_processor import process_audio_chunk

logger = logging.getLogger(__name__)

class WebRTCAudioInput:
    def __init__(self, track: MediaStreamTrack, session_id: str, pipeline, use_elevenlabs: bool):
        self.track = track
        self.session_id = session_id
        self.pipeline = pipeline
        self.use_elevenlabs = use_elevenlabs
        self.running = True

    async def run(self):
        logger.info(f"🎤 Starting WebRTC audio input for session {self.session_id}")

        try:
            while self.running:
                frame = await self.track.recv()

                audio = frame.to_ndarray()

                # ✅ Correct planar handling
                if audio.ndim == 2:
                    audio_mono = audio[0]
                else:
                    audio_mono = audio

                if audio_mono.dtype != np.int16:
                    audio_mono = np.clip(audio_mono, -1.0, 1.0)
                    audio_mono = (audio_mono * 32767).astype(np.int16)

                pcm = audio_mono.tobytes()

                await process_audio_chunk(
                    session_id=self.session_id,
                    audio_data=pcm,
                    pipeline=self.pipeline,
                    use_elevenlabs=self.use_elevenlabs,
                )

        except asyncio.CancelledError:
            logger.info(f"Audio input cancelled for session {self.session_id}")

        except Exception as e:
            logger.exception(f"Fatal WebRTC audio error: {e}")

        finally:
            logger.info(f"🛑 WebRTC audio input stopped for session {self.session_id}")

    def stop(self):
        self.running = False
