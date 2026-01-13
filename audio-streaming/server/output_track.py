# server/output_track.py - LARGER QUEUE + BACKPRESSURE SUPPORT

import logging
import asyncio
from aiortc import MediaStreamTrack, MediaStreamError
from av import AudioFrame
import numpy as np

logger = logging.getLogger(__name__)

class AudioOutputTrack(MediaStreamTrack):
    """48kHz output with large queue to prevent drops"""
    kind = "audio"

    def __init__(self, session_id=None):
        super().__init__()
        self.session_id = session_id
        
        # 48kHz config
        self.sample_rate = 48000
        self.samples_per_frame = 960
        self.bytes_per_frame = 1920
        
        # ✅ CRITICAL: Larger queue to handle bursts without dropping
        # 500 frames = ~10 seconds of audio at 48kHz
        self.audio_queue = asyncio.Queue(maxsize=500)
        
        # Partial buffer
        self._partial_buffer = bytearray()
        
        # ✅ REDUCED: Pre-buffer only ONCE (at the very start) - LARGER buffer for gapless playback
        self._min_frames_to_start = 25  # 500ms initial buffer (increased from 10)
        self._is_playing = False
        self._ever_started = False
        
        # Stats
        self._frame_count = 0
        self._chunks_added = 0
        self._stopped = False
        self._frames_dropped = 0
        
        logger.info(f"🎵 AudioOutputTrack: {self.sample_rate}Hz, queue_size=500 (~10s buffer)")

    def add_audio(self, pcm_bytes: bytes):
        """Add 48kHz audio"""
        if self._stopped or not pcm_bytes:
            return
        
        try:
            self._chunks_added += 1
            
            # ✅ CRITICAL: Validate audio is int16 PCM
            if len(pcm_bytes) % 2 != 0:
                logger.warning(f"⚠️ Odd-length audio chunk: {len(pcm_bytes)} bytes, truncating")
                pcm_bytes = pcm_bytes[:-1]
            
            # Add to partial buffer
            self._partial_buffer.extend(pcm_bytes)
            
            # ✅ CRITICAL: Extract complete frames carefully
            frames_added = 0
            while len(self._partial_buffer) >= self.bytes_per_frame:
                # Extract exactly one frame (1920 bytes = 960 samples * 2 bytes)
                frame_data = bytes(self._partial_buffer[:self.bytes_per_frame])
                self._partial_buffer = self._partial_buffer[self.bytes_per_frame:]
                
                # ✅ Verify frame is valid
                if len(frame_data) != self.bytes_per_frame:
                    logger.error(f"❌ Invalid frame size: {len(frame_data)} bytes")
                    continue
                
                try:
                    self.audio_queue.put_nowait(frame_data)
                    frames_added += 1
                    
                    # ✅ Start playing only ONCE (first time we have enough frames)
                    if not self._ever_started and self.audio_queue.qsize() >= self._min_frames_to_start:
                        self._is_playing = True
                        self._ever_started = True
                        logger.info(f"✅ Initial pre-buffer ready! Queue has {self.audio_queue.qsize()} frames")
                    
                except asyncio.QueueFull:
                    self._frames_dropped += 1
                    if self._frames_dropped <= 10:
                        logger.warning(f"⚠️ Queue full ({self.audio_queue.qsize()}/500), dropped frame #{self._frames_dropped} - THIS CAUSES CHOPPY AUDIO!")
            
            # Log periodically
            if self._chunks_added <= 5 or self._chunks_added % 20 == 0:
                logger.info(f"📥 Chunk {self._chunks_added}: {len(pcm_bytes)}B → {frames_added} frames, "
                           f"queue: {self.audio_queue.qsize()}/500, buffer_leftover: {len(self._partial_buffer)}B, "
                           f"started: {self._ever_started}, dropped: {self._frames_dropped}")
        
        except Exception as e:
            logger.error(f"❌ Error adding audio: {e}")
            import traceback
            traceback.print_exc()

    async def recv(self):
        """Get audio frame from queue"""
        if self._stopped:
            raise MediaStreamError
        
        # ✅ Wait for initial pre-buffer ONLY (never pre-buffer again)
        if not self._is_playing and not self._ever_started:
            # First time only - wait for buffer
            audio = np.zeros(self.samples_per_frame, dtype=np.int16)
            
            if self._frame_count % 25 == 0:
                logger.info(f"⏳ Initial pre-buffering... (queue: {self.audio_queue.qsize()}/{self._min_frames_to_start})")
        
        else:
            # Once started, always try to play (even if queue empty)
            self._is_playing = True  # Keep playing forever once started
            
            try:
                # ✅ CRITICAL FIX: Use longer timeout to wait for slow TTS
                # This prevents gaps when TTS is generating slowly
                pcm = await asyncio.wait_for(
                    self.audio_queue.get(),
                    timeout=0.1  # 100ms timeout (was 20ms) - wait longer for audio
                )
                audio = np.frombuffer(pcm, dtype=np.int16)
                
            except asyncio.TimeoutError:
                # ✅ If queue is getting low, log it
                if self.audio_queue.qsize() < 5:
                    if self._frame_count % 50 == 0:
                        logger.warning(f"⚠️ Queue running low: {self.audio_queue.qsize()}/500 - audio gaps likely")
                
                # Queue empty - send silence but keep playing
                audio = np.zeros(self.samples_per_frame, dtype=np.int16)
        
        # Create frame
        audio_2d = audio.reshape(1, -1)
        frame = AudioFrame.from_ndarray(audio_2d, format='s16', layout='mono')
        frame.sample_rate = self.sample_rate
        frame.pts = self._frame_count * self.samples_per_frame
        
        self._frame_count += 1
        
        # Log first few frames and periodically
        if self._frame_count <= 5 or (self._ever_started and self._frame_count <= 20) or self._frame_count % 200 == 0:
            has_audio = np.any(audio != 0)
            logger.info(f"📤 Frame {self._frame_count}: has_audio={has_audio}, queue: {self.audio_queue.qsize()}/500")
        
        return frame

    def stop(self):
        logger.info(f"🛑 Stop:")
        logger.info(f"   Frames sent: {self._frame_count}")
        logger.info(f"   Chunks added: {self._chunks_added}")
        logger.info(f"   Frames dropped: {self._frames_dropped}")
        logger.info(f"   Queue remaining: {self.audio_queue.qsize()}")
        
        self._stopped = True
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break

    def reset(self):
        """Reset only clears queue, keeps playing state"""
        logger.info(f"♻️ Reset (keeping playing state)")
        
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        self._partial_buffer.clear()
        self._chunks_added = 0
        self._frames_dropped = 0
        # ✅ DON'T reset _is_playing or _ever_started
        # Once we start playing, we stay playing forever