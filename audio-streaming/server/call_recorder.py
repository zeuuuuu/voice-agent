# server/call_recorder.py — FINAL, CORRECT VERSION

import wave
import os
import time
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class CallRecorder:
    def __init__(self, session_id: str, sample_rate: int = 48000):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.sample_width = 2

        self.events = []   # 🔥 TIMELINE EVENTS
        self.start_time = time.time()

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        os.makedirs("recordings", exist_ok=True)

        self.output_path = f"recordings/call_{session_id}_{timestamp}_conversation.wav"

        logger.info(f"📞 CallRecorder initialized for session {session_id}")

    # -------------------------------------------------
    # EVENT RECORDING
    # -------------------------------------------------

    def add_user_audio(self, pcm: bytes):
        self._add_event("user", pcm)

    def add_ai_audio(self, pcm: bytes):
        self._add_event("ai", pcm)

    def _add_event(self, speaker: str, pcm: bytes):
        if not pcm:
            return

        self.events.append({
            "speaker": speaker,
            "time": time.time() - self.start_time,
            "pcm": pcm,
        })

    # -------------------------------------------------
    # FINALIZE — TRUE SEQUENTIAL AUDIO
    # -------------------------------------------------

    def finalize(self):
        if not self.events:
            logger.warning("⚠️ No audio events recorded")
            return None

        # Sort by time (ABSOLUTE ORDER)
        self.events.sort(key=lambda e: e["time"])

        conversation = np.concatenate([
            np.frombuffer(e["pcm"], dtype=np.int16)
            for e in self.events
        ])

        with wave.open(self.output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(conversation.tobytes())

        duration = len(conversation) / self.sample_rate
        logger.info(f"🎧 Sequential conversation saved: {self.output_path} ({duration:.2f}s)")

        return self.output_path
