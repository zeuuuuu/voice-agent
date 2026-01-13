# # server/pipeline.py — FINAL FIX: AI audio wired to AudioOutputTrack

# from langchain_core.runnables import RunnableGenerator
# from .agent import agent_stream
# from .tts import tts_stream

# def create_voice_pipeline(audio_track, piper_model_path: str, use_elevenlabs: bool = False):
#     """
#     Create agent + TTS pipeline.
#     GUARANTEE: all AI PCM audio is sent to AudioOutputTrack.add_audio()
#     """

#     async def tts_with_context(event_stream):
#         async for event in tts_stream(
#             event_stream,
#             audio_track,
#             piper_model_path,
#             use_elevenlabs,
#         ):
#             # --------------------------------------------------
#             # 🔥 CRITICAL FIX: Intercept AI audio here
#             # --------------------------------------------------
#             if isinstance(event, dict):
#                 pcm = event.get("audio")
#                 if pcm:
#                     audio_track.add_audio(pcm)

#             yield event

#     pipeline = (
#         RunnableGenerator(agent_stream)
#         | RunnableGenerator(tts_with_context)
#     )

#     return pipeline

#server/pipeline.py

from langchain_core.runnables import RunnableGenerator
from .agent import agent_stream
from .tts import tts_stream

def create_voice_pipeline(audio_track, piper_model_path: str, use_elevenlabs: bool = False):
    """Voice pipeline: Agent → TTS"""
    
    async def tts_with_context(event_stream):
        async for event in tts_stream(
            event_stream,
            audio_track,
            piper_model_path,
            use_elevenlabs,
        ):
            yield event
    
    pipeline = (
        RunnableGenerator(agent_stream)
        | RunnableGenerator(tts_with_context)
    )
    
    return pipeline