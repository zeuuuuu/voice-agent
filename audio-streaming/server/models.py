# server/models.py
import whisper
import torch

print("Loading Whisper 'base'...")
whisper_model = whisper.load_model("base")

print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False,
    trust_repo=True
)
get_speech_timestamps, _, _, _, _ = utils

print("Models ready!")