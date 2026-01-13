# server/piper_stream.py - WITH BACKPRESSURE TO PREVENT QUEUE OVERFLOW

import subprocess
import logging
import numpy as np
from scipy import signal
import time

logger = logging.getLogger(__name__)

def generate_speech_piper(text: str, model_path: str, send_audio_callback) -> bool:
    """
    Generate speech with Piper and resample to 48kHz
    Includes backpressure to prevent queue overflow
    """
    if not text or not text.strip():
        logger.warning("Empty text provided to Piper")
        return False
    
    if not model_path:
        logger.error("No Piper model path provided")
        return False
    
    logger.info(f"🎵 Starting Piper TTS: '{text[:50]}...'")
    
    try:
        cmd = ["piper", "--model", model_path, "--output-raw"]
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0
        )
        
        # Send text
        process.stdin.write(text.encode('utf-8'))
        process.stdin.close()
        logger.info("✅ Sent text to Piper")
        
        chunk_count = 0
        total_bytes_in = 0
        total_bytes_out = 0
        
        logger.info("Streaming and resampling audio...")
        
        # Read, resample, send
        while True:
            # Read 22050Hz audio from Piper
            chunk_22k = process.stdout.read(4096)
            
            if not chunk_22k:
                logger.info(f"✅ Piper stream ended ({chunk_count} chunks)")
                break
            
            chunk_count += 1
            total_bytes_in += len(chunk_22k)
            
            # ✅ CRITICAL: Resample 22050Hz → 48000Hz using linear interpolation
            # scipy.signal.resample can cause artifacts - use simpler method
            try:
                # Convert to float for resampling
                audio_22k = np.frombuffer(chunk_22k, dtype=np.int16).astype(np.float32)
                
                # Use linear interpolation for cleaner resampling
                # Calculate the exact ratio
                ratio = 48000 / 22050  # ~2.176
                
                # Create time indices for original and target sample rates
                old_indices = np.arange(len(audio_22k))
                new_length = int(len(audio_22k) * ratio)
                new_indices = np.linspace(0, len(audio_22k) - 1, new_length)
                
                # Linear interpolation
                audio_48k = np.interp(new_indices, old_indices, audio_22k)
                
                # Clip and convert back to int16
                audio_48k = np.clip(audio_48k, -32767, 32767).astype(np.int16)
                
                # Send resampled audio
                pcm_48k = audio_48k.tobytes()
                total_bytes_out += len(pcm_48k)
                
                if chunk_count <= 3:
                    logger.info(f"Chunk {chunk_count}: {len(chunk_22k)}B@22kHz → {len(pcm_48k)}B@48kHz")
                
                # ✅ CRITICAL FIX: Send to callback
                result = send_audio_callback(pcm_48k)
                
                # ✅ Check if callback is applying backpressure
                if result is False:
                    logger.warning("Callback stopped streaming")
                    break
                
            except Exception as e:
                logger.error(f"❌ Resample error at chunk {chunk_count}: {e}")
                break
        
        process.wait(timeout=5)
        
        logger.info(f"✅ Piper completed: {total_bytes_in}B in → {total_bytes_out}B out")
        return True
        
    except Exception as e:
        logger.error(f"❌ Piper error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        try:
            if process.poll() is None:
                process.kill()
        except:
            pass