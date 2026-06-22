import subprocess
import os


class AudioProcessingError(Exception):
    pass


def normalize_highway_audio(input_path: str, output_path: str) -> str:
    """
    Advanced FFmpeg filter chain for highway environments:
    - highpass/lowpass: Isolates human voice (300Hz - 3400Hz).
    - afftdn: FFT-based background noise reduction (removes wind/rumble).
    - loudnorm: Broadcast-standard volume normalization.
    - ar/ac: Forces 16kHz mono audio (optimal for Whisper ASR).
    """
    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-af", "highpass=f=300,lowpass=f=3400,afftdn=nf=-25,loudnorm",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]

    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_path
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(
            f"FFmpeg pipeline failed: {e.stderr.decode()}")
