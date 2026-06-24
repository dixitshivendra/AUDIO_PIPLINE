import subprocess
from core.metrics import get_json_logger

logger = get_json_logger("audio_ops")

FFMPEG_TIMEOUT = 60  # seconds


class AudioProcessingError(Exception):
    """Raised when FFmpeg audio processing fails."""


def normalize_highway_audio(input_path: str, output_path: str) -> str:
    """
    Advanced FFmpeg filter chain for highway environments:
    - highpass/lowpass: Isolates human voice (300Hz - 3400Hz).
    - afftdn: FFT-based background noise reduction (removes wind/rumble).
    - loudnorm: Broadcast-standard volume normalization.
    - ar/ac: Forces 16kHz mono audio (optimal for Whisper ASR).
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        "highpass=f=300,lowpass=f=3400,afftdn=nf=-25,loudnorm",
        "-ar",
        "16000",
        "-ac",
        "1",
        output_path,
    ]

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=FFMPEG_TIMEOUT,
        )
        return output_path
    except subprocess.TimeoutExpired:
        raise AudioProcessingError(
            f"FFmpeg timed out after {FFMPEG_TIMEOUT}s processing {input_path}"
        )
    except subprocess.CalledProcessError as e:
        raise AudioProcessingError(f"FFmpeg pipeline failed: {e.stderr.decode()}")
