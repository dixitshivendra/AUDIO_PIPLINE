"""Unit tests for audio processing operations."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from worker.audio_ops import AudioProcessingError, normalize_highway_audio


class TestNormalizeHighwayAudio:
    """Tests for FFmpeg audio normalization pipeline."""

    def test_successful_normalization(self, tmp_path):
        """Should call FFmpeg with correct arguments."""
        input_file = tmp_path / "input.m4a"
        output_file = tmp_path / "output.wav"
        input_file.write_bytes(b"\x00" * 100)

        with patch("worker.audio_ops.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
            result = normalize_highway_audio(str(input_file), str(output_file))

            assert result == str(output_file)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "ffmpeg" in cmd
            assert "-af" in cmd
            assert "highpass=f=300" in cmd[cmd.index("-af") + 1]

    def test_ffmpeg_failure_raises_error(self, tmp_path):
        """Should raise AudioProcessingError on FFmpeg failure."""
        input_file = tmp_path / "input.m4a"
        output_file = tmp_path / "output.wav"
        input_file.write_bytes(b"\x00" * 100)

        with patch("worker.audio_ops.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "ffmpeg", stderr=b"error message"
            )
            with pytest.raises(AudioProcessingError, match="FFmpeg pipeline failed"):
                normalize_highway_audio(str(input_file), str(output_file))

    def test_ffmpeg_timeout_raises_error(self, tmp_path):
        """Should raise AudioProcessingError on timeout."""
        input_file = tmp_path / "input.m4a"
        output_file = tmp_path / "output.wav"
        input_file.write_bytes(b"\x00" * 100)

        with patch("worker.audio_ops.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 60)
            with pytest.raises(AudioProcessingError, match="timed out"):
                normalize_highway_audio(str(input_file), str(output_file))
