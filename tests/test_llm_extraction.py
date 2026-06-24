"""Unit tests for LLM extraction pipeline.

Covers both happy path and edge cases including malformed JSON responses,
missing fields, and various confidence threshold scenarios.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from worker.llm_extraction import (
    AmbiguityException,
    AudioNoiseException,
    CrisisData,
    LowConfidenceException,
    process_dispatch,
    to_lower,
)


class TestToLower:
    """Tests for the to_lower validator helper."""

    def test_lowercases_string(self):
        assert to_lower("HIGH") == "high"

    def test_passthrough_non_string(self):
        assert to_lower(None) is None
        assert to_lower(42) == 42

    def test_already_lowercase(self):
        assert to_lower("low") == "low"


class TestCrisisDataModel:
    """Tests for the CrisisData Pydantic model."""

    def test_valid_emergency(self):
        data = CrisisData(
            is_actionable_emergency=True,
            needs_human_review=False,
            severity="high",
            vehicle_type="car",
            victim_count=2,
        )
        assert data.is_actionable_emergency is True
        assert data.severity == "high"

    def test_severity_lowercase_normalization(self):
        data = CrisisData(
            is_actionable_emergency=True,
            needs_human_review=False,
            severity="HIGH",
        )
        assert data.severity == "high"

    def test_emergency_requires_severity(self):
        with pytest.raises(ValueError, match="severity"):
            CrisisData(
                is_actionable_emergency=True,
                needs_human_review=False,
            )

    def test_human_review_requires_reason(self):
        with pytest.raises(ValueError, match="rejection_reason"):
            CrisisData(
                is_actionable_emergency=False,
                needs_human_review=True,
            )

    def test_optional_fields_default_none(self):
        data = CrisisData(
            is_actionable_emergency=False,
            needs_human_review=False,
        )
        assert data.severity is None
        assert data.vehicle_type is None
        assert data.victim_count is None

    def test_gps_latitude_validation(self):
        with pytest.raises(ValueError):
            CrisisData(
                is_actionable_emergency=False,
                needs_human_review=False,
                gps_latitude=999,
            )

    def test_gps_longitude_validation(self):
        with pytest.raises(ValueError):
            CrisisData(
                is_actionable_emergency=False,
                needs_human_review=False,
                gps_longitude=200,
            )

    def test_victim_count_negative_rejected(self):
        with pytest.raises(ValueError):
            CrisisData(
                is_actionable_emergency=True,
                needs_human_review=False,
                severity="low",
                victim_count=-1,
            )

    def test_language_normalization(self):
        data = CrisisData(
            is_actionable_emergency=False,
            needs_human_review=False,
            language="HI",
        )
        assert data.language == "hi"

    def test_injury_type_normalization(self):
        data = CrisisData(
            is_actionable_emergency=True,
            needs_human_review=False,
            severity="high",
            injury_type="SEVERE",
        )
        assert data.injury_type == "severe"

    def test_all_fields_populated(self):
        data = CrisisData(
            is_actionable_emergency=True,
            needs_human_review=False,
            severity="critical",
            vehicle_type="truck",
            victim_count=5,
            language="hi",
            gps_latitude=28.6139,
            gps_longitude=77.2090,
            road_name="NH-44",
            landmark="near India Gate",
            injury_type="severe",
        )
        assert data.gps_latitude == 28.6139
        assert data.road_name == "NH-44"


class TestProcessDispatch:
    """Tests for the full process_dispatch pipeline."""

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    @patch("worker.llm_extraction._extract_crisis_data")
    def test_successful_processing(self, mock_extract, mock_transcribe, mock_normalize, tmp_path):
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[
                {"no_speech_prob": 0.1, "avg_logprob": -0.3},
                {"no_speech_prob": 0.05, "avg_logprob": -0.2},
            ],
            text="There is a car accident on the highway with injuries",
        )

        crisis_data = {
            "is_actionable_emergency": True,
            "needs_human_review": False,
            "severity": "high",
            "vehicle_type": "car",
            "victim_count": 2,
        }
        mock_extract.return_value = json.dumps(crisis_data)

        result = process_dispatch(str(audio_file))

        assert result["status"] == "completed"
        assert result["extracted_data"]["severity"] == "high"
        assert result["extracted_data"]["victim_count"] == 2

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    def test_high_no_speech_quarantines(self, mock_transcribe, mock_normalize, tmp_path):
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.9, "avg_logprob": -0.3}],
            text="some text here for minimum length",
        )

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"
        assert "LowConfidenceException" in result["reason"]

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    def test_low_logprob_quarantines(self, mock_transcribe, mock_normalize, tmp_path):
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.1, "avg_logprob": -1.5}],
            text="some text here for minimum length",
        )

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"
        assert "LowConfidenceException" in result["reason"]

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    def test_short_transcript_quarantines(self, mock_transcribe, mock_normalize, tmp_path):
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.1, "avg_logprob": -0.3}],
            text="hello",
        )

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"
        assert "AudioNoiseException" in result["reason"]

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    def test_empty_segments_quarantine(self, mock_transcribe, mock_normalize, tmp_path):
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(segments=[], text="")

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    @patch("worker.llm_extraction._extract_crisis_data")
    def test_malformed_json_quarantines(self, mock_extract, mock_transcribe, mock_normalize, tmp_path):
        """LLM returns invalid JSON that fails Pydantic validation."""
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.1, "avg_logprob": -0.3}],
            text="There is a car accident on the highway with injuries",
        )
        mock_extract.return_value = "not valid json at all"

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    @patch("worker.llm_extraction._extract_crisis_data")
    def test_missing_required_fields_quarantines(self, mock_extract, mock_transcribe, mock_normalize, tmp_path):
        """LLM returns JSON missing required fields."""
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.1, "avg_logprob": -0.3}],
            text="There is a car accident on the highway with injuries",
        )
        mock_extract.return_value = json.dumps({"some_field": "value"})

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    @patch("worker.llm_extraction._extract_crisis_data")
    def test_human_review_flagged_quarantines(self, mock_extract, mock_transcribe, mock_normalize, tmp_path):
        """LLM flags the audio for human review."""
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.1, "avg_logprob": -0.3}],
            text="I think maybe there was something but I am not sure about the details",
        )
        mock_extract.return_value = json.dumps({
            "is_actionable_emergency": False,
            "needs_human_review": True,
            "rejection_reason": "Ambiguous - caller seems uncertain",
        })

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"
        assert "AmbiguityException" in result["reason"]

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    @patch("worker.llm_extraction._extract_crisis_data")
    def test_not_actionable_emergency_quarantines(self, mock_extract, mock_transcribe, mock_normalize, tmp_path):
        """Speech detected but no emergency identified."""
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.return_value = MagicMock(
            segments=[{"no_speech_prob": 0.1, "avg_logprob": -0.3}],
            text="Hello how are you doing today I am fine",
        )
        mock_extract.return_value = json.dumps({
            "is_actionable_emergency": False,
            "needs_human_review": False,
        })

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"
        assert "AmbiguityException" in result["reason"]

    @patch("worker.llm_extraction.normalize_highway_audio")
    @patch("worker.llm_extraction._transcribe_audio")
    def test_whisper_api_error_quarantines(self, mock_transcribe, mock_normalize, tmp_path):
        """Whisper API connection error is caught."""
        audio_file = tmp_path / "test.m4a"
        audio_file.write_bytes(b"\x00" * 100)

        mock_transcribe.side_effect = ConnectionError("Groq API unavailable")

        result = process_dispatch(str(audio_file))
        assert result["status"] == "requires_human_review"
        assert "System Error" in result["reason"]
