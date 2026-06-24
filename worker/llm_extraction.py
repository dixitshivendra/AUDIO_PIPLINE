"""LLM extraction pipeline for emergency audio processing.

Handles transcription via Groq Whisper and structured crisis data extraction
via Llama 3.3. Includes circuit breaker pattern to prevent cascading failures
when the Groq API is unhealthy.
"""

import os
import uuid
from typing import Annotated, Literal

from openai import OpenAI, APIStatusError, RateLimitError, APIConnectionError
from pydantic import BaseModel, BeforeValidator, Field, model_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

try:
    from tenacity import CircuitBreaker
except ImportError:
    class CircuitBreaker:
        def __init__(self, fail_max=5, reset_timeout=30):
            self.fail_max = fail_max
            self.reset_timeout = reset_timeout
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

from .audio_ops import normalize_highway_audio
from core.metrics import get_json_logger

logger = get_json_logger("llm_extraction")

GENERAL_SYSTEM_PROMPT = """You are an expert meeting/call analyst. Analyze the following transcript and extract structured insights.

You MUST return ONLY valid JSON matching this exact schema:
{
  "summary": "Concise 2-3 sentence summary of the conversation",
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "sentiment_score": 0.0 to 1.0 (1.0 = very positive, 0.0 = very negative),
  "speakers": [
    {"name": "Speaker name or label", "role": "role if mentioned or null", "sentiment": "positive|negative|neutral"}
  ],
  "keywords": ["top 5-10 keywords/phrases"],
  "topics": ["main topics discussed"],
  "action_items": [
    {"text": "action item description", "owner": "person name or null", "due": "due date if mentioned or null"}
  ],
  "decisions": [
    {"text": "decision made", "context": "brief context"}
  ],
  "language": "detected language code (en, hi, es, etc)",
  "entities": {
    "people": ["names mentioned"],
    "organizations": ["company/org names"],
    "dates": ["dates/times mentioned"],
    "monies": ["currency amounts mentioned"]
  },
  "compliance_flags": ["any compliance concerns, risk phrases, or escalation signals detected"],
  "competitor_mentions": ["competitor brand names mentioned"],
  "objections": ["objections or concerns raised during the conversation"],
  "talk_ratio": {"speaker_name": percentage_of_talk_time}
}

Rules:
- If speakers cannot be identified, use "Speaker 1", "Speaker 2", etc.
- sentiment_score should reflect overall tone: positive interactions > 0.6, negative < 0.4
- action_items should be concrete and actionable
- compliance_flags should include: profanity, threats, legal mentions, data sharing concerns, escalation signals
- Return ONLY the JSON, no markdown, no explanation."""

API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY or "not-set", base_url="https://api.groq.com/openai/v1") if API_KEY else None


def get_client():
    global client
    if client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("CRITICAL: Missing OPENAI_API_KEY in environment.")
        client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return client

# Circuit breaker: open after 5 consecutive failures, reset after 30s
_circuit_breaker = CircuitBreaker(
    fail_max=5,
    reset_timeout=30,
)


class AudioNoiseException(Exception):
    """Raised when audio contains no discernible speech."""


class LowConfidenceException(Exception):
    """Raised when transcription confidence is too low."""


class AmbiguityException(Exception):
    """Raised when the LLM response is ambiguous or not actionable."""


def to_lower(v):
    """Helper to lowercase LLM string outputs before validation."""
    if isinstance(v, str):
        return v.lower()
    return v


class CrisisData(BaseModel):
    """Structured schema for emergency dispatch data extraction.

    Fields cover emergency classification, vehicle details, location,
    injury assessment, and language detection for Indian highways.
    """

    is_actionable_emergency: bool = Field(
        description="True ONLY if the text describes a vehicular/highway emergency.",
    )
    needs_human_review: bool = Field(
        description="True if the text is a prank, ambiguous, contradictory, or lacks details.",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="Explanation if human review is required.",
    )
    severity: Annotated[
        Literal["low", "medium", "high", "critical"] | None,
        BeforeValidator(to_lower),
    ] = Field(default=None)
    vehicle_type: Annotated[
        Literal["car", "bike", "truck", "bus", "auto", "unknown"] | None,
        BeforeValidator(to_lower),
    ] = Field(default=None)
    victim_count: int | None = Field(default=None, ge=0)
    language: Annotated[
        Literal["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "ur", "unknown"] | None,
        BeforeValidator(to_lower),
    ] = Field(default=None, description="Detected language of the audio.")
    gps_latitude: float | None = Field(
        default=None, ge=-90, le=90,
        description="Extracted GPS latitude if mentioned.",
    )
    gps_longitude: float | None = Field(
        default=None, ge=-180, le=180,
        description="Extracted GPS longitude if mentioned.",
    )
    road_name: str | None = Field(
        default=None,
        description="Road or highway name/number if mentioned.",
    )
    landmark: str | None = Field(
        default=None,
        description="Nearby landmark or reference point if mentioned.",
    )
    injury_type: Annotated[
        Literal["none", "minor", "moderate", "severe", "fatal", "unknown"] | None,
        BeforeValidator(to_lower),
    ] = Field(default=None, description="Injury classification if determinable.")

    @model_validator(mode="after")
    def validate_consistency(self):
        """Ensure extracted data is internally consistent."""
        if self.is_actionable_emergency and self.severity is None:
            raise ValueError("Actionable emergencies must have a severity assigned.")
        if self.needs_human_review and not self.rejection_reason:
            raise ValueError(
                "A rejection_reason must be provided if human review is flagged."
            )
        return self


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((
        ConnectionError, TimeoutError,
        APIStatusError, RateLimitError, APIConnectionError,
    )),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
def _transcribe_audio(audio_path: str) -> dict:
    """Transcribe audio using Groq Whisper with retry and circuit breaker.

    Args:
        audio_path: Path to the normalized WAV file.

    Returns:
        Whisper API response with segments and text.

    Raises:
        CircuitBreakerError: If the Groq API is in a failure state.
        ConnectionError: If the API is unreachable.
    """
    with _circuit_breaker:
        with open(audio_path, "rb") as audio_file:
            return get_client().audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((
        ConnectionError, TimeoutError,
        APIStatusError, RateLimitError, APIConnectionError,
    )),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
def _extract_crisis_data(transcript_text: str, system_prompt: str) -> dict:
    """Extract crisis data from transcript using LLM with retry and circuit breaker.

    Args:
        transcript_text: The full transcript from Whisper.
        system_prompt: System message containing the CrisisData JSON schema.

    Returns:
        JSON string containing the extracted crisis data.

    Raises:
        CircuitBreakerError: If the Groq API is in a failure state.
    """
    with _circuit_breaker:
        completion = get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return completion.choices[0].message.content


class AudioAnalysisData(BaseModel):
    """Structured schema for general audio/meeting analysis."""

    summary: str = Field(description="2-3 sentence summary")
    sentiment: Annotated[
        Literal["positive", "negative", "neutral", "mixed"] | None,
        BeforeValidator(to_lower),
    ] = Field(default=None)
    sentiment_score: float | None = Field(default=None, ge=0.0, le=1.0)
    speakers: list[dict] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    action_items: list[dict] = Field(default_factory=list)
    decisions: list[dict] = Field(default_factory=list)
    language: str | None = Field(default=None)
    entities: dict = Field(default_factory=dict)
    compliance_flags: list[str] = Field(default_factory=list)
    competitor_mentions: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    talk_ratio: dict = Field(default_factory=dict)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((
        ConnectionError, TimeoutError,
        APIStatusError, RateLimitError, APIConnectionError,
    )),
    before_sleep=before_sleep_log(logger, "WARNING"),
)
def _extract_general_data(transcript_text: str, system_prompt: str) -> dict:
    """Extract general meeting/call data from transcript using LLM."""
    with _circuit_breaker:
        completion = get_client().chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return completion.choices[0].message.content


def process_general(raw_audio_path: str) -> dict:
    """General-purpose audio analysis pipeline for meetings/calls.

    Produces: summary, sentiment, speakers, keywords, topics, action_items,
    decisions, compliance flags, competitor mentions, and more.
    """
    normalized_audio = f"/tmp/clean_{uuid.uuid4().hex}.wav"
    normalize_highway_audio(raw_audio_path, normalized_audio)
    transcript_text = ""

    try:
        transcript_response = _transcribe_audio(normalized_audio)

        if not hasattr(transcript_response, "segments") or not transcript_response.segments:
            raise AudioNoiseException("Whisper returned empty segments.")

        segments = transcript_response.segments
        avg_no_speech = sum(
            s.get("no_speech_prob", 0) if isinstance(s, dict)
            else getattr(s, "no_speech_prob", 0)
            for s in segments
        ) / len(segments)

        avg_logprob = sum(
            s.get("avg_logprob", 0) if isinstance(s, dict)
            else getattr(s, "avg_logprob", 0)
            for s in segments
        ) / len(segments)

        transcript_text = transcript_response.text.strip()

        logger.info("Transcription complete", extra={
            "transcript_length": len(transcript_text),
            "avg_no_speech": avg_no_speech,
            "avg_logprob": avg_logprob,
        })

        if avg_no_speech > 0.5:
            raise LowConfidenceException(f"High no-speech probability ({avg_no_speech:.2f}).")
        if avg_logprob < -1.2:
            raise LowConfidenceException(f"Transcription confidence too low ({avg_logprob:.2f}).")
        if len(transcript_text.split()) < 3:
            raise AudioNoiseException(f"Transcript too short: '{transcript_text}'")

        json_content = _extract_general_data(transcript_text, GENERAL_SYSTEM_PROMPT)
        extraction = AudioAnalysisData.model_validate_json(json_content)

        return {
            "status": "completed",
            "extracted_data": extraction.model_dump(),
            "confidence": avg_logprob,
            "transcript": transcript_text,
        }

    except (AudioNoiseException, LowConfidenceException, ValueError) as e:
        return {
            "status": "requires_human_review",
            "reason": f"{type(e).__name__}: {str(e)}",
            "raw_transcript": transcript_text,
        }
    except Exception as e:
        logger.error("LLM extraction failed after retries, returning minimal result", extra={"error": str(e)})
        return {
            "status": "completed",
            "extracted_data": {},
            "transcript": transcript_text or None,
            "confidence": None,
        }
    finally:
        if os.path.exists(normalized_audio):
            os.remove(normalized_audio)


def process_dispatch(raw_audio_path: str) -> dict:
    """Full audio processing pipeline: normalize -> transcribe -> extract -> validate.

    Orchestrates the complete RAKSHA processing flow:
    1. FFmpeg normalization (highpass, lowpass, noise reduction, loudnorm)
    2. Groq Whisper transcription with segment-level confidence
    3. Confidence validation (no_speech_prob, avg_logprob, word count)
    4. Llama 3.3 structured extraction with Pydantic validation

    Args:
        raw_audio_path: Path to the original uploaded audio file.

    Returns:
        Dict with "status" key ("completed" or "requires_human_review")
        and additional data depending on outcome.
    """
    normalized_audio = f"/tmp/clean_{uuid.uuid4().hex}.wav"
    normalize_highway_audio(raw_audio_path, normalized_audio)
    transcript_text = ""

    try:
        transcript_response = _transcribe_audio(normalized_audio)

        if not hasattr(transcript_response, "segments") or not transcript_response.segments:
            raise AudioNoiseException("Whisper returned empty segments.")

        segments = transcript_response.segments

        avg_no_speech = sum(
            s.get("no_speech_prob", 0) if isinstance(s, dict)
            else getattr(s, "no_speech_prob", 0)
            for s in segments
        ) / len(segments)

        avg_logprob = sum(
            s.get("avg_logprob", 0) if isinstance(s, dict)
            else getattr(s, "avg_logprob", 0)
            for s in segments
        ) / len(segments)

        transcript_text = transcript_response.text.strip()

        logger.info(
            "Transcription complete",
            extra={
                "transcript": transcript_text,
                "avg_no_speech": avg_no_speech,
                "avg_logprob": avg_logprob,
            },
        )

        if avg_no_speech > 0.45:
            raise LowConfidenceException(
                f"High no-speech probability ({avg_no_speech:.2f})."
            )
        if avg_logprob < -1.0:
            raise LowConfidenceException(
                f"Transcription confidence too low ({avg_logprob:.2f})."
            )
        if len(transcript_text.split()) < 4:
            raise AudioNoiseException(f"Transcript too short: '{transcript_text}'")

        system_prompt = f"""You are an emergency dispatch AI. Extract crisis data.
        You MUST return ONLY valid JSON matching this exact schema:
        {CrisisData.model_json_schema()}
        """

        json_content = _extract_crisis_data(transcript_text, system_prompt)
        extraction = CrisisData.model_validate_json(json_content)

        if extraction.needs_human_review:
            raise AmbiguityException(
                f"LLM flagged for review: {extraction.rejection_reason}"
            )
        if not extraction.is_actionable_emergency:
            raise AmbiguityException(
                "Speech detected, but no actionable emergency identified."
            )

        return {
            "status": "completed",
            "extracted_data": extraction.model_dump(),
            "confidence": avg_logprob,
            "transcript": transcript_text,
        }

    except (AudioNoiseException, LowConfidenceException, AmbiguityException, ValueError) as e:
        return {
            "status": "requires_human_review",
            "reason": f"{type(e).__name__}: {str(e)}",
            "raw_transcript": transcript_text,
        }
    except Exception as e:
        return {
            "status": "requires_human_review",
            "reason": f"System Error: {str(e)}",
            "raw_transcript": transcript_text,
        }
    finally:
        if os.path.exists(normalized_audio):
            os.remove(normalized_audio)
