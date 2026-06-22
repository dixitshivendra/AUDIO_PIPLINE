import os
from typing import Literal, Annotated
from pydantic import BaseModel, Field, model_validator, BeforeValidator
from openai import OpenAI
from .audio_ops import normalize_highway_audio

# Fail-Fast Validation
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("CRITICAL: Missing OPENAI_API_KEY in environment.")

# Initialize OpenAI client pointing to Groq's high-speed API
client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")


class AudioNoiseException(Exception):
    pass


class LowConfidenceException(Exception):
    pass


class AmbiguityException(Exception):
    pass


# Helper function to lowercase LLM string outputs before validation
def to_lower(v):
    if isinstance(v, str):
        return v.lower()
    return v


class CrisisData(BaseModel):
    is_actionable_emergency: bool = Field(
        description="True ONLY if the text describes a vehicular/highway emergency.")
    needs_human_review: bool = Field(
        description="True if the text is a prank, ambiguous, contradictory, or lacks details.")
    rejection_reason: str | None = Field(
        default=None, description="Explanation if human review is required.")

    # FIX: Using BeforeValidator so "High" automatically becomes "high"
    severity: Annotated[Literal["low", "medium", "high", "critical"]
                        | None, BeforeValidator(to_lower)] = Field(default=None)
    vehicle_type: Annotated[Literal["car", "bike", "truck", "bus", "auto",
                                    "unknown"] | None, BeforeValidator(to_lower)] = Field(default=None)

    victim_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.is_actionable_emergency and self.severity is None:
            raise ValueError(
                "Actionable emergencies must have a severity assigned.")
        if self.needs_human_review and not self.rejection_reason:
            raise ValueError(
                "A rejection_reason must be provided if human review is flagged.")
        return self


def process_dispatch(raw_audio_path: str) -> dict:
    normalized_audio = "clean_temp.wav"
    normalize_highway_audio(raw_audio_path, normalized_audio)
    transcript_text = ""

    try:
        with open(normalized_audio, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        if not hasattr(transcript_response, "segments") or not transcript_response.segments:
            raise AudioNoiseException("Whisper returned empty segments.")

        segments = transcript_response.segments

        print("SEGMENT TYPE:", type(segments[0]))
        print("FIRST SEGMENT:", segments[0])

        # Handle both dict and object formats
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

        print("TRANSCRIPT:", transcript_text)
        print("AVG_NO_SPEECH:", avg_no_speech)
        print("AVG_LOGPROB:", avg_logprob)

        if avg_no_speech > 0.45:
            raise LowConfidenceException(
                f"High no-speech probability ({avg_no_speech:.2f})."
            )

        if avg_logprob < -1.0:
            raise LowConfidenceException(
                f"Transcription confidence too low ({avg_logprob:.2f})."
            )

        if len(transcript_text.split()) < 4:
            raise AudioNoiseException(
                f"Transcript too short: '{transcript_text}'"
            )

        if avg_no_speech > 0.45:
            raise LowConfidenceException(
                f"High no-speech probability ({avg_no_speech:.2f}).")
        if avg_logprob < -1.0:
            raise LowConfidenceException(
                f"Transcription confidence too low ({avg_logprob:.2f}).")
        if len(transcript_text.split()) < 4:
            raise AudioNoiseException(
                f"Transcript too short: '{transcript_text}'")

        # FIX: Inject the schema into the prompt and ask for standard JSON format
        system_prompt = f"""You are an emergency dispatch AI. Extract crisis data. 
        You MUST return ONLY valid JSON matching this exact schema:
        {CrisisData.model_json_schema()}
        """

        # FIX: Use .create() instead of .parse(), and ask for json_object type
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        # FIX: Manually validate the JSON string returned by Groq using Pydantic
        extraction = CrisisData.model_validate_json(
            completion.choices[0].message.content)

        if extraction.needs_human_review:
            raise AmbiguityException(
                f"LLM flagged for review: {extraction.rejection_reason}")
        if not extraction.is_actionable_emergency:
            raise AmbiguityException(
                "Speech detected, but no actionable emergency identified.")

        return {
            "status": "completed",
            "extracted_data": extraction.model_dump(),
            "confidence": avg_logprob,
            "transcript": transcript_text
        }

    except (AudioNoiseException, LowConfidenceException, AmbiguityException, ValueError) as e:
        return {"status": "requires_human_review", "reason": f"{type(e).__name__}: {str(e)}", "raw_transcript": transcript_text}
    except Exception as e:
        return {"status": "requires_human_review", "reason": f"System Error: {str(e)}", "raw_transcript": transcript_text}
