import os
import json
from pydantic import BaseModel
from openai import OpenAI

# 🔴 PUT YOUR ACTUAL GROQ KEY HERE 🔴


client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class CrisisData(BaseModel):
    is_actionable_emergency: bool
    severity: str


try:
    print("1. Sending Audio to Groq Whisper...")
    with open("clean_emergency.m4a", "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="verbose_json"
        )
    print(f"Transcript Received: '{transcript.text}'")

    print("\n2. Sending Transcript to Llama 3.3 for Parsing...")

    # FIX: Tell it to use standard JSON and pass the schema in the prompt!
    system_prompt = f"Extract data. You MUST return ONLY valid JSON matching this schema: {CrisisData.model_json_schema()}"

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript.text}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    print("\nSUCCESS! Parsed Data:")
    # FIX: Manually validate the JSON string using Pydantic
    parsed_data = CrisisData.model_validate_json(
        completion.choices[0].message.content)
    print(parsed_data.model_dump_json(indent=2))

except Exception as e:
    print(f"\nCRASHED: {str(e)}")
