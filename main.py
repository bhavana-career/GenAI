import os
from dotenv import load_dotenv
import pandas as pd

from google import genai
from groq import Groq
from mistralai.client import Mistral

load_dotenv()

rows = []

####################################################
# GOOGLE
####################################################

print("=" * 60)
print("GOOGLE")
print("=" * 60)

google_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

for model in google_client.models.list():

    name = model.name.lower()

    if "embed" in name:
        capability = "Embedding"

    elif "imagen" in name:
        capability = "Image Generation"

    elif "veo" in name:
        capability = "Video Generation"

    elif "tts" in name or "audio" in name:
        capability = "Audio"

    elif "ocr" in name:
        capability = "OCR"

    else:
        capability = "Chat"

    print(f"{model.name:<50} {capability}")

    rows.append(
        {
            "Provider": "Google",
            "Model": model.name,
            "Capability": capability,
        }
    )

####################################################
# GROQ
####################################################

print("\n")
print("=" * 60)
print("GROQ")
print("=" * 60)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

models = groq_client.models.list()

for model in models.data:

    name = model.id.lower()

    if "whisper" in name:
        capability = "Speech-to-Text"

    elif "guard" in name:
        capability = "Safety"

    else:
        capability = "Chat"

    print(f"{model.id:<50} {capability}")

    rows.append(
        {
            "Provider": "Groq",
            "Model": model.id,
            "Capability": capability,
        }
    )

####################################################
# MISTRAL
####################################################

print("\n")
print("=" * 60)
print("MISTRAL")
print("=" * 60)

mistral_client = Mistral(
    api_key=os.getenv("MISTRAL_API_KEY")
)

models = mistral_client.models.list()

for model in models.data:

    name = model.id.lower()

    if "embed" in name:
        capability = "Embedding"

    elif "ocr" in name:
        capability = "OCR"

    elif "tts" in name:
        capability = "Text-to-Speech"

    elif "transcribe" in name:
        capability = "Speech-to-Text"

    elif "voxtral" in name:
        capability = "Audio"

    elif "moderation" in name:
        capability = "Moderation"

    else:
        capability = "Chat"

    print(f"{model.id:<50} {capability}")

    rows.append(
        {
            "Provider": "Mistral",
            "Model": model.id,
            "Capability": capability,
        }
    )

####################################################
# SAVE
####################################################

df = pd.DataFrame(rows)

df.to_csv("available_models.csv", index=False)

print("\n")
print("=" * 60)
print(df)
print("=" * 60)

print("\nSaved to available_models.csv")