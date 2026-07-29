import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from google import genai
from groq import Groq
from mistralai.client import Mistral

load_dotenv()

model = init_chat_model(
    "google_genai:gemini-3.6-flash",
    max_tokens=500,
)

response = model.invoke("Explain recursion.")
print(response.content)