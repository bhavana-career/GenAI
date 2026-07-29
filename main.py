from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

models = {
    "Gemini": init_chat_model(
        "google_genai:gemini-3.6-flash"
    ),
    "Groq": init_chat_model(
        "groq:llama-3.3-70b-versatile"
    ),
    "Mistral": init_chat_model(
        "mistralai:mistral-small-latest"
    ),
}

query = "Explain recursion in one paragraph."

for provider, model in models.items():
    print(f"\n===== {provider} =====")

    try:
        response = model.invoke(query)
        print(response.content)
    except Exception as e:
        print(f"Error: {e}")