import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def main():
    print("Comparing Free/Working Models from Google, Groq, and Mistral\n")
    
    # Models mapped from available_models.csv that are free/working
    model_configs = {
        "Gemini (Google)": "google_genai:gemini-3.6-flash",
        "Llama 3.3 (Groq)": "groq:llama-3.3-70b-versatile",
        "Mistral Small (Mistral)": "mistralai:mistral-small-latest"
    }

    models = {}
    for name, model_string in model_configs.items():
        try:
            # We set fixed temperature and max_tokens here to ensure a fair, 
            # apples-to-apples comparison across all models
            models[name] = init_chat_model(
                model_string,
                temperature=0.5,
                max_tokens=150
            )
            print(f"Successfully initialized {name}")
        except Exception as e:
            print(f"Failed to initialize {name}: {e}")

    prompt = "In exactly one sentence, describe what artificial intelligence is."
    print(f"\nPrompt: {prompt}\n" + "="*50)

    for name, model in models.items():
        print(f"\n[{name}]")
        try:
            response = model.invoke(prompt)
            print(getattr(response, 'text', getattr(response, 'content', str(response))))
        except Exception as e:
            print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()
