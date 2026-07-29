import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def main():
    print("Testing Model Parameters\n")
    
    # 1. Creative Model (High Temperature)
    print("Initializing Creative Model...")
    creative_model = init_chat_model(
        "google_genai:gemini-3.6-flash",
    )

    # 2. Strict/Predictable Model (Low Temperature)
    print("Initializing Strict Model (temp=0.1)...")
    strict_model = init_chat_model(
        "google_genai:gemini-3.6-flash",
    )

    prompt = "Write a creative and unusual name for a pet cat. Just the name and a short reason."
    print(f"\nPrompt: {prompt}\n")
    
    print("--- Creative Response (High Temp) ---")
    res1 = creative_model.invoke(prompt)
    print(getattr(res1, 'text', getattr(res1, 'content', str(res1))))
    
    print("\n--- Strict Response (Low Temp) ---")
    res2 = strict_model.invoke(prompt)
    print(getattr(res2, 'text', getattr(res2, 'content', str(res2))))

if __name__ == "__main__":
    main()
