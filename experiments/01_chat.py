import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def main():
    # Initialize the Gemini chat model as shown in main.py
    model = init_chat_model("google_genai:gemini-3.6-flash")

    print("🤖 Chatting with Gemini 3.6 Flash (Type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
            
        try:
            # For langchain Chat Models, the response is typically an AIMessage object
            # We access the text content using .content
            response = model.invoke(user_input)
            
            # Handle both `.text` (used in main.py) and standard Langchain `.content`
            content = getattr(response, 'text', getattr(response, 'content', str(response)))
            print(f"AI: {content}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
