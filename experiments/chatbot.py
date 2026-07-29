import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def main():
    # Initialize the Groq model using a capable, free model
    print("Initializing Groq Chatbot (Llama 3.3)...")
    try:
        model = init_chat_model("groq:llama-3.3-70b-versatile")
    except Exception as e:
        print(f"Failed to initialize Groq model: {e}")
        print("Please ensure your GROQ_API_KEY is set in the .env file.")
        return

    # We will store the conversation history here so the bot remembers previous messages
    chat_history = [
        SystemMessage(content="You are a helpful, friendly AI assistant powered by Groq.")
    ]

    print("\n" + "="*50)
    print("Groq Chatbot Ready! (Type 'quit' or 'exit' to stop)")
    print("="*50 + "\n")

    while True:
        # Prompt as requested by the user
        user_input = input("User: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("Chatbot: Goodbye!")
            break
            
        if not user_input.strip():
            continue
            
        # Add the user's message to the history
        chat_history.append(HumanMessage(content=user_input))
        
        try:
            # Pass the entire conversation history to the model
            response = model.invoke(chat_history)
            
            # Extract the text content safely
            ai_response = getattr(response, 'text', getattr(response, 'content', str(response)))
            
            print(f"Chatbot: {ai_response}")
            
            # Add the AI's response to the history so it remembers for the next turn
            chat_history.append(response)
            
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
