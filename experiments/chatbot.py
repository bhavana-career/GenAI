import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# --- Personalities Dictionary ---
PERSONALITIES = {
    "1": ("Helpful Assistant", "You are a helpful, friendly, and concise AI assistant."),
    "2": ("Grumpy Old Man", "You are a grumpy old man who complains about modern technology, but you still answer the questions."),
    "3": ("Space Pirate", "You are a swaggering space pirate. Use pirate slang but talk about sci-fi concepts."),
    "4": ("Sarcastic Genius", "You are a super-genius who answers questions correctly but with heavy sarcasm and condescension."),
    "5": ("Poet", "You are a classical poet. All your responses must be beautifully written, poetic, and preferably rhyme.")
}

def main():
    print("Initializing Groq Chatbot (Llama 3.3)...")
    try:
        model = init_chat_model("groq:llama-3.3-70b-versatile")
    except Exception as e:
        print(f"Failed to initialize Groq model: {e}")
        print("Please ensure your GROQ_API_KEY is set in the .env file.")
        return

    # --- Terminal Menu for Selecting Personality ---
    print("\n" + "="*50)
    print("🎭 CHOOSE A PERSONALITY")
    print("="*50)
    for key, (name, _) in PERSONALITIES.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter the number of the personality (default is 1): ").strip()
    
    # Fallback to Helpful Assistant if they type something invalid
    if choice not in PERSONALITIES:
        choice = "1"
        
    chosen_name, chosen_system_prompt = PERSONALITIES[choice]
    print(f"\n>> You selected: {chosen_name} <<")

    # We store the conversation history here, injecting the dynamic personality prompt!
    chat_history = [
        SystemMessage(content=chosen_system_prompt)
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
