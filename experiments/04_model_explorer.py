import os
import csv
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

def get_provider_prefix(provider):
    mapping = {
        'Google': 'google_genai',
        'Groq': 'groq',
        'Mistral': 'mistralai'
    }
    return mapping.get(provider, '')

def main():
    # Find available_models.csv in the root folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root_dir, 'available_models.csv')
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # List of known free/working models based on the CSV
    known_working_models = [
        'models/gemini-3.6-flash',
        'llama-3.3-70b-versatile',
        'mistral-small-latest'
    ]
    
    available_models = []
    
    # Parse the CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Model'] in known_working_models and row['Capability'] == 'Chat':
                available_models.append(row)

    if not available_models:
        print("No matching free models found in CSV.")
        return

    print("Model Explorer")
    print("=" * 40)
    for i, m in enumerate(available_models):
        print(f"{i+1}. {m['Provider']} - {m['Model']}")
        
    choice = input("\nSelect a model number to test (or 'q' to quit): ")
    if choice.lower() == 'q':
        return
        
    try:
        idx = int(choice) - 1
        selected_model = available_models[idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    provider = selected_model['Provider']
    raw_model_name = selected_model['Model']
    
    # Strip 'models/' prefix for Google if present, init_chat_model handles it
    clean_model_name = raw_model_name.replace("models/", "")
    provider_prefix = get_provider_prefix(provider)
    
    full_init_string = f"{provider_prefix}:{clean_model_name}"
    print(f"\nInitializing -> {full_init_string}")
    
    try:
        model = init_chat_model(full_init_string)
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
        
    print("\nModel is ready! Type 'quit' to exit.")
    while True:
        user_msg = input("You: ")
        if user_msg.lower() in ['quit', 'exit']:
            break
            
        try:
            response = model.invoke(user_msg)
            print(f"AI: {getattr(response, 'text', getattr(response, 'content', str(response)))}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
