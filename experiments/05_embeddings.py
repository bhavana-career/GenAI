import os
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

def main():
    print("Initializing Mistral Embeddings...")
    try:
        # Mistral's embedding model is excellent for this
        embeddings = MistralAIEmbeddings(model="mistral-embed")
    except Exception as e:
        print(f"Failed to initialize embeddings: {e}")
        return

    word = "Apple"
    print(f"\nTurning the word '{word}' into a vector (a simple list of numbers)...\n")
    
    try:
        # Generate the embedding vector
        vector = embeddings.embed_query(word)
        
        total_dimensions = len(vector)
        print(f"The computer generated a full list of {total_dimensions} numbers.")
        
        # Slicing the list so it's not scary/overwhelming!
        print("\nBut since we requested less dimensions so we don't feel bad... 🎀")
        print("Here are just the first 5 numbers of that vector:\n")
        
        # This is literally just a standard Python list of floats!
        short_vector = vector[:5]
        
        # We print it out so you can see exactly what it looks like
        print(short_vector)
        
        print("\n...and it just goes on like that! Nothing more to it.")
        
    except Exception as e:
        print(f"Error generating embedding: {e}")

if __name__ == "__main__":
    main()
