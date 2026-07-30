import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import TokenTextSplitter

def main():
    # 1. Take the path and load the document
    file_path = input("Enter the file path (or press Enter for default sample.txt): ")
    if not file_path.strip():
        file_path = r"C:\Users\bhava\OneDrive\Documents\Downloads\AI\sample.txt"
        
    try:
        loader = TextLoader(file_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} document(s).")
        print(f"Original document character count: {len(docs[0].page_content)}\n")
        
        # 2. Initialize the Token Text Splitter
        # This splits the text based on the number of tokens (using OpenAI's tiktoken encoding by default)
        # LLMs have context windows based on tokens, not characters, so this is often more precise for fitting into context limits.
        token_splitter = TokenTextSplitter(
            chunk_size=50,       # Maximum number of tokens per chunk
            chunk_overlap=10     # Number of tokens to overlap between chunks
        )
        
        # 3. Split the document into token-sized chunks
        chunks = token_splitter.split_documents(docs)
        
        print(f"--- Token Splitting Results ---")
        print(f"Total chunks created: {len(chunks)}\n")
        
        # Print out the first few chunks to inspect the result
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} (Character length: {len(chunk.page_content)}):")
            print(chunk.page_content)
            print("-" * 40)
            
        if len(chunks) > 3:
            print(f"... and {len(chunks) - 3} more chunks.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
