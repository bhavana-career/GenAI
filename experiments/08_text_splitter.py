import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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
        
        # 2. Initialize the Text Splitter
        # RecursiveCharacterTextSplitter is highly recommended for generic text.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=150,      # Maximum size of each chunk
            chunk_overlap=30,    # Overlap between consecutive chunks to preserve context
            length_function=len,
            separators=["\n\n", "\n", " ", ""] # Order of preference for splitting
        )
        
        # 3. Split the document into smaller chunks
        chunks = text_splitter.split_documents(docs)
        
        print(f"--- Text Splitting Results ---")
        print(f"Total chunks created: {len(chunks)}\n")
        
        # Print out the first few chunks to inspect the result
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} (Length: {len(chunk.page_content)}):")
            print(chunk.page_content)
            print("-" * 40)
            
        if len(chunks) > 3:
            print(f"... and {len(chunks) - 3} more chunks.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
