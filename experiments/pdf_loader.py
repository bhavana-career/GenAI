import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    # 1. Provide the path and load the PDF document
    pdf_path = r"AI\Resume.pdf"
    
    try:
        print(f"Loading PDF from: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        print(f"Loaded {len(docs)} page(s) from the PDF.")
        total_chars = sum(len(doc.page_content) for doc in docs)
        print(f"Original document character count: {total_chars}\n")
        
        # 2. Initialize the Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,      # Maximum size of each chunk
            chunk_overlap=40,    # Overlap to preserve context between chunks
            length_function=len,
        )
        
        # 3. Split the PDF pages into smaller chunks
        chunks = text_splitter.split_documents(docs)
        
        print(f"--- Text Splitting Results ---")
        print(f"Total chunks created: {len(chunks)}\n")
        
        # Print out the first few chunks to inspect the result
        for i, chunk in enumerate(chunks[:3]):
            print(f"Chunk {i+1} (Length: {len(chunk.page_content)}):")
            print(chunk.page_content)
            # You can also see which page this chunk came from!
            print(f"[Source: Page {chunk.metadata.get('page', 'Unknown')}]")
            print("-" * 40)
            
        if len(chunks) > 3:
            print(f"... and {len(chunks) - 3} more chunks.")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure the 'pypdf' package is installed (uv pip install pypdf)")

if __name__ == "__main__":
    main()