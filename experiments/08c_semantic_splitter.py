import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

def main():
    # We will use a longer sample text to demonstrate the difference
    sample_text = """Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

The term "artificial intelligence" had previously been used to describe machines that mimic and display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.

Various sub-fields of AI research are centered around particular goals and the use of particular tools. The traditional problems (or goals) of AI research include reasoning, knowledge representation, planning, learning, natural language processing, perception, and the ability to move and manipulate objects.
"""
    
    print("=== Text Splitting Comparison ===")
    print(f"Original text length: {len(sample_text)} characters\n")

    # 1. Recursive Character Text Splitter
    print("--- 1. Recursive Character Text Splitter ---")
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=30,
    )
    
    recursive_chunks = recursive_splitter.create_documents([sample_text])
    print(f"Total chunks created: {len(recursive_chunks)}")
    for i, chunk in enumerate(recursive_chunks):
        print(f"\nChunk {i+1} (Length: {len(chunk.page_content)}):")
        print(chunk.page_content)
        
    print("\n" + "="*50 + "\n")

    # 2. Semantic Text Splitter
    print("--- 2. Semantic Chunker ---")
    print("Initializing Mistral Embeddings for Semantic Chunker...")
    try:
        embeddings = MistralAIEmbeddings(model="mistral-embed")
        
        # Semantic chunker uses embeddings to find semantic similarities between sentences
        # and groups them into chunks accordingly.
        semantic_splitter = SemanticChunker(
            embeddings, 
            breakpoint_threshold_type="percentile"
        )
        
        semantic_chunks = semantic_splitter.create_documents([sample_text])
        print(f"Total chunks created: {len(semantic_chunks)}")
        for i, chunk in enumerate(semantic_chunks):
            print(f"\nChunk {i+1} (Length: {len(chunk.page_content)}):")
            print(chunk.page_content)
            
    except Exception as e:
        print(f"Failed to use Semantic Chunker. Ensure you have your MISTRAL_API_KEY set. Error: {e}")

if __name__ == "__main__":
    main()
