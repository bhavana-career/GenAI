import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

# Take the path
file_path = input("Enter the file path (or press Enter for default sample.txt): ")
if not file_path.strip():
    file_path = r"C:\Users\bhava\OneDrive\Documents\Downloads\AI\sample.txt"

# Load the document
data = TextLoader(file_path)
doc = data.load() 
content = doc[0].page_content

# System prompt template will generate the output
model = init_chat_model("google_genai:gemini-3.6-flash")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Analyze or summarize the following document content."),
    ("user", "{content}")
])

chain = prompt | model
response = chain.invoke({"content": content})

print("Generated Output:")
print("-" * 50)
print(getattr(response, 'text', getattr(response, 'content', str(response))))
