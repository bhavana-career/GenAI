from langchain_community.document_loaders import TextLoader

data = TextLoader(r"C:\Users\bhava\OneDrive\Documents\Downloads\AI\sample.txt")
print(data) #which data we have in text format

doc = data.load() 
print(doc)  #document have two parts metadata and text(page_content) in the list