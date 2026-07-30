from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader(r"C:\Users\bhava\OneDrive\Documents\Downloads\AI\Resume.pdf")

docs=data.load()


print(len(docs))