import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

# 1. Define the Pydantic Model (The Blueprint)
class PersonInfo(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age as an integer")
    hobbies: list[str] = Field(description="A list of the person's hobbies")

def main():
    print("Initializing Groq Model...")
    try:
        model = init_chat_model("groq:llama-3.3-70b-versatile")
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        return
    
    # 2. Set up the Output Parser using our Pydantic model
    parser = PydanticOutputParser(pydantic_object=PersonInfo)
    
    # 3. Create a Prompt that automatically includes formatting instructions
    prompt = PromptTemplate(
        template="Extract the information from the following text.\n{format_instructions}\n\nText: {text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    # 4. Connect everything together (Prompt -> Model -> Parser)
    chain = prompt | model | parser
    
    sample_text = "My name is Alice Smith, I am 28 years old, and I love hiking, reading sci-fi novels, and playing the guitar on weekends."
    
    print("\nReading Text:")
    print(f'"{sample_text}"')
    print("-" * 50)
    
    try:
        # 5. Execute the chain
        result = chain.invoke({"text": sample_text})
        
        # The result is perfectly extracted and converted into correct Python data types
        print("Success! Data successfully extracted into Python variables:")
        print(f"Name:    {result.name} (Type: {type(result.name).__name__})")
        print(f"Age:     {result.age} (Type: {type(result.age).__name__})")
        print(f"Hobbies: {result.hobbies} (Type: {type(result.hobbies).__name__})")
        
        print("\nRaw Pydantic Object Representation:")
        print(result)
        
    except Exception as e:
        print(f"Error parsing output: {e}")

if __name__ == "__main__":
    main()
