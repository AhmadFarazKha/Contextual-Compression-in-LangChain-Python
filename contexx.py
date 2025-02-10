from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
from typing import List

def read_and_compress_text(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Read text from a file and compress it using LangChain's text splitter.
    
    Args:
        file_path (str): Path to the text file
        chunk_size (int): Maximum size of text chunks
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        List[Document]: List of compressed text chunks as LangChain Documents
    """
    try:
        # Read the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Initialize the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Split the text into chunks
        docs = text_splitter.create_documents([text])
        
        # Print the compressed output
        print("\nCompressed Text Chunks:")
        print("-" * 50)
        for i, doc in enumerate(docs, 1):
            print(f"\nChunk {i}:")
            print(doc.page_content)
            print("-" * 50)
        
        print(f"\nOriginal text length: {len(text)} characters")
        print(f"Number of chunks: {len(docs)}")
        print(f"Average chunk size: {sum(len(doc.page_content) for doc in docs) / len(docs):.2f} characters")
        
        return docs
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")
        return []

if __name__ == "__main__":
    # Using the provided Windows-style path
    file_path = "data\\notes.txt"
    compressed_docs = read_and_compress_text(file_path)