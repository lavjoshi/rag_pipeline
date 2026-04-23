
import json
import os
from typing import Iterable
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
load_dotenv()


class AgenticChunker:
        def __init__(self, model_name: str = "gemini-3-flash-preview"):
            match model_name:
                case "gemini-3-flash-preview" | "gemini-3-flash":
                    self.llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
                case "qwen3.6" | "qwen3":
                    self.llm = ChatOllama(model="qwen3.6")
                case _:
                    raise ValueError(f"Unsupported model name: {model_name}. Supported values: gemini-3-flash-preview, gemini-3-flash, qwen3.6, qwen3")

        def split_documents(self, documents: Iterable[Document]):
            all_chunks = []
            for document in documents:
                system_prompt = "You are an assistant that splits the given document into semantically meaningful chunks. Return the chunks as a JSON array of strings, no other text."
                user_prompt = f"{system_prompt}\n\nDocument:\n```{document.page_content}```"
                
                print("Invoking LLM with the following prompt:")
                print(user_prompt)
                
                response = self.llm.invoke(user_prompt)
                chunks = response.content_blocks[0]['text']
                try:
                    chunk_list = json.loads(chunks)
                    all_chunks.extend(chunk_list)
                except json.JSONDecodeError:
                    print("Failed to parse chunks as JSON. Response was:")
                    print(chunks)
            
            return all_chunks
              

def main():
    documents = [Document(page_content="This is a sample document. It contains multiple sentences. The goal is to split it into meaningful chunks. I'm testing the AgenticChunker. It should return chunks that are semantically coherent. It's important that the chunks are not too long or too short. Let's see how it performs.")]
    chunker = AgenticChunker("qwen3.6")
    chunks = chunker.split_documents(documents)
    print("Generated Chunks:")
    for index, chunk in enumerate(chunks, start=1):
        print(f"{index}. {chunk}")

if __name__ == "__main__":
    main()