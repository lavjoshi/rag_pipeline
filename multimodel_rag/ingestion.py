
import os
import json
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma



load_dotenv()

def create_ai_enhanced_summary(text:str, tables, images):
    llm = ChatOllama(model="qwen3.6")

    prompt = f"""You are an assistant that summarizes content for retrieval. Please create a concise summary of the following content. If there are tables, summarize the key insights from the tables. If there are images, describe what the images depict based on their captions or any available metadata.\n\n
                Content to analyze:\n
                Text Content:\n{text}\n\n
                """
    
    if tables:
        prompt += f"Tables content:\n"
        for i, table in enumerate(tables):
            prompt += f"Table {i+1}:\n{table}\n\n"

        prompt+= """
            Your task:
            
            Generate a concise summary and searchable description that covers:
            - Key insights from the text content.
            - Important information from the tables (e.g., trends, comparisons, key data points).
            - Descriptions of any images based on their captions or metadata.
            - Ensure the summary is informative and captures the essence of the content for effective retrieval.
            - Questions this content could answer.

            Make it detailed and searchable
        
        """
    print("Prompt for LLM:")
    print(prompt)
    
    message_content = [{"type": "text", "text": prompt}]

    for image in images:
        message_content.append({"type": "image_url", "image_url": f"data:image/png;base64,{image}"})

    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    print("LLM response:")  
    print(response)
    return response.content



def partition_pdf_document(file_path: str):
    elements = partition_pdf(filename=file_path, strategy="hi_res",
                             infer_table_structure=True,
                             extract_image_block_types=["Image"],
                             extract_image_block_to_payload=True)
    return elements

def chunk_by_title_elements(elements):
    chunks = chunk_by_title(elements=elements,
                            max_characters=3000,
                            new_after_n_chars=2400,
                            combine_text_under_n_chars=500)
    return chunks

def get_chunk_contents(chunk):
    data = {
        'text' :chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }

    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for orig_element in chunk.metadata.orig_elements:
            
            element_type = orig_element.to_dict().get('type', 'unknown')
            
            if element_type == 'Table':
                html_table = getattr(orig_element.metadata, 'text_as_html', orig_element.text)
                data['tables'].append(html_table)
                data['types'].append('table')
            elif element_type == 'Image':
                if hasattr(orig_element, 'metadata') and hasattr(orig_element.metadata, 'image_base64'):
                    image = getattr(orig_element.metadata, 'image_base64', None)
                    if image:
                        data['images'].append(image)
                        data['types'].append('image')
            
    data['types'] = list(set(data['types']))
    print(f"Chunk  data:", data)
    return data


def summarize_chunks(chunks):
    langchain_document = []
    for chunk in chunks:
        content_data = get_chunk_contents(chunk)
         
        if content_data['table'] or content_data['images']:
            summarized_content = create_ai_enhanced_summary(content_data['text'], content_data['tables'], content_data['images'])
        else:
            summarized_content = content_data['text']

        doc = Document(
            page_content=summarized_content,
            metadata={
                "original_content": json.dumps({
                    "raw_text": content_data['text'],
                    "tables_html": content_data['tables'],
                    "images_base64": content_data['images'],
                })
            }
        )
        langchain_document.append(doc)
    return langchain_document

def create_vector_store(chunks, persist_dir: str = "db/chroma_db_v2", model: str = "gemini-embedding-001"):
    match model.lower():
        case "gemini-embedding-001":
            embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        case "qwen3-embedding":
            embedding_model = OllamaEmbeddings(model="qwen3-embedding")

        case _:
            raise ValueError(
                f"Unsupported embedding model: {model}. "
                "Supported values: gemini-embedding-001"
            )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store created at: {persist_dir}")
    return vector_store


def generate_answer(query, chunks):


    llm = ChatOllama(model="qwen3.6")
    prompt = f"""You are an assistant that answers questions based on retrieved content chunks. Please provide a comprehensive answer to the following question using the information from the content. If the content contain tables, extract key insights from them. If there are images, describe what they depict based on any available metadata. Ensure your answer is detailed and directly addresses the question.
    Question: {query}  
    
    Content:"""

    for i, chunk in enumerate(chunks):
        prompt += f"\nDocument {i+1}:\n\n"
        if chunk.metadata.get("original_content"):
            original_content = json.loads(chunk.metadata["original_content"])
            
            raw_text = original_content.get("raw_text", "")
            if raw_text:
               prompt += f"Text Content:\n{raw_text}\n\n"
            
            tables = original_content.get("tables_html", [])
            
            if tables:
                prompt += "Tables \n"
                for j, table in enumerate(tables):
                    prompt += f"Table {j+1}:\n{table}\n"
            
            prompt += "\n\n"
    
    message_content = [{"type": "text", "text": prompt}]
    print("Prompt for LLM:")
    print(prompt)
    for chunk in chunks:
        if chunk.metadata.get("original_content"):
            original_content = json.loads(chunk.metadata["original_content"])
            images = original_content.get("images_base64", [])
            for image in images:
                message_content.append({"type": "image_url", "image_url": f"data:image/png;base64,{image}"})
    
    message = HumanMessage(content=message_content)
    response = llm.invoke([message])
    print("LLM response:")
    print(response.content)
    return response.content


if __name__ == "__main__":
    elements = partition_pdf_document("docs/sample.pdf")
    chunks = chunk_by_title_elements(elements)
    langchain_docs = summarize_chunks(chunks)
    vector_store = create_vector_store(langchain_docs, model="qwen3-embedding")    
    query = "What are the parameters for a Transformer base model?"
    answer = generate_answer(query, langchain_docs)
    print("Final Answer:")
    print(answer)