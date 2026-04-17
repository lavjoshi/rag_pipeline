
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from agentic_chunker import AgenticChunker

load_dotenv()


def get_docs_directory() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "docs")


def load_documents(docs_dir: str):
    loader = DirectoryLoader(
        docs_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents = loader.load()

    print("Loaded document names:")
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        print(f"{index}. {os.path.basename(source)}")

    return documents

def get_separator(chunk_size: int, chunk_overlap: int, type: str = "character"):
    match type.lower():
        case "character" | "char":
            return CharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        case "recursive_character" | "recursive":
            return RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". "," ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        case "semantic_chunker" | "semantic":
            return SemanticChunker(
                embeddings=GoogleGenerativeAIEmbeddings(model="gemini-embedding-001"),
                breakpoint_threshold_type="percent",
                breakpoint_threshold_amount=70
            )
        case "agentic" | "agent":
            return AgenticChunker()
        case _:
            raise ValueError(
                f"Unsupported splitter type: {type}. "
                "Supported values: character, recursive_character, agentic"
            )



def split_documents(documents, chunk_size: int, chunk_overlap: int):
    splitter = get_separator(chunk_size, chunk_overlap)

    all_chunks = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        document_name = os.path.basename(source)
        document_chunks = splitter.split_documents([document])
        print(f"{document_name}: {len(document_chunks)} chunks created")

        # if document_chunks:
        #     print(f"{document_name} - first chunk:\n{document_chunks[0].page_content}\n")
        #     print(f"{document_name} - last chunk:\n{document_chunks[-1].page_content}\n")

        all_chunks.extend(document_chunks)

    return all_chunks


def create_vector_store(chunks, persist_dir: str = "db/chroma_db"):
    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Vector store created at: {persist_dir}")
    return vector_store


def main():
    print("Running the ingestion pipeline...")
    docs_dir = get_docs_directory()
    documents = load_documents(docs_dir)
    chunks = split_documents(documents, chunk_size=800, chunk_overlap=0)
    print(f"Total chunks created: {len(chunks)}")
    vector_store = create_vector_store(chunks)


if __name__ == "__main__":
    main()
    