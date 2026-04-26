
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document;
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from flashrank import Ranker, RerankRequest

load_dotenv()
llm = ChatOllama(model="qwen3.6")
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
)


def rerank(query, passages):
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/Users/joshil/personal/learning/RAG/temp")
        rerankrequest = RerankRequest(query=query, passages=passages)
        return ranker.rerank(rerankrequest)




documents = vector_store._collection.get(include=["documents"]);
langchain_docs = [Document(page_content=doc) for doc in documents["documents"]]


bm25_retriever = BM25Retriever.from_documents(langchain_docs)
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

ensemble_retriever = EnsembleRetriever(retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5])

query= "What is Nvidia's most successful product?"
chunks = ensemble_retriever.invoke(query, kwargs={"verbose": True})
print(f"Retrieved {len(chunks)} chunks from Ensemble Retriever")
print("Sample retrieved chunk content:")

passages = [{"id": index, "text": chunk.page_content} for index, chunk in enumerate(chunks, start=1)]

for chunk in passages:
    print(f"ID: {chunk['id']}, Text: {chunk['text'][:200]}...")

reranked_chunks = rerank(query, passages)
reranked_chunks = sorted(reranked_chunks, key=lambda x: x["score"], reverse=True)

print("Reranked chunks:")
for chunk in reranked_chunks:
    print(f"ID: {chunk['id']}, Score: {chunk['score']}, Text: {chunk['text'][:200]}...")



