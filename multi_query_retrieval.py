
import os, json
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings


load_dotenv()

def reciprocal_rank_fusion(all_retrieved_docs, k=60, top_k=3):
    doc_scores = {}
    for docs in all_retrieved_docs:
        for rank, doc in enumerate(docs):
            score = 1 / (rank + 60)  # Reciprocal of the rank
            doc_id = doc.metadata.get("source", "") + doc.page_content[:100]  # Unique identifier for the document
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += score
            else:
                doc_scores[doc_id] = {'score': score, 'document': doc}

    # Sort documents by their cumulative scores and return the top_k
    sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
    return [entry['document'] for entry in sorted_docs[:top_k]]

llm = ChatOllama(model="qwen3.6")
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = Chroma(
        persist_directory="db/chroma_db",
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

query = "Why Nvida stock soared in last 3 years?"

prompt = f"""Generate 3 variation of this query that would help retrieve relavant documents: 

        Original query:    {query}

        Return 3 query that are semantically similar but use different wording. Just return the queries as a JSON array of strings, no other text.
"""
print(prompt)
response = llm.invoke(prompt)
print("LLM response:")

arr = json.loads(response.content)
print(arr)

all_retrieved_docs = []

for query in arr:
    docs = vector_store.as_retriever(search_kwargs={"k": 5}).invoke(query)
    print(f"Retrieved {len(docs)} documents for query: {query}")
    all_retrieved_docs.append(docs)

final_docs = reciprocal_rank_fusion(all_retrieved_docs, k=60, top_k=3)
print(f"Final retrieved documents after Reciprocal Rank Fusion: {len(final_docs)}")
print("Top retrieved documents:")
for index, doc in enumerate(final_docs, start=1):
    print(f"{index}. Source: {doc.metadata.get('source', 'unknown')}, Content: {doc.page_content[:200]}...")

