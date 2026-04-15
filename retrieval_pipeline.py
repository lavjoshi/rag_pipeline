
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


load_dotenv()

chat_history = []
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

def ask_llm(query: str, relevant_docs: list):
    
    system_prompt = "You are a helpful assistant that provides concise answers based on the provided context."
    context = "\n\n".join([f"Document {index + 1}:\n{doc.page_content}" for index, doc in enumerate(relevant_docs)])
    user_prompt = f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}"
    
    print("Invoking LLM with the following prompt:")
    print(user_prompt)
    
    response = llm.invoke(user_prompt)
    return response.content_blocks[0]['text']

def retrieve_doc(query: str, top_k: int = 3, persist_dir: str = "db/chroma_db"):

    embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(query)
    return relevant_docs

def ask(query: str):
    print(f"You asked: {query}")
    question = query
    if(len(chat_history) > 0):
        messages = [    SystemMessage(content="Given the chat history, rewrire the question to be searchable. Just return rephrased question.")
                    ] + chat_history + [HumanMessage(content=f"New Question: {question}")]
        result = llm.invoke(messages)
        question = result.content_blocks[0]['text']
        print(f"Searchable question: {question}")
    
        
    relevant_docs = retrieve_doc(question)
    answer = ask_llm(question, relevant_docs)
    return answer


def chat():
    print("Ask a question, type 'exit' to quit:")
    
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        answer = ask(query)
        print("---Answer---")
        print(answer)
        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=answer))

if __name__ == "__main__":
    chat()