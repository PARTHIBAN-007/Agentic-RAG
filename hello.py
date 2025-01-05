import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
import tempfile
from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, List
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from time import sleep
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
COLLECTION_NAME = "cohere_rag"

if not cohere_api_key or not qdrant_api_key or not qdrant_url:
    st.error("API keys not found in .env file. Please configure them correctly.")
    st.stop()

# Initialize Qdrant client
def init_qdrant() -> QdrantClient:
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)

# Initialize embedding and chat model
embedding = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=cohere_api_key)
chat_model = ChatCohere(model="command-r7b-12-2024", temperature=0.1, max_tokens=512, verbose=True, cohere_api_key=cohere_api_key)

client = init_qdrant()

# Process uploaded document
def process_document(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        os.unlink(tmp_path)
        return texts
    except Exception as e:
        st.error(f"Error processing document: {e}")
        return []

# Create vector stores
def create_vector_stores(texts):
    try:
        try:
            client.create_collection(collection_name=COLLECTION_NAME, vectors_config=VectorParams(size=1024, distance=Distance.COSINE))
            st.success(f"Created new collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e
        
        vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embedding)
        
        with st.spinner('Storing documents in Qdrant...'):
            vector_store.add_documents(texts)
            st.success("Documents successfully stored in Qdrant!")
        
        return vector_store
    except Exception as e:
        st.error(f"Error in vector store creation: {e}")
        return None

# Query processing
def process_query(vectorstore, query) -> tuple[str, list]:
    try:
        retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 10, "score_threshold": 0.7})
        relevant_docs = retriever.get_relevant_documents(query)

        if relevant_docs:
            retrieval_qa_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(chat_model, retrieval_qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            response = retrieval_chain.invoke({"input": query})
            return response['answer'], relevant_docs
        else:
            fallback_agent = create_react_agent(chat_model, [DuckDuckGoSearchRun(num_results=5)], debug=False)
            fallback_response = fallback_agent.invoke({"messages": [HumanMessage(content=query)]})
            return fallback_response.content, []
    except Exception as e:
        st.error(f"Error: {e}")
        return "I encountered an error. Please try again.", []

# Streamlit app setup
st.title("RAG Agent with Cohere ⌘R")

uploaded_file = st.file_uploader("Choose a PDF File", type=["pdf"])

if uploaded_file:
    texts = process_document(uploaded_file)
    vectorstore = create_vector_stores(texts)
    if vectorstore:
        st.session_state.vectorstore = vectorstore
        st.success("File uploaded and processed successfully!")

if "vectorstore" in st.session_state:
    if query := st.text_input("Ask a question about the document:"):
        answer, sources = process_query(st.session_state.vectorstore, query)
        st.write("Answer:", answer)
        if sources:
            st.write("Sources:")
            for source in sources:
                st.markdown(f"- {source.page_content[:200]}...")
