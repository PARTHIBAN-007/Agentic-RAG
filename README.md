# Agentic RAG

## Project Description
Agentic RAG (Retrieval-Augmented Generation) is a Streamlit-based application that leverages Cohere's language models and Qdrant's vector database to process documents, retrieve relevant information, and answer user queries. The application supports uploading PDF files, processing them to extract text, and storing the text in a vector database for efficient retrieval. It also provides fallback to web search using DuckDuckGo for queries not covered by the stored documents.

## Features
- Upload and process PDF documents to extract text.
- Store extracted text in Qdrant vector database.
- Retrieve relevant information from stored documents based on user queries.
- Fallback to web search using DuckDuckGo for queries not covered by the stored documents.
- Interactive chat interface for asking questions about the uploaded document.
- Clear chat history and all stored data functionality.

## Installation Instructions
To install and set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/PARTHIBAN-007/Agentic-RAG.git
    ```

2. Navigate into the project directory:
    ```sh
    cd Agentic-RAG
    ```

3. Create and activate a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

4. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

5. Set up environment variables by creating a `.env` file in the project directory with the following content:
    ```env
    COHERE_API_KEY=your_cohere_api_key
    QDRANT_API_KEY=your_qdrant_api_key
    QDRANT_URL=your_qdrant_url
    ```

## Usage Instructions
To use the project, follow these steps:

1. Run the Streamlit application:
    ```sh
    streamlit run app.py
    ```

2. Upload a PDF or image file using the file uploader in the web interface.

3. Ask questions about the uploaded document using the chat input.

4. View the answers and sources in the chat interface.

5. Use the sidebar buttons to clear chat history or all stored data.



## Credits
- [Cohere](https://cohere.ai) for providing the language models.
- [Qdrant](https://qdrant.tech) for providing the vector database.
- [LangChain](https://github.com/langchain-ai/langchain) for the document processing and retrieval framework.
- [Streamlit](https://streamlit.io) for the web application framework.
- [DuckDuckGo](https://duckduckgo.com) for the web search functionality.
