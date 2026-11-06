# Gemini RAG Chatbot

A **Retrieval-Augmented Generation (RAG)** app that answers questions from your own documents, built with **Gemini 2.5 Flash**, **LangChain** and **ChromaDB**.

## Features
* **Document upload**: Index PDF and TXT files.
* **Token-based chunking**: Uses `TokenTextSplitter` (512 tokens, 50 overlap).
* **MMR retrieval**: Maximum Marginal Relevance returns passages that are both relevant and varied.
* **Source citations**: Shows the source filename and page number with answers.
* **Flexible embeddings**:
  * Local: HuggingFace `all-mpnet-base-v2` (free, runs on CPU).
  * Cloud: Google Generative AI embeddings.
* **Chat history**: Stores per-user chat history through Firebase.

## Tech Stack
* Frontend: Streamlit
* Orchestration: LangChain
* LLM: Gemini 2.5 Flash
* Vector store: ChromaDB

## Getting Started

1. Use the project virtual environment instead of the system Python. The app was verified against the workspace venv at `myenv`.
   ```powershell
   cd C:\knowledge_base
   .\myenv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. Create a `.env` file:
   ```env
   GOOGLE_API_KEY=your_key_here
   USE_LOCAL_EMBEDDINGS=false
   ```

3. Run the app from the same venv:
   ```powershell
   cd C:\knowledge_base
   .\myenv\Scripts\python.exe -m streamlit run app.py
   ```
