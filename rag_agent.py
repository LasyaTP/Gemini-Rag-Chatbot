import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
import shutil

load_dotenv()

class RAGAgent:
    """
    Retrieval-Augmented Generation (RAG) Agent for querying documents using Google's Gemini API.
    """
    LLM_MODEL = "gemini-2.5-flash"  # Using gemini-2.5-flash as it's the best price-performance model
    EMBEDDING_MODEL = "models/embedding-001"
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Upgraded for better semantic understanding
    CHROMA_DB_PATH = "chroma_db"

    def __init__(self):
        """Initialize the RAG Agent with LLM and vector store."""
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set your Google API key to use the RAG Agent."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=self.LLM_MODEL,
            temperature=0.3,
            convert_system_message_to_human=True
        )

        if self.USE_LOCAL_EMBEDDINGS:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.LOCAL_EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        else:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=self.EMBEDDING_MODEL
            )

        self.vector_store = self._initialize_vector_store()

        self.text_splitter = TokenTextSplitter(
            chunk_size=512,  # Tokens for better embedding alignment
            chunk_overlap=50,
            encoding_name="cl100k_base"
        )

        # Add cross-encoder for re-ranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.rag_chain = None
        self._setup_rag_chain()

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the Chroma vector store."""
        return Chroma(
            persist_directory=self.CHROMA_DB_PATH,
            embedding_function=self.embeddings
        )

    def _setup_rag_chain(self):
        """Set up the RAG chain with a custom prompt using LCEL."""
        prompt_template = """You are a helpful assistant answering questions based on provided documents.

Context from documents:
{context}

Question: {question}

Answer: Based on the context provided, """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = self.vector_store.as_retriever(
            search_type="mmr",  # Use MMR for relevance and diversity
            search_kwargs={
                "k": 10,  # Retrieve more for re-ranking
                "lambda_mult": 0.5,  # Balance relevance (0.5) and diversity
                "fetch_k": 20,  # Fetch more for MMR selection
                "search_type": "hybrid"  # Enable hybrid search (semantic + keyword)
            }
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )

        self.retriever = retriever

    def process_file(self, uploaded_file) -> int:
        """
        Process an uploaded PDF or TXT file and add it to the vector store.

        Args:
            uploaded_file: Streamlit UploadedFile object

        Returns:
            Number of chunks added
        """
        import tempfile

        file_name = uploaded_file.name.lower()
        if file_name.endswith('.pdf'):
            suffix = ".pdf"
        elif file_name.endswith('.txt'):
            suffix = ".txt"
        else:
            raise ValueError(f"Unsupported file type. Only PDF and TXT files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding='utf-8')

            documents = loader.load()

            chunks = self.text_splitter.split_documents(documents)

            # Add documents to vector store
            self.vector_store.add_documents(chunks)
            
            # Persist the vector store (for Chroma)
            if hasattr(self.vector_store, 'persist'):
                self.vector_store.persist()
            
            # Reinitialize the RAG chain with updated vector store
            self._setup_rag_chain()

            return len(chunks)
        finally:
            os.unlink(tmp_path)

    def query(self, question: str, chat_history: List[Dict]) -> Dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: User's question
            chat_history: Previous chat messages for context
            
        Returns:
            Dict with 'answer' and 'sources' keys
        """
        try:
            # Check if we have documents
            doc_count = self.get_document_count()
            if doc_count == 0:
                return {
                    "answer": "No documents have been indexed yet. Please upload a document first.",
                    "sources": []
                }
            
            # Get answer from chain
            answer = self.rag_chain.invoke(question)

            # Get source documents from retriever
            source_documents = self.retriever.invoke(question)

            # Extract sources
            sources = []
            for doc in source_documents:
                # For text files, page might not exist, use line number or default to 0
                page = doc.metadata.get("page", doc.metadata.get("line", 0))
                source_name = doc.metadata.get("source", "Unknown")
                # Extract just the filename from the full path
                if isinstance(source_name, str):
                    source_name = source_name.split("/")[-1].split("\\")[-1]
                
                sources.append({
                    "source": source_name,
                    "page": page
                })

            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Query error: {error_details}")  # Debug print
            return {
                "answer": f"Error processing query: {str(e)}\n\nPlease check the console for more details.",
                "sources": []
            }

    def get_document_count(self) -> int:
        """Get the number of documents/chunks in the vector store."""
        try:
            all_docs = self.vector_store.get()
            if all_docs and "ids" in all_docs:
                return len(all_docs["ids"])
            return 0
        except Exception as e:
            print(f"Error getting document count: {e}")
            return 0

    def clear_database(self):
        """Clear the vector database."""
        try:
            if os.path.exists(self.CHROMA_DB_PATH):
                shutil.rmtree(self.CHROMA_DB_PATH)

            self.vector_store = self._initialize_vector_store()
            self._setup_rag_chain()
        except Exception as e:
            print(f"Error clearing database: {e}")