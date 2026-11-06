import os
import shutil
import tempfile
from typing import List, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import TokenTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class RAGAgent:
    """
    Retrieval-Augmented Generation (RAG) agent for querying documents using Gemini.
    """
    LLM_MODEL = "gemini-2.5-flash"
    EMBEDDING_MODEL = "models/gemini-embedding-001"  # change to "models/embedding-001" if this is not found
    USE_LOCAL_EMBEDDINGS = False  # kept so app.py does not break; local embeddings removed
    CHROMA_DB_PATH = "chroma_db"

    PROMPT_TEMPLATE = """You are a helpful assistant answering questions based on provided documents.
Use only the context below. If the answer is not in the context, say you could not find it in the documents.

Context from documents:
{context}

Question: {question}

Answer:"""

    def __init__(self):
        """Initialize the RAG agent with LLM and vector store."""
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set your Google API key to use the RAG Agent."
            )

        self.llm = ChatGoogleGenerativeAI(
            model=self.LLM_MODEL,
            temperature=0.3,
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.EMBEDDING_MODEL)

        self.text_splitter = TokenTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            encoding_name="cl100k_base",
        )

        self.prompt = PromptTemplate(
            template=self.PROMPT_TEMPLATE,
            input_variables=["context", "question"],
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

        self.vector_store = self._initialize_vector_store()
        self._setup_retriever()

    def _initialize_vector_store(self) -> Chroma:
        """Initialize or load the Chroma vector store."""
        return Chroma(
            persist_directory=self.CHROMA_DB_PATH,
            embedding_function=self.embeddings,
        )

    def _setup_retriever(self):
        """MMR retriever: relevant and diverse passages."""
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "fetch_k": 20,
                "lambda_mult": 0.5,
            },
        )

    def process_file(self, uploaded_file) -> int:
        """
        Process an uploaded PDF or TXT file and add it to the vector store.
        Returns the number of chunks added.
        """
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            suffix = ".pdf"
        elif file_name.endswith(".txt"):
            suffix = ".txt"
        else:
            raise ValueError("Unsupported file type. Only PDF and TXT files are supported.")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
            else:
                loader = TextLoader(tmp_path, encoding="utf-8")

            documents = loader.load()

            # The temp path would otherwise show up as the "source" name
            for doc in documents:
                doc.metadata["source"] = uploaded_file.name

            chunks = self.text_splitter.split_documents(documents)
            self.vector_store.add_documents(chunks)

            if hasattr(self.vector_store, "persist"):
                self.vector_store.persist()

            self._setup_retriever()
            return len(chunks)
        finally:
            os.unlink(tmp_path)

    def query(self, question: str, chat_history: List[Dict]) -> Dict:
        """
        Query the RAG system. Returns a dict with 'answer' and 'sources'.
        """
        try:
            if self.get_document_count() == 0:
                return {
                    "answer": "No documents have been indexed yet. Please upload a document first.",
                    "sources": [],
                }

            # Retrieve once, and use the same docs for the answer and the sources
            docs = self.retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            answer = self.chain.invoke({"context": context, "question": question})

            sources = []
            for doc in docs:
                source_name = doc.metadata.get("source", "Unknown")
                if isinstance(source_name, str):
                    source_name = source_name.replace("\\", "/").split("/")[-1]

                # PyPDF pages are 0-indexed, so add 1. TXT files have no pages.
                if "page" in doc.metadata:
                    page = doc.metadata["page"] + 1
                else:
                    page = "N/A"

                sources.append({"source": source_name, "page": page})

            return {"answer": answer, "sources": sources}
        except Exception as e:
            import traceback
            print(f"Query error: {traceback.format_exc()}")
            return {
                "answer": f"Error processing query: {str(e)}\n\nPlease check the console for more details.",
                "sources": [],
            }

    def get_document_count(self) -> int:
        """Get the number of chunks in the vector store."""
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
            # Deleting the collection avoids Windows file-lock errors from rmtree
            self.vector_store.delete_collection()
        except Exception as e:
            print(f"Error deleting collection: {e}")
            if os.path.exists(self.CHROMA_DB_PATH):
                shutil.rmtree(self.CHROMA_DB_PATH, ignore_errors=True)

        self.vector_store = self._initialize_vector_store()
        self._setup_retriever()
