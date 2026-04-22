import streamlit as st
from dotenv import load_dotenv
from rag_agent import RAGAgent
from firebase_utils import initialize_firebase_auth, load_chat_history, save_message, clear_chat_history, get_user_id

load_dotenv()

@st.cache_resource
def get_rag_agent():
    try:
        return RAGAgent()
    except ValueError as e:
        st.error(f"Initialization Error: {e}")
        st.stop()

USER_ID = initialize_firebase_auth()

st.set_page_config(page_title="Gemini RAG Chatbot with Streamlit", page_icon="🤖", layout="wide")

st.title("Gemini RAG chatbot")
st.caption(f"Chat with your documents using **Gemini**, **LangChain**, and **ChromaDB**. (User ID: {USER_ID})")

rag_agent = get_rag_agent()
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history(USER_ID)
if "rag_initialized" not in st.session_state:
    st.session_state.rag_initialized = rag_agent.get_document_count() > 0

def handle_upload():
    uploaded_file = st.session_state.file_uploader
    if uploaded_file is not None:
        with st.spinner(f"Indexing '{uploaded_file.name}'... This may take a moment."):
            try:
                chunks_added = rag_agent.process_file(uploaded_file)
                st.session_state.rag_initialized = True
                st.success(f"Successfully indexed '{uploaded_file.name}'! Added {chunks_added} knowledge chunks.")
            except Exception as e:
                st.error(f"Error during file processing: {e}")

def handle_clear_chat():
    clear_chat_history()
    st.session_state.messages = []
    st.rerun()

def handle_clear_db():
    rag_agent.clear_database()
    st.session_state.rag_initialized = False
    st.session_state.messages = []
    clear_chat_history()
    st.cache_resource.clear()
    st.success("Vector Database Cleared. Please upload new documents.")
    st.rerun()

with st.sidebar:
    st.header("1. Knowledge Base")
    st.file_uploader("Upload Document (PDF or TXT)", type=["pdf", "txt"], key="file_uploader", on_change=handle_upload, accept_multiple_files=False)
    doc_count = rag_agent.get_document_count()
    st.metric(label="Indexed Chunks in ChromaDB", value=f"{doc_count:,}")
    st.caption("Chunks are text snippets ready for retrieval.")
    if doc_count > 0:
        st.button("Clear Vector Database", on_click=handle_clear_db, help="Deletes the local chroma_db folder and resets the agent.")
    st.markdown("---")
    st.header("2. App Status")
    try:
        embedding_info = "Local (HuggingFace)" if RAGAgent.USE_LOCAL_EMBEDDINGS else f"Google ({RAGAgent.EMBEDDING_MODEL})"
        st.info(f"LLM: {RAGAgent.LLM_MODEL}\nEmbeddings: {embedding_info}")
    except:
        st.warning("RAG Agent models not fully initialized.")
    st.markdown("**Persistence (Simulated):** Firebase (via `st.session_state`)")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and message.get("is_rag"):
            sources_list = []
            for src in message["sources"]:
                sources_list.append(f"*{src['source']}* (p. {src['page']})")
            with st.expander("Sources Found"):
                st.markdown("- " + "\n- ".join(sources_list))

if prompt := st.chat_input("Ask a question about your documents...", disabled=rag_agent.get_document_count() == 0):
    st.chat_message("user").markdown(prompt)
    save_message(USER_ID, "user", prompt)
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating response..."):
            response = rag_agent.query(prompt, st.session_state.messages)
            st.markdown(response["answer"])
            if response["sources"]:
                unique_sources = []
                seen = set()
                for src in response["sources"]:
                    key = (src['source'], src['page'])
                    if key not in seen:
                        seen.add(key)
                        unique_sources.append(src)
                sources_list = []
                for src in unique_sources:
                    sources_list.append(f"*{src['source']}* (p. {src['page']})")
                with st.expander("Sources Found"):
                    st.markdown("- " + "\n- ".join(sources_list))
                save_message(USER_ID, "assistant", response["answer"], is_rag=True, sources=unique_sources)
            else:
                save_message(USER_ID, "assistant", response["answer"], is_rag=False, sources=[])

if st.session_state.messages:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.button("Clear Chat History", on_click=handle_clear_chat)