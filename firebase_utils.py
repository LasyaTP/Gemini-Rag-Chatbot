import streamlit as st
import random
import string
from typing import List, Dict, Optional

def initialize_firebase_auth() -> str:
    """
    Simulates Firebase user authentication.
    In a real app, this would handle signInWithCustomToken or check session.
    """
    if "user_id" not in st.session_state:
        st.session_state.user_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        print(f"Simulated Auth: New User ID created: {st.session_state.user_id}")
    return st.session_state.user_id

def load_chat_history(user_id: str) -> List[Dict]:
    """
    Simulates loading chat history from Firestore for a specific user.
    Uses st.session_state as a fallback for the demo.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []
    return st.session_state.messages

def save_message(user_id: str, role: str, content: str, is_rag: bool = False, sources: Optional[List] = None):
    """
    Simulates saving a message to Firestore.
    """
    new_message = {
        "role": role,
        "content": content,
        "is_rag": is_rag,
        "sources": sources,
        "timestamp": random.random() # Placeholder for sort key
    }

    if "messages" in st.session_state:
        st.session_state.messages.append(new_message)
    else:
        st.session_state.messages = [new_message]

def clear_chat_history():
    """Simulates clearing the chat history in Firestore."""
    st.session_state.messages = []

def get_user_id() -> str:
    """Helper to get the current user ID."""
    return st.session_state.get('user_id', 'anonymous')