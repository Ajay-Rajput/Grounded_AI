import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

from rag.retriever import Retriever
from rag.generator import Generator

 # -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="GroundedAI",
    page_icon="🤖",
    layout="wide"
)

# -------------------- DARK MODE CSS --------------------
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .stChatMessage {
            border-radius: 12px;
            padding: 10px;
        }
     </style>
 """, unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center;'>🤖 Grounded-AI</h1>
    <p style='text-align: center; font-size:18px; color: #9CA3AF;'>
        RAG Powered Document Assistant
    </p>
""", unsafe_allow_html=True)


# -------------------- LOAD API KEY --------------------
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not set.")
    st.stop()

# -------------------- LOAD SYSTEM --------------------
@st.cache_resource
def load_system():
    documents = []
    for file in os.listdir("data"):
        if file.endswith(".txt"):
            with open(os.path.join("data", file), "r", encoding="utf-8") as f:
                documents.append(f.read())

    retriever = Retriever()
    retriever.build_index(documents)

    generator = Generator(api_key=api_key)

    return retriever, generator

retriever, generator = load_system()

# -------------------- CHAT MEMORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -------------------- USER INPUT --------------------
if prompt := st.chat_input("Ask something about your documents..."):

    # Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    results = retriever.retrieve(prompt, top_k=3)

    context = ""
    for item in results:
        context += item["chunk"] + "\n\n"

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generator.generate(prompt, context)
            st.markdown(response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})
