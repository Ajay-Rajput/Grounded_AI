import streamlit as st
import os
from dotenv import load_dotenv
from pypdf import PdfReader
from rag.retriever import Retriever
from rag.generator import Generator

# ---------------- LOAD ENV ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not set.")
    st.stop()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DocuMind AI", page_icon="📄", layout="wide")

# ---------------- TITLE CENTER ----------------
st.markdown("""
    <div style='text-align: center; padding-bottom: 20px;'>
        <h1>📄 DocuMind AI</h1>
        <p style='color:gray;'>RAG Powered Document Assistant</p>
    </div>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- PDF UPLOADER ----------------
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:

    reader = PdfReader(uploaded_file)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    st.success("PDF Loaded Successfully!")

    retriever = Retriever()
    retriever.build_index([text])

    st.session_state.retriever = retriever

    st.info("Document indexed. You can now ask questions.")

# ---------------- CHAT DISPLAY ----------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------- USER INPUT ----------------
if prompt := st.chat_input("Ask something about the uploaded document..."):

    if not st.session_state.retriever:
        st.warning("Please upload a PDF first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Retrieve context
    results = st.session_state.retriever.retrieve(prompt, top_k=3)

    context = ""
    for item in results:
        context += item["chunk"] + "\n\n"

    # Generate answer
    generator = Generator(api_key=api_key)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generator.generate(prompt, context)
            st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
