# 🤖 Grounded-AI

A **RAG (Retrieval-Augmented Generation)** powered document assistant built with Streamlit and Groq. Upload your `.txt` documents and ask questions — Grounded-AI finds the most relevant context and answers using a blazing-fast LLM.

---

## ✨ Features

- 📄 **Document Ingestion** — Automatically reads all `.txt` files from the `data/` folder
- 🔍 **Semantic Search** — Uses `sentence-transformers` + `FAISS` for fast vector similarity search
- 🧠 **LLM Generation** — Powered by **Groq's** `llama-3.1-8b-instant` model for near-instant responses
- 💬 **Chat Memory** — Maintains conversation history within the session
- 🌙 **Dark Mode UI** — Sleek dark-themed Streamlit interface

---

## 🏗️ Project Structure

```
Grounded-AI/
├── app.py              # Main Streamlit app
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── data/               # Place your .txt documents here
└── rag/
    ├── retriever.py    # Chunking, embedding & FAISS retrieval
    ├── generator.py    # Groq LLM response generation
    ├── embedder.py     # Sentence-transformer embedder
    └── vector_store.py # Vector store utilities
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Ajay-Rajput/Grounded-AI.git
cd Grounded-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Add Your Documents

Place your `.txt` files inside the `data/` folder. These will be automatically indexed when the app starts.

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub (make sure `.env` is in `.gitignore` ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Deploy — your app will be live in minutes!

---

## ⚙️ How It Works

```
User Query
    │
    ▼
Sentence Transformer (Embedding)
    │
    ▼
FAISS Vector Search → Top-K Relevant Chunks
    │
    ▼
Groq LLM (llama-3.1-8b-instant) + Context
    │
    ▼
Answer
```

1. On startup, all `.txt` files in `data/` are split into overlapping chunks
2. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in a FAISS index
3. When a user asks a question, the top-3 most similar chunks are retrieved
4. The chunks are passed as context to Groq's LLM, which generates the final answer

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| [Streamlit](https://streamlit.io) | Web UI framework |
| [Groq](https://groq.com) | Ultra-fast LLM inference |
| [FAISS](https://github.com/facebookresearch/faiss) | Vector similarity search |
| [sentence-transformers](https://www.sbert.net) | Text embeddings |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | Environment variable management |

---

## 📄 License

MIT License — feel free to use, modify, and distribute.
