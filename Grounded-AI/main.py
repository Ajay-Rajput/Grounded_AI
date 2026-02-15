from rag.retriever import Retriever
from rag.generator import Generator
import os

# 🔹 1. Load API Key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    print("Error: OPENAI_API_KEY not set")
    exit()

generator = Generator(api_key=api_key)

# 🔹 2. Load Multiple Documents from data/ folder
documents = []

for file in os.listdir("data"):
    if file.endswith(".txt"):
        with open(os.path.join("data", file), "r", encoding="utf-8") as f:
            documents.append(f.read())

print(f"Loaded {len(documents)} documents")

# 🔹 3. Build Retriever Index
retriever = Retriever()
retriever.build_index(documents)

# 🔹 4. Ask Question
query = input("Ask your question: ")

retrieved = retriever.retrieve(query, top_k=3)

context = ""
for i, item in enumerate(retrieved):
    print(f"\nChunk {i+1} | Score: {item['score']:.4f}")
    print(item["chunk"])
    print("-" * 50)

    context += f"[Chunk {i+1}]\n{item['chunk']}\n\n"

# 🔹 5. Generate Answer
response = generator.generate(query, context)

print("\nFinal Answer:\n")
print(response)
