from groq import Groq


class Generator:
    def __init__(self, api_key):
        self.client = Groq(api_key=api_key)

    def generate(self, query, context):
        prompt = f"""
You are a helpful AI tutor.

Context:
{context}

Question:
{query}

Answer clearly and clearly based only on the provided context.
"""

        response = self.client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        return response.choices[0].message.content
