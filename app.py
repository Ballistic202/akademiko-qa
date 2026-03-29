from flask import Flask, request, jsonify
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
import os

app = Flask(__name__)

openai_client = AzureOpenAI(
    azure_endpoint=os.environ["OPENAI_ENDPOINT"],
    api_key=os.environ["OPENAI_KEY"],
    api_version="2024-02-01"
)

search_client = SearchClient(
    endpoint=os.environ["SEARCH_ENDPOINT"],
    index_name="akademiko-knowledge-source-index",
    credential=AzureKeyCredential(os.environ["SEARCH_KEY"])
)

@app.route("/")
def home():
    return "Akademiko Q&A работи!"

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question")
    embedding = openai_client.embeddings.create(
        input=question,
        model="text-embedding-3-small"
    ).data[0].embedding
    results = search_client.search(
        search_text=question,
        vector_queries=[VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=5,
            fields="snippet_vector"
        )],
        select=["snippet", "blob_url"],
        top=3
    )
    chunks = [r["snippet"] for r in results]
    context = "\n\n".join(chunks)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Отговаряй на български на база предоставеното учебно съдържание."},
            {"role": "user", "content": f"Контекст:\n{context}\n\nВъпрос: {question}"}
        ]
    )
    return jsonify({"answer": response.choices[0].message.content})

if __name__ == "__main__":
    app.run()
