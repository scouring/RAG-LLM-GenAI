# Wine Recommendation with Embeddings and LLMs 🍷🤖

This project demonstrates a **modern AI-powered wine recommendation system** using **vector embeddings**, **Qdrant vector database**, and a **large language model (LLM)** to deliver personalized suggestions based on user queries.  

It’s a practical showcase of skills in **Python**, **NLP embeddings**, **semantic search**, and **LLM integration** — perfect for AI/ML portfolios.

---

## 🚀 Features

- **Vector Embeddings**: Uses [Sentence-Transformers](https://www.sbert.net/) `all-MiniLM-L6-v2` to encode wine tasting notes into numerical vectors.  
- **Vector Database**: Leverages [Qdrant](https://qdrant.tech/) as an in-memory vector database to store and search embeddings efficiently.  
- **Semantic Search**: Finds wines most relevant to a user query, e.g., "Suggest me an amazing Malbec from Argentina."  
- **LLM Integration**: Connects search results to an OpenAI GPT-4 Turbo model to generate natural language recommendations.  
- **Data Handling**: Cleans and samples wine dataset to ensure smooth embedding and indexing.

---

## 📂 Dataset

- Uses a **CSV of top-rated wines**.
- Filters out entries with missing varieties (`NaN`) to avoid errors in embeddings.
- Samples **700 records** for efficient indexing and search.

---

## 🛠 Tech Stack

- **Python 3.9+**
- [SentenceTransformers](https://www.sbert.net/) for embeddings
- [Qdrant](https://qdrant.tech/) vector database
- [OpenAI GPT-4 Turbo](https://platform.openai.com/docs/models/gpt-4)
- [Pandas](https://pandas.pydata.org/) for data processing
- Google Colab environment for quick prototyping

---

## ⚡ How It Works

1. Load and clean the wine dataset.
2. Sample a subset for efficient processing.
3. Encode wine tasting notes into embeddings using `SentenceTransformer`.
4. Create an in-memory **Qdrant collection** to store vectors and metadata.
5. Perform **semantic search** to find wines matching the user prompt.
6. Pass the search results to an **LLM** to generate user-friendly recommendations.

---

## 📝 Example Usage

```python
user_prompt = "Suggest me an amazing Malbec wine from Argentina"

# Perform vector search
hits = qdrant.search(
    collection_name="top_wines",
    query_vector=encoder.encode(user_prompt).tolist(),
    limit=3
)

# Collect search results
search_results = [hit.payload for hit in hits]

# Generate natural language recommendation using GPT-4
completion = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[
        {"role": "system", "content": "You are a wine specialist chatbot."},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": str(search_results)}
    ]
)
print(completion.choices[0].message)

