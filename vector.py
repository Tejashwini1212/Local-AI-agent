from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd 

df = pd.read_csv("realistic_restaurant_reviews.csv")  
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents= not os.path.exists(db_location)  
documents=[]
ids =[]

if add_documents:
    for i, row in df.iterrows():
        doc = Document(
            page_content=f"{row.get('Title', '')}\n{row.get('Review', '')}",
            metadata={
                "rating": row.get("Rating", None),
                "date": row.get("Date", None),
                # Reviewer is OPTIONAL now
                "reviewer": row.get("Reviewer", "Unknown"),
            },
        )
        documents.append(doc)
        ids.append(str(i))
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)
if add_documents and documents:
    vector_store.add_documents(documents=documents, ids=ids)

# ✅ EXPORT retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)