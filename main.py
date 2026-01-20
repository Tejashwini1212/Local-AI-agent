from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # this should now work with your fixed vector.py

# Initialize the model
model = OllamaLLM(model="llama3.2", temperature=0.7)

# Prompt template
template = """
You are an expert in answering questions about pizza restaurants.
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Chat loop
while True:
    print("\n-------------------------")
    question = input("Please enter your question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break

    # Retrieve relevant reviews
    retrieved_docs = retriever.get_relevant_documents(question)
    reviews_text = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate answer
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print("\nAnswer:\n", result)
