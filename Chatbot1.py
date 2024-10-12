import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Create a text file with Scope Factory information
scope_factory_info = """
# Scope Factory Information

## Company Overview
- Name: Scope Factory (مصنع سكوب)
- Type: Manufacturing factory
- Products: Glass, aluminum, and casting products
- Location: Medina, Saudi Arabia

## Contact Information
- Address: P.O. Box 25344, Medina, Saudi Arabia
- Postal Code: 41466
- Phone: 014-8225566
- Fax: 014-8225577
- Mobile: 0591004444
- Website: www.scope-factory.sa

## Key Features
- Registered supplier for the Neom project

## Product Categories
1. Glass products
2. Aluminum products
3. Casting products

## Additional Information
- Specializes in manufacturing for construction and industrial applications
- Serves clients in Saudi Arabia and potentially beyond
"""

with open('scope_factory_info.txt', 'w', encoding='utf-8') as f:
    f.write(scope_factory_info)

# Load and split the document
with open('scope_factory_info.txt', 'r', encoding='utf-8') as file:
    scope_factory_text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
scope_factory_docs = text_splitter.create_documents([scope_factory_text])

# Initialize the embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist the vector database
db = Chroma.from_documents(scope_factory_docs, embedding_function, persist_directory="./scope_factory_db")
db.persist()

# Initialize the language model
groq_api_key = "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"  
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

# Create the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
)

def query_scope_factory(query: str):
    result = qa_chain({"query": query})
    return result["result"]

def run_chatbot():
    print("Welcome to the Scope Factory Chatbot!")
    print("Ask any question about Scope Factory, or type 'exit' to quit.")

    while True:
        user_input = input("Your question: ")
        if user_input.lower() == 'exit':
            print("Thank you for using Scope Factory Chatbot. Goodbye!")
            break

        response = query_scope_factory(user_input)
        print(f"\nAnswer: {response}\n")

if __name__ == "__main__":
    run_chatbot()