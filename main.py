import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings  # Import for embeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize the Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Set up the Streamlit app
st.title("News Retrival 📈")
st.sidebar.title("News Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
main_placeholder = st.empty()

# Set up the embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load data when URLs are processed
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()
    
    # Split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)
    
    # Create embeddings and save to FAISS index
    vectorstore_openai = FAISS.from_documents(docs, embedding_model)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)
    
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

# Query input from the user
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            
            # Retrieve relevant documents using FAISS
            retriever = vectorstore.as_retriever()
            relevant_docs = retriever.get_relevant_documents(query)
            
            # Combine relevant document content
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Prepare the message with the context and the query
            message_for_groq = f"Context:\n{context}\n\nQuestion: {query}"
            
            # Send the message to the Groq client
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": message_for_groq}],
                    model="llama3-8b-8192",  # Ensure this model is valid
                )
                answer_from_groq = chat_completion.choices[0].message.content
            except Exception as e:
                answer_from_groq = f"Error occurred while calling Groq: {e}"
            
            # Display the answer from Groq
            st.header("Answer from Groq")
            st.write(answer_from_groq)
            
            # Display the retrieved sources
        