import os
import streamlit as st
import faiss
import numpy as np
import pdfplumber
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# New LangChain imports for handling unstructured text/docs
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader

# 1. Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Please check your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# 2. Unified File Processing Function
def process_file(file):
    file.seek(0)
    file_extension = file.name.split(".")[-1].lower()
    
    # We will write temporary files because LangChain loaders expect file paths, 
    # while Streamlit provides an in-memory byte stream.
    temp_filename = f"temp_upload.{file_extension}"
    with open(temp_filename, "wb") as f:
        f.write(file.read())
        
    documents = []
    
    try:
        # Dynamic routing based on extension
        if file_extension == "pdf":
            # Keeping our highly reliable pdfplumber approach
            text = ""
            with pdfplumber.open(temp_filename) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text() 
                    if extracted:
                        text += extracted + "\n"
            documents = [Document(page_content=text)]
            
        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_filename)
            documents = loader.load()
            
        elif file_extension == "txt":
            loader = TextLoader(temp_filename)
            documents = loader.load()
            
        elif file_extension == "csv":
            loader = CSVLoader(temp_filename)
            documents = loader.load()
            
        else:
            st.error("Unsupported file format.")
            return None, None
            
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None, None
    finally:
        # Clean up the temporary file from disk immediately
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    # Extract clean text from LangChain documents
    combined_text = "\n\n".join([doc.page_content for doc in documents])

    if not combined_text.strip():
        return None, None

    # Chunking & Embedding
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(combined_text)

    if not chunks:
        return None, None

    model = load_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    
    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(np.array(embeddings).astype('float32'))
    
    return chunks, index

# -------------------- UI LAYOUT --------------------
st.title("My RAG Assistant 🤖")

# Change the file uploader to accept multiple formats
uploaded_file = st.file_uploader("📂 Choose a file", type=["pdf", "docx", "txt", "csv"])

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing file for the first time... 📄"):
            # Call our newly unified multi-format function
            chunks, index = process_file(uploaded_file)
            
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.current_file = uploaded_file.name

    # 2. Retrieve the processed data from memory instantly
    chunks = st.session_state.chunks
    index = st.session_state.index

    # If extraction failed, stop execution
    if chunks is None or index is None:
        st.error("❌ Could not extract any readable text from this PDF. Please try a different text-based PDF.")
    else:
        st.success("PDF loaded from memory! ⚡")
        st.divider()

        query = st.text_input("❓ Type your question here...")

        if query:
            with st.spinner("Thinking... 🤔"):
                # Your existing search and Groq query code goes here exactly as it was...
                # Encode and format the query for FAISS
                model = load_embedding_model()
                query_embedding = model.encode([query], normalize_embeddings=True)
                query_vector = np.array(query_embedding).astype('float32')
                
                # Search the index
                k = 4
                distances, indices = index.search(query_vector, k)

                retrieved_chunks = [chunks[i] for i in indices[0]]
                context = "\n\n".join(retrieved_chunks)

                # Query Groq
                try:
                    response = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a senior technical analyst. Your job is to read the provided context "
                                    "and synthesize the information to answer the user's question completely in your own words. "
                                    "CRITICAL RULES: "
                                    "1. ABSOLUTELY NO BULLET POINTS OR LISTS. You must write in paragraph form. "
                                    "2. DO NOT quote or copy text directly from the context. "
                                    "3. Explain the concepts naturally in 3 to 4 cohesive sentences. "
                                    "If you merely repeat the source text, you have failed your instructions."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
                            }
                        ],
                        temperature=0.6,
                        max_tokens=300
                    )
                    answer = response.choices[0].message.content
                except Exception as e:
                    answer = f"⚠️ Error with the LLM API: {e}"

            # Output the answer
            st.subheader("💡 Answer")
            st.write(answer)

            with st.expander("🔍 View Retrieved Context"):
                st.write(context)