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
from rank_bm25 import BM25Okapi

# New LangChain imports for handling unstructured text/docs
from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader

import subprocess
import sys
try:
    import docx2txt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "docx2txt"])

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

# STEP 5A: Load the Cross-Encoder model cache resource
@st.cache_resource
def load_reranker_model():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# 2. Unified File Processing Function
def process_file(file):
    file.seek(0)
    file_extension = file.name.split(".")[-1].lower()
    
    temp_filename = f"temp_upload.{file.name.split('.')[-1].lower()}"
    with open(temp_filename, "wb") as f:
        f.write(file.read())
    documents = []
    try:
        if file_extension == "pdf":
            text = ""
            with pdfplumber.open(temp_filename) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text() 
                    if extracted:
                        text += extracted + "\n"
            documents = [Document(page_content=text)]
        elif file_extension == "docx":
            documents = Docx2txtLoader(temp_filename).load()
        elif file_extension == "txt":
            documents = TextLoader(temp_filename).load()
        elif file_extension == "csv":
            documents = CSVLoader(temp_filename).load()
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return None, None, None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    combined_text = "\n\n".join([doc.page_content for doc in documents])
    if not combined_text.strip():
        return None, None, None

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(combined_text)
    if not chunks:
        return None, None, None

    # ---- 1. BUILD DENSE INDEX (FAISS) ----
    model = load_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    dense_index = faiss.IndexFlatIP(len(embeddings[0]))
    dense_index.add(np.array(embeddings).astype('float32'))
    
    # ---- 2. BUILD SPARSE INDEX (BM25) ----
    tokenized_chunks = [chunk.lower().split(" ") for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    
    return chunks, dense_index, bm25_index

# STEP 5B: Upgraded Hybrid Search incorporating Cross-Encoder Scoring & Deduplication
def hybrid_search(search_query, chunks, dense_index, bm25_index, top_n=4):
    # 1. Dense Vector Search (FAISS)
    model = load_embedding_model()
    query_embedding = model.encode([search_query], normalize_embeddings=True)
    query_vector = np.array(query_embedding).astype('float32')
    
    # Grab a wider net (top 10 chunks) to leave options for the reranker layer
    initial_k = min(len(chunks), 10)
    _, dense_indices = dense_index.search(query_vector, initial_k)
    dense_ranked_list = dense_indices[0].tolist()
    
    # 2. Sparse Keyword Search (BM25)
    tokenized_query = search_query.lower().split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_list = np.argsort(bm25_scores)[::-1][:initial_k].tolist()
    
    # 3. Reciprocal Rank Fusion (RRF) Execution
    rrf_scores = {}
    k_constant = 60
    
    for rank, chunk_idx in enumerate(dense_ranked_list):
        if chunk_idx not in rrf_scores:
            rrf_scores[chunk_idx] = 0.0
        rrf_scores[chunk_idx] += 1.0 / (k_constant + (rank + 1))
        
    for rank, chunk_idx in enumerate(bm25_ranked_list):
        if chunk_idx not in rrf_scores:
            rrf_scores[chunk_idx] = 0.0
        rrf_scores[chunk_idx] += 1.0 / (k_constant + (rank + 1))
        
    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    candidate_indices = [item[0] for item in sorted_chunks[:10]]
    candidate_chunks = [chunks[idx] for idx in candidate_indices]
    
    if not candidate_chunks:
        return []

    # ---- 4. CROSS-ENCODER RERANKING ROUTINE ----
    reranker = load_reranker_model()
    pair_inputs = [[search_query, chunk] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(pair_inputs)
    
    reranked_results = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
    
    # ---- 5. CONTEXT DEDUPLICATION ----
    final_chunks = []
    seen_texts = set()
    
    for chunk, score in reranked_results:
        simplified_text = "".join(chunk.lower().split())
        if simplified_text not in seen_texts:
            seen_texts.add(simplified_text)
            final_chunks.append(chunk)
            
        if len(final_chunks) == top_n:
            break
            
    return final_chunks

# -------------------- UI LAYOUT --------------------
st.title("My RAG Assistant 🤖")

uploaded_file = st.file_uploader("📂 Choose a file", type=["pdf", "docx", "txt", "csv"])

if uploaded_file:
    if "current_file" not in st.session_state or st.session_state.current_file != uploaded_file.name:
        with st.spinner("Processing file for the first time... 📄"):
            chunks, dense_index, bm25_index = process_file(uploaded_file)
            
            st.session_state.chunks = chunks
            st.session_state.dense_index = dense_index
            st.session_state.bm25_index = bm25_index
            st.session_state.current_file = uploaded_file.name

    chunks = st.session_state.chunks
    dense_index = st.session_state.dense_index
    bm25_index = st.session_state.bm25_index

    if chunks is None or dense_index is None or bm25_index is None:
        st.error("❌ Could not extract any readable text from this file. Please try a different text-based document.")
    else:
        st.success("File loaded from memory! ⚡")
        st.divider()

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if query := st.chat_input("❓ Ask something about your document..."):
            with st.chat_message("user"):
                st.write(query)
            
            st.session_state.messages.append({"role": "user", "content": query})

            with st.spinner("Thinking... 🤔"):
                search_query = query
                if len(st.session_state.messages) > 1:
                    history_str = ""
                    for msg in st.session_state.messages[:-1]:
                        history_str += f"{msg['role'].capitalize()}: {msg['content']}\n"
                    
                    try:
                        rewrite_response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a query-rewriter. Given a chat history and a follow-up question, "
                                        "rephrase the follow-up question into a standalone question that can be understood "
                                        "WITHOUT the chat history. Do NOT answer the question. Just output the rephrased question."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": f"Chat History:\n{history_str}\nFollow-up Question: {query}\n\nStandalone Question:"
                                }
                            ],
                            temperature=0.0
                        )
                        search_query = rewrite_response.choices[0].message.content.strip()
                    except Exception as e:
                        search_query = query

                # EXECUTE UPGRADED HYBRID SEARCH (Featuring Step 5 Reranking)
                retrieved_chunks = hybrid_search(
                    search_query=search_query, 
                    chunks=chunks, 
                    dense_index=dense_index, 
                    bm25_index=bm25_index, 
                    top_n=4
                )
                context = "\n\n".join(retrieved_chunks)

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

            with st.chat_message("assistant"):
                st.write(answer)
                st.markdown("---")
                with st.expander("🔍 View Reranked Context Breakdown"):
                    st.caption(f"**FAISS & BM25 Integrated Search Query Used:** `{search_query}`")
                    st.markdown("**Top Chunks Utilized for Synthesis (Filtered via Cross-Encoder):**")
                    for idx, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"> **Source Block {idx+1}:** {chunk}")

            st.session_state.messages.append({"role": "assistant", "content": answer})