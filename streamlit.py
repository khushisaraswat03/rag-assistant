import os
import sys
import streamlit as st
import faiss
import numpy as np
import pdfplumber
import tempfile
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from langchain_core.documents import Document
from langchain_community.document_loaders import Docx2txtLoader, CSVLoader, TextLoader

# 1. Pipeline Environment Overrides & Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Please check your Secrets or .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# 2. Unified File Ingestion Pipeline (100% Local Math)
def process_file(file):
    file.seek(0)
    file_extension = file.name.split(".")[-1].lower()
    logger.info(f"🚀 Starting local ingestion pipeline for: '{file.name}'")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(file.read())
        temp_filename = temp_file.name
        
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
        logger.error(f"❌ Extraction error: {e}", exc_info=True)
        st.error(f"⚠️ Extraction Error: {e}")
        return None, None, None, None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    combined_text = "\n\n".join([doc.page_content for doc in documents])
    if not combined_text.strip():
        return None, None, None, None

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(combined_text)
    logger.info(f"🧩 Split data into {len(chunks)} fragments.")
    if not chunks:
        return None, None, None, None

    # ---- BUILD LOCAL DENSE INDEX (TF-IDF Vector Space) ----
    vectorizer = TfidfVectorizer(max_features=384)  # Match standard embedding dimensions
    tfidf_matrix = vectorizer.fit_transform(chunks).toarray().astype('float32')
    
    dense_index = faiss.IndexFlatIP(tfidf_matrix.shape[1])
    dense_index.add(tfidf_matrix)
    logger.info("✅ Local FAISS Dense Index established via TF-IDF Vectorizer.")
    
    # ---- BUILD SPARSE KEYWORD INDEX ----
    tokenized_chunks = [chunk.lower().split(" ") for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    logger.info("✅ BM25 Sparse Index established.")
    
    return chunks, dense_index, bm25_index, vectorizer

# 3. Hybrid Retrieval Logic via Reciprocal Rank Fusion
def hybrid_search(search_query, chunks, dense_index, bm25_index, vectorizer, top_n=4):
    logger.info(f"🔍 Executing Hybrid Retrieval Pass for Query: '{search_query}'")
    
    # 1. Local Dense Search Pass
    query_vector = vectorizer.transform([search_query]).toarray().astype('float32')
    initial_k = min(len(chunks), 10)
    _, dense_indices = dense_index.search(query_vector, initial_k)
    dense_ranked_list = dense_indices[0].tolist()
    
    # 2. Sparse Search Pass
    tokenized_query = search_query.lower().split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_list = np.argsort(bm25_scores)[::-1][:initial_k].tolist()
    
    # 3. Reciprocal Rank Fusion (RRF) Sorting Tiers
    rrf_scores = {}
    k_constant = 60
    for rank, chunk_idx in enumerate(dense_ranked_list):
        if chunk_idx not in rrf_scores: rrf_scores[chunk_idx] = 0.0
        rrf_scores[chunk_idx] += 1.0 / (k_constant + (rank + 1))
    for rank, chunk_idx in enumerate(bm25_ranked_list):
        if chunk_idx not in rrf_scores: rrf_scores[chunk_idx] = 0.0
        rrf_scores[chunk_idx] += 1.0 / (k_constant + (rank + 1))
        
    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Deduplication and Output Extraction
    final_chunks = []
    seen_texts = set()
    for chunk_idx, score in sorted_chunks:
        chunk = chunks[chunk_idx]
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
        with st.spinner("Processing file securely on local engine... 📄⚡"):
            chunks, dense_index, bm25_index, vectorizer = process_file(uploaded_file)
            
            st.session_state.chunks = chunks
            st.session_state.dense_index = dense_index
            st.session_state.bm25_index = bm25_index
            st.session_state.vectorizer = vectorizer
            st.session_state.current_file = uploaded_file.name

    chunks = st.session_state.chunks
    dense_index = st.session_state.dense_index
    bm25_index = st.session_state.bm25_index
    vectorizer = st.session_state.vectorizer

    if chunks is None or dense_index is None or bm25_index is None:
        st.error("❌ Could not extract text. Please try a different document.")
    else:
        st.success("File processed securely and loaded instantly! ⚡")
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
                                        "WITHOUT the chat history. Do NOT answer the question."
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
                    except Exception:
                        search_query = query

                retrieved_chunks = hybrid_search(
                    search_query=search_query, 
                    chunks=chunks, 
                    dense_index=dense_index, 
                    bm25_index=bm25_index, 
                    vectorizer=vectorizer,
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
                                    "3. Explain the concepts naturally in 3 to 4 cohesive sentences."
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
                with st.expander("🔍 View Hybrid Context Breakdown"):
                    st.caption(f"**Search Query Used:** `{search_query}`")
                    for idx, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"> **Source Block {idx+1}:** {chunk}")

            st.session_state.messages.append({"role": "assistant", "content": answer})