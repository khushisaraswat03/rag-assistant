import os
import sys
import time
import requests
import streamlit as st
import faiss
import numpy as np
import pdfplumber
import tempfile
import logging

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

if "HF_TOKEN" in st.secrets:
    os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
elif os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY is missing. Please check your Secrets or .env file.")
    st.stop()
if not os.environ.get("HF_TOKEN"):
    st.error("❌ HF_TOKEN is missing. Please check your Secrets or .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------
# ☁️ CLOUD API WRAPPERS (Bulletproofed with Retry & Batching)
# ---------------------------------------------------------
class CloudEmbeddingModel:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

    def encode(self, texts, normalize_embeddings=True):
        logger.info(f"☁️ Embedding {len(texts)} chunks via Hugging Face API...")
        all_embeddings = []
        batch_size = 10 
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            success = False
            retries = 3
            
            while not success and retries > 0:
                try:
                    response = requests.post(
                        self.api_url, 
                        headers=self.headers, 
                        json={"inputs": batch, "options": {"wait_for_model": True}},
                        timeout=45
                    )
                    
                    if response.status_code == 200:
                        all_embeddings.extend(response.json())
                        success = True
                        time.sleep(1) # Rate limit protection
                    elif response.status_code in [503, 429]:
                        logger.warning(f"HF API busy (Status {response.status_code}). Retrying in 10 seconds...")
                        time.sleep(10)
                        retries -= 1
                    else:
                        logger.error(f"HF API Error: {response.text}")
                        st.error(f"Hugging Face API Error: {response.text}")
                        st.stop()
                        
                except requests.exceptions.RequestException as e:
                    logger.warning(f"Connection issue: {e}. Retrying...")
                    time.sleep(5)
                    retries -= 1
            
            if not success:
                st.error("❌ Hugging Face API failed to respond after multiple attempts. Please try again later.")
                st.stop()
                
        return all_embeddings

class CloudRerankerModel:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.headers = {"Authorization": f"Bearer {os.environ.get('HF_TOKEN')}"}

    def predict(self, pairs):
        logger.info(f"☁️ Reranking {len(pairs)} candidate pairs via Hugging Face API...")
        inputs = [{"text": query, "text_pair": doc} for query, doc in pairs]
        
        retries = 3
        while retries > 0:
            try:
                response = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json={"inputs": inputs, "options": {"wait_for_model": True}},
                    timeout=45
                )
                
                if response.status_code == 200:
                    results = response.json()
                    scores = []
                    for res in results:
                        if isinstance(res, list):
                            scores.append(res[0].get("score", 0.0))
                        elif isinstance(res, dict):
                            scores.append(res.get("score", 0.0))
                        else:
                            scores.append(res)
                    return scores
                elif response.status_code in [503, 429]:
                    logger.warning(f"HF Reranker busy (Status {response.status_code}). Retrying in 10 seconds...")
                    time.sleep(10)
                    retries -= 1
                else:
                    logger.error(f"HF Reranker Error: {response.text}")
                    return [0.0] * len(pairs)
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Connection issue: {e}. Retrying...")
                time.sleep(5)
                retries -= 1
                
        return [0.0] * len(pairs) # Fallback if reranking fails entirely

@st.cache_resource
def load_embedding_model():
    return CloudEmbeddingModel()

@st.cache_resource
def load_reranker_model():
    return CloudRerankerModel()
# ---------------------------------------------------------

# 2. Unified File Ingestion Pipeline
def process_file(file):
    file.seek(0)
    file_extension = file.name.split(".")[-1].lower()
    logger.info(f"🚀 Starting ingestion pipeline for: '{file.name}'")
    
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
        return None, None, None
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

    combined_text = "\n\n".join([doc.page_content for doc in documents])
    if not combined_text.strip():
        return None, None, None

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_text(combined_text)
    logger.info(f"🧩 Split data into {len(chunks)} fragments.")
    if not chunks:
        return None, None, None

    # ---- BUILD INDEXES ----
    model = load_embedding_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    
    dense_index = faiss.IndexFlatIP(len(embeddings[0]))
    dense_index.add(np.array(embeddings).astype('float32'))
    logger.info("✅ FAISS Dense Index established.")
    
    tokenized_chunks = [chunk.lower().split(" ") for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)
    logger.info("✅ BM25 Sparse Index established.")
    
    return chunks, dense_index, bm25_index

# 3. Two-Stage Retrieval Logic
def hybrid_search(search_query, chunks, dense_index, bm25_index, top_n=4):
    model = load_embedding_model()
    query_embedding = model.encode([search_query], normalize_embeddings=True)
    query_vector = np.array(query_embedding).astype('float32')
    
    initial_k = min(len(chunks), 10)
    _, dense_indices = dense_index.search(query_vector, initial_k)
    dense_ranked_list = dense_indices[0].tolist()
    
    tokenized_query = search_query.lower().split(" ")
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_ranked_list = np.argsort(bm25_scores)[::-1][:initial_k].tolist()
    
    rrf_scores = {}
    k_constant = 60
    for rank, chunk_idx in enumerate(dense_ranked_list):
        if chunk_idx not in rrf_scores: rrf_scores[chunk_idx] = 0.0
        rrf_scores[chunk_idx] += 1.0 / (k_constant + (rank + 1))
    for rank, chunk_idx in enumerate(bm25_ranked_list):
        if chunk_idx not in rrf_scores: rrf_scores[chunk_idx] = 0.0
        rrf_scores[chunk_idx] += 1.0 / (k_constant + (rank + 1))
        
    sorted_chunks = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    candidate_indices = [item[0] for item in sorted_chunks[:10]]
    candidate_chunks = [chunks[idx] for idx in candidate_indices]

    if not candidate_chunks:
        return []

    reranker = load_reranker_model()
    pair_inputs = [[search_query, chunk] for chunk in candidate_chunks]
    rerank_scores = reranker.predict(pair_inputs)
    
    reranked_results = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)
    
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
        with st.spinner("Processing file via Hugging Face Cloud APIs... ☁️📄"):
            chunks, dense_index, bm25_index = process_file(uploaded_file)
            
            st.session_state.chunks = chunks
            st.session_state.dense_index = dense_index
            st.session_state.bm25_index = bm25_index
            st.session_state.current_file = uploaded_file.name

    chunks = st.session_state.chunks
    dense_index = st.session_state.dense_index
    bm25_index = st.session_state.bm25_index

    if chunks is None or dense_index is None or bm25_index is None:
        st.error("❌ Could not extract text. Please try a different document.")
    else:
        st.success("File processed securely and loaded! ⚡")
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
                with st.expander("🔍 View Reranked Context Breakdown"):
                    st.caption(f"**Search Query Used:** `{search_query}`")
                    for idx, chunk in enumerate(retrieved_chunks):
                        st.markdown(f"> **Source Block {idx+1}:** {chunk}")

            st.session_state.messages.append({"role": "assistant", "content": answer})