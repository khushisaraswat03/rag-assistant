import streamlit as st
from pypdf import PdfReader
from langchain_text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="RAG Assistant", layout="centered")

st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Main container transparent */
[data-testid="stAppViewContainer"] {
    background: transparent;
}

/* Remove header white */
[data-testid="stHeader"] {
    background: transparent;
}

/* Card style (UPLOAD + sections) */
.block-container {
    background: rgba(30, 41, 59, 0.6);
    padding: 2rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
}

/* Input box */
.stTextInput input {
    background-color: black;
    color: white;
    border-radius: 10px;
    border: none;
}

/* File uploader */
.stFileUploader {
    background-color: #1e293b;
    padding: 15px;
    border-radius: 12px;
}
[data-testid="stFileUploader"] button {
    background-color: #0f172a !important;
    color: white !important;
    border: 1px solid #64748b;
    border-radius: 8px;
    font-weight: 600;
}

[data-testid="stFileUploader"] button span {
    color: white !important;
}
/* Answer box (lighter than background) */
.answer-box {
    background: black;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #475569;
    font-size: 16px;
}

/* Headings */
h1, h2, h3, p {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# -------------------- UI --------------------
st.markdown("<h1>🤖 Personal Knowledge Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Ask questions from your PDF instantly</p>", unsafe_allow_html=True)

st.divider()

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embed_model, tokenizer, model

# -------------------- PROCESS PDF --------------------
@st.cache_data
def process_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return chunks, index

# -------------------- FILE UPLOAD --------------------
st.markdown("### 📂 Upload your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    st.success("PDF uploaded successfully!")

    # Load models only after upload
    with st.spinner("Loading models... ⏳"):
        embed_model, tokenizer, model_llm = load_models()

    # Process PDF (cached)
    with st.spinner("Processing PDF... 📄"):
        chunks, index = process_pdf(uploaded_file)

    st.divider()

    # -------------------- QUERY --------------------
    st.subheader("❓ Ask a Question")
    query = st.text_input("Type your question here...")

    if query:
        with st.spinner("Thinking... 🤔"):

            # Query embedding
            query_embedding = embed_model.encode([query])

            # Search
            k = 4
            distances, indices = index.search(np.array(query_embedding), k)

            # Context
            retrieved_chunks = [chunks[i] for i in indices[0]]
            context = "\n".join(retrieved_chunks)

            # Prompt
            prompt = f"""
You are an AI assistant.

Explain the answer clearly in simple terms using 3-4 sentences.
Do not just repeat phrases. Combine ideas into a proper explanation.

Context:
{context}

Question:
{query}

Answer:
"""

            # Generate answer
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = model_llm.generate(**inputs, max_new_tokens=200)

            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.divider()

        # -------------------- OUTPUT --------------------
        st.subheader("💡 Answer")
        st.markdown(f"""
        <div class="answer-box">
        {answer}
        </div>
        """, unsafe_allow_html=True)

        # Show context
        with st.expander("🔍 View Retrieved Context"):
            st.write(context)