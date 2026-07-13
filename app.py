import os
import faiss
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq

# Ensure your key is accessible. (Since we aren't in Streamlit, 
# this reads it directly from your WSL2 environment variable)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    # Fallback to manual insertion if environment variable isn't loaded yet
    GROQ_API_KEY = "your_actual_groq_api_key_here" 

client = Groq(api_key=GROQ_API_KEY)

# Step 1: Load PDF
pdf_path = "sample.pdf"
reader = PdfReader(pdf_path)

# Step 2: Extract text
text = ""
for page in reader.pages:
    extracted = page.extract_text()
    if extracted:  
        text += extracted + "\n"

print("\nTotal characters extracted:", len(text))

# Step 3: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,      # Adjusted to match your roadmap choice
    chunk_overlap=50     
)
chunks = text_splitter.split_text(text)
print("Total chunks:", len(chunks))

# Step 4: Load embedding model & vector spaces (Enforcing Cosine Similarity)
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks, normalize_embeddings=True) # Normalized doc embeddings

embedding_array = np.array(embeddings)
dimension = embedding_array.shape[1]

# Swapped from FlatL2 to IndexFlatIP for Cosine Inner Product
index = faiss.IndexFlatIP(dimension) 
index.add(embedding_array)

# Step 5: Query setup
query = "What is LLM?"
query_embedding = model.encode([query], normalize_embeddings=True) # Normalized query embedding

# Search top 4 similar chunks
k = 4
distances, indices = index.search(np.array(query_embedding), k)

# Get retrieved chunks and combine into context
retrieved_chunks = [chunks[i] for i in indices[0]]
context = "\n\n".join(retrieved_chunks)

# Step 6: Generate structured response via Groq API (Chat Message Format)
print("\n----- THINKING VIA GROQ API... 🤔 -----")
try:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise, technical AI assistant. Your goal is to explain "
                    "answers clearly in simple terms using 3-4 sentences. "
                    "Synthesize the ideas smoothly into a cohesive explanation based "
                    "strictly on the provided context—do not just repeat phrases or copy fragments."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0.3,
        max_tokens=300
    )
    answer = response.choices[0].message.content
except Exception as e:
    answer = f"⚠️ An error occurred with the LLM API: {e}"

print("\n----- FINAL ANSWER -----")
print(answer)

print("\n----- VIEW RETRIEVED CONTEXT -----")
print(context)