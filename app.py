from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# Step 1: Load PDF
pdf_path = "sample.pdf"

reader = PdfReader(pdf_path)

# Step 2: Extract text
text = ""

for page in reader.pages:
    extracted = page.extract_text()
    if extracted:  # handle empty pages
        text += extracted

# Step 3: Print output
print("----- EXTRACTED TEXT -----\n")
print(text[:1000])  # first 1000 characters

# Extra: show size
print("\nTotal characters:", len(text))



# Step 4: Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # size of each chunk
    chunk_overlap=50     # overlap between chunks
)

chunks = text_splitter.split_text(text)

# Step 5: Print chunk info
print("\n----- CHUNKS -----\n")
print("Total chunks:", len(chunks))

# Print first 2 chunks
for i, chunk in enumerate(chunks[:2]):
    print(f"\nChunk {i+1}:\n{chunk[:200]}")



# Step 6: Load embedding model (FREE)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 7: Convert chunks into embeddings
embeddings = model.encode(chunks)

# Step 8: Print info
print("\n----- EMBEDDINGS -----\n")
print("Number of embeddings:", len(embeddings))
print("Shape of one embedding:", embeddings[0].shape)


embedding_array = np.array(embeddings)

dimension = embedding_array.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embedding_array)

# Step 9: Query
query = "What is Machine Learning?"

# Convert query to embedding
query_embedding = model.encode([query])

# Search top 2 similar chunks
k = 2
distances, indices = index.search(np.array(query_embedding), k)

print("\n----- SEARCH RESULTS -----\n")

for i in indices[0]:
    print(chunks[i])
    print("\n---\n")