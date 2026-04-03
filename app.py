from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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