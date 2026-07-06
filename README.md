Personal Knowledge Assistant (RAG Chatbot)

A Retrieval-Augmented Generation (RAG) system that lets you upload a PDF and ask natural-language questions about its content. The app retrieves the most relevant passages from the document and uses them to generate a grounded answer — entirely with free, local models (no API keys required).

Features

PDF ingestion: upload any PDF and extract its text automatically.

Chunking: splits document text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter for better retrieval precision.

Semantic search: embeds chunks with all-MiniLM-L6-v2 (Sentence-Transformers) and indexes them with FAISS for fast similarity search.

Local answer generation: uses google/flan-t5-base to generate answers grounded in retrieved context — fully offline, no external API costs.

Interactive UI: Streamlit app with file upload, live Q&A, and an expandable panel to inspect exactly which chunks were retrieved for each answer.

Cached model/document loading for fast repeated queries within a session.

Tech Stack

ComponentToolText extractionPyPDFChunkingLangChain Text SplittersEmbeddingsSentence-Transformers (all-MiniLM-L6-v2)Vector storeFAISSGenerationHugging Face Transformers (flan-t5-base)UIStreamlit

How It Works

Ingestion (once per document):
PDF upload → text extraction → chunking → embedding → FAISS index

Query (per question):
User question → query embedding → top-k retrieval from FAISS → context + question assembled into a prompt → Flan-T5 generates the answer

Getting Started

bashgit clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app.py

Then open the local URL Streamlit prints in your browser, upload a PDF, and start asking questions.

Known Limitations

flan-t5-base has a limited input context window, so very long retrieved contexts can get truncated.
Single-turn Q&A only — no conversation memory across questions yet.
Retrieval is pure dense vector similarity; no keyword/hybrid search or reranking yet.
No formal evaluation pipeline yet — retrieval and answer quality have been assessed manually.
Currently supports PDF only.

Author

Built by Khushi Saraswat as part of ongoing work in retrieval-augmented systems.
