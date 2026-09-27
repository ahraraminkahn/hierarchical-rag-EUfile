Hierarchical RAG Pipeline for PDF Document Q&A

A Retrieval-Augmented Generation (RAG) pipeline that extracts structured content from a PDF, builds hierarchy-aware chunks, and answers natural-language questions using hybrid (vector + keyword) retrieval, cross-encoder reranking, and an LLM.

Overview

This project ingests a PDF (the notebook is set up around the EU AI Act, though it will work with any structured document), reconstructs its chapter/section/subsection hierarchy from the raw text, and enriches each chunk with that structural metadata plus a detected content type (definition, example, theorem, algorithm, equation, or concept). The enriched chunks are embedded and stored in a vector database, then retrieved through an ensemble of semantic and keyword search, reranked with a cross-encoder, and passed to an LLM to produce grounded answers.

Pipeline Steps
PDF Extraction — Reads the PDF page by page with pypdf, capturing page numbers alongside text.
Hierarchy Detection — Uses regex patterns to identify chapters, sections, and subsections from the page text.
Document Construction — Builds langchain_core.documents.Document objects per page, tagged with the current chapter/section/subsection.
Chunking — Splits documents into ~1000-character chunks (150-character overlap) with RecursiveCharacterTextSplitter.
Metadata Enrichment — Adds chunk IDs, document/author tags, detected content type, and a derived topic field (subsection → section → chapter title fallback).
Embedding & Vector Store — Embeds chunks with sentence-transformers/all-MiniLM-L6-v2 and stores them in a local Qdrant collection.
Hybrid Retrieval — Combines a Qdrant similarity retriever with a BM25Retriever via EnsembleRetriever (50/50 weighting).
Reranking — Reranks the top candidates with a cross-encoder/ms-marco-MiniLM-L-6-v2 cross-encoder, keeping the top 3.
Answer Generation — Feeds the reranked context into a prompt template and generates an answer with gpt-4o-mini via ChatOpenAI.
Tech Stack
PDF parsing: pypdf
Orchestration: langchain, langchain-community, langchain-classic, langchain-core, langchain-text-splitters
Embeddings: langchain-huggingface, sentence-transformers
Vector store: qdrant-client, langchain-qdrant
Keyword retrieval: rank_bm25
LLM: langchain-openai (gpt-4o-mini)
Tokenization: tiktoken
Requirements
bash
pip install pypdf langchain langchain-community langchain-openai \
    langchain-qdrant qdrant-client tiktoken langchain-core \
    langchain-classic langchain-huggingface langchain-text-splitters \
    sentence-transformers rank_bm25

You will also need an OpenAI API key available as an environment variable (the notebook currently reads it via google.colab.userdata, intended for Colab; swap this for os.environ["OPENAI_API_KEY"] when running locally).

Usage
Place your source PDF in the project directory and update PDF_PATH to point to it.
Run the notebook/script top to bottom:
Extracts and structures the PDF
Builds and stores embeddings in Qdrant (persisted at ./qdrant_db)
Sets up the hybrid retriever + reranker
Ask a question:
python
query = "what are the classification rules for high-risk AI systems?"
response = chain.invoke(query)
print(response.content)
Project Structure
.
├── EU-AI-Act.pdf        # Source PDF (or your own document)
├── qdrant_db/           # Local persisted Qdrant vector store (generated)
└── notebook.ipynb       # Main pipeline
Notes & Known Issues
Chunk metadata currently hardcodes document = "Understanding Deep Learning" and author = "Simon J. D. Prince" — leftover from another source document. Update these fields to match whatever PDF you're actually processing.
The chapter/section/subsection regex patterns assume a fairly clean numbered-heading structure (1 Title, 1.1 Title, 1.1.1 Title); scanned or inconsistently formatted PDFs may need adjusted patterns.
The google.colab.userdata import ties the LLM setup to Google Colab; replace with standard environment variable handling for local or production use.
Possible Improvements
Externalize configuration (PDF path, model names, chunk size) into a config file or CLI args.
Add automated evaluation of retrieval/answer quality.
Cache embeddings to avoid recomputing on every run.
Add a lightweight UI (e.g., Streamlit) for interactive querying.
