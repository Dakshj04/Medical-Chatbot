import re
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_files(data):   
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents
def clean_text(text: str) -> str:
    """Clean raw PDF text: remove footers, URLs, phones, addresses, and normalize spaces."""
    cleaned_lines = []
    for line in text.split("\n"):
        line_stripped = line.strip()

        # Skip empty lines
        if not line_stripped:
            continue

        # Drop lines with URLs
        if re.search(r"http[s]?://|www\.", line_stripped):
            continue

        # Drop lines with phone numbers (basic US pattern, extend if needed)
        if re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", line_stripped):
            continue

        # Drop lines that look like postal addresses
        if re.search(r"\d{1,5}\s\w+.*(Street|St|Avenue|Ave|Road|Rd|Highway|Hwy)\b", line_stripped, re.IGNORECASE):
            continue

        # Drop lines with company/org boilerplate
        if any(keyword in line_stripped for keyword in ["Inc.", "Ltd", "All rights reserved", "Copyright"]):
            continue

        cleaned_lines.append(line_stripped)

    # Normalize whitespace
    return " ".join(cleaned_lines)

def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """Keep only 'source' in metadata and clean page_content."""
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source", "unknown")
        cleaned_content = clean_text(doc.page_content)  # apply full cleaner
        minimal_docs.append(
            Document(
                page_content=cleaned_content,
                metadata={"source": src}
            )
        )
    return minimal_docs
# split doc in smaller chunks

def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
      
    )
    text_chunks=text_splitter.split_documents(minimal_docs)
    return text_chunks



def download_embeddings():
    """Download and return the higging face embedding model."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
    model_name=model_name, 
) 
    return embeddings
