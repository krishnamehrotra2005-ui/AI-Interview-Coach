"""
parsing.py - PDF Parsing and Text Chunking Module

This module handles:
1. Extracting text from uploaded PDF files (resumes or job descriptions) using pdfplumber.
2. Cleaning and normalizing the extracted text.
3. Splitting text into manageable chunks with overlap for embedding in our vector store.
"""

import io
import re
from typing import List, Union
import pdfplumber


def clean_text(text: str) -> str:
    """
    Clean and normalize raw extracted text.
    - Replaces multiple spaces, newlines, and tabs with a single space.
    - Strips leading and trailing whitespace.
    """
    if not text:
        return ""
    # Replace carriage returns and excessive whitespace
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Replace multiple consecutive spaces with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_pdf(file_source: Union[str, io.BytesIO, bytes]) -> str:
    """
    Extract all readable text from a PDF file using pdfplumber.
    
    Args:
        file_source: Can be a file path (str), a BytesIO stream, or raw bytes
                     (such as Streamlit's UploadedFile.read()).

    Returns:
        Cleaned text extracted across all pages in the PDF.
    """
    if isinstance(file_source, bytes):
        file_source = io.BytesIO(file_source)

    extracted_pages = []
    with pdfplumber.open(file_source) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_pages.append(page_text)

    combined_text = "\n".join(extracted_pages)
    return clean_text(combined_text)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of approximately `chunk_size` characters with `overlap`.
    
    Why chunking is needed for RAG:
    Embedding models have token limits and perform best when comparing focused,
    specific paragraphs rather than an entire multi-page document at once.
    
    Args:
        text: The normalized document text to split.
        chunk_size: Target character length per chunk.
        overlap: Overlap in characters between adjacent chunks to prevent cutting off context.

    Returns:
        List of non-empty text chunk strings.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = start + chunk_size
        chunk = cleaned[start:end]

        # Try to break at a natural sentence or word boundary if not at the end of the text
        if end < text_length:
            last_space = chunk.rfind(" ")
            if last_space > chunk_size // 2:
                chunk = chunk[:last_space]
                start += last_space - overlap
            else:
                start += chunk_size - overlap
        else:
            start += chunk_size

        chunk = chunk.strip()
        if chunk and chunk not in chunks:
            chunks.append(chunk)

    return chunks
