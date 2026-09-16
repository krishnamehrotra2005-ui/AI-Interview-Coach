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
    Splits text into coherent chunks of approximately `chunk_size` characters,
    strictly respecting sentence and word boundaries so text is never truncated
    mid-word (e.g. avoiding fragments like 'ical/logical' or 'em Engineering').
    
    Args:
        text: The document text to split.
        chunk_size: Target character length per chunk.
        overlap: Desired contextual overlap in characters.

    Returns:
        List of clean, grammatically coherent text chunks.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if len(cleaned) <= chunk_size:
        return [cleaned]

    # Split into logical sentence/bullet units first
    raw_segments = re.split(r'(?<=[.!?•;\n])\s+', cleaned)
    segments = [s.strip() for s in raw_segments if s.strip()]
    if not segments:
        segments = [cleaned]

    chunks = []
    current_chunk = []
    current_len = 0

    for seg in segments:
        seg_len = len(seg)

        # If an individual segment exceeds chunk_size, split it on whole word boundaries
        if seg_len > chunk_size:
            words = seg.split()
            sub_chunk = []
            sub_len = 0
            for w in words:
                if sub_len + len(w) + 1 > chunk_size and sub_chunk:
                    chunk_str = " ".join(sub_chunk).strip()
                    if chunk_str and chunk_str not in chunks:
                        chunks.append(chunk_str)
                    # Retain last few whole words for overlap
                    overlap_words = max(1, overlap // 10)
                    sub_chunk = sub_chunk[-overlap_words:]
                    sub_len = sum(len(x) + 1 for x in sub_chunk)
                sub_chunk.append(w)
                sub_len += len(w) + 1
            if sub_chunk:
                chunk_str = " ".join(sub_chunk).strip()
                if chunk_str and chunk_str not in chunks:
                    chunks.append(chunk_str)
            continue

        # Add segment to current chunk or flush if full
        if current_len + seg_len + 1 > chunk_size and current_chunk:
            chunk_str = " ".join(current_chunk).strip()
            if chunk_str and chunk_str not in chunks:
                chunks.append(chunk_str)
            # Retain the last segment for overlap if suitable
            if overlap > 0 and len(current_chunk[-1]) <= overlap:
                current_chunk = [current_chunk[-1], seg]
                current_len = len(current_chunk[0]) + seg_len + 1
            else:
                current_chunk = [seg]
                current_len = seg_len
        else:
            current_chunk.append(seg)
            current_len += seg_len + 1

    if current_chunk:
        chunk_str = " ".join(current_chunk).strip()
        if chunk_str and chunk_str not in chunks:
            chunks.append(chunk_str)

    return chunks if chunks else [cleaned]
