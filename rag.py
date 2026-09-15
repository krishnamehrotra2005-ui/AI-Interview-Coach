"""
rag.py - Retrieval-Augmented Generation (RAG) Module

This module implements a basic, transparent RAG system using:
1. sentence-transformers ('all-MiniLM-L6-v2') for generating local 384-dimensional dense embeddings.
2. FAISS (IndexFlatIP) for fast, local vector similarity search using cosine similarity.
3. Plain top-k similarity retrieval to discover:
   - Matching Skills: JD requirements that closely match candidate experience.
   - Skill Gaps: JD requirements with the lowest similarity to the candidate's resume.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Standard lightweight local embedding model (runs completely offline on CPU)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Global cached model instance to avoid reloading across Streamlit interactions
_CACHED_MODEL: SentenceTransformer = None


def get_embedding_model() -> SentenceTransformer:
    """
    Returns the cached SentenceTransformer model instance.
    Downloads the model once (~80MB) and keeps it in memory.
    """
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _CACHED_MODEL


class FAISSRAGStore:
    """
    A lightweight, in-memory FAISS vector store for basic RAG.
    
    Uses IndexFlatIP (Inner Product) with L2-normalized vectors,
    which mathematically equals cosine similarity:
        cosine_similarity = dot(A, B) / (||A|| * ||B||)
    """

    def __init__(self):
        self.model = get_embedding_model()
        self.dimension = 384  # all-MiniLM-L6-v2 produces 384-dimensional vectors
        self.index = faiss.IndexFlatIP(self.dimension)
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []

    def add_texts(self, texts: List[str], metadatas: List[Dict[str, Any]] = None) -> None:
        """
        Embeds text chunks and inserts them into the FAISS index.
        """
        if not texts:
            return

        # 1. Generate dense vectors using local sentence-transformers
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype=np.float32)

        # 2. L2-normalize vectors so Inner Product becomes Cosine Similarity
        faiss.normalize_L2(embeddings)

        # 3. Add to FAISS index
        self.index.add(embeddings)
        self.documents.extend(texts)

        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most similar text chunks for a given query text.
        """
        if self.index.ntotal == 0:
            return []

        # Embed query text and normalize
        query_vec = self.model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)
        query_vec = np.array(query_vec, dtype=np.float32)
        faiss.normalize_L2(query_vec)

        # Perform top-k search in FAISS
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "similarity_score": float(score)
                })
        return results


def analyze_matches_and_gaps(
    resume_chunks: List[str],
    jd_chunks: List[str],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Compares Resume chunks against Job Description chunks using plain cosine similarity:
    
    1. Embeds all resume chunks into a FAISS index.
    2. For each chunk of the Job Description, queries the resume index for the highest similarity match.
    3. Chunks with the highest maximum similarity are classified as 'Matching Skills'.
    4. Chunks with the lowest maximum similarity are classified as 'Skill Gaps' (JD requirements missing in resume).

    Returns:
        dict containing:
        - "matching_skills": List of {jd_chunk, best_resume_match, score}
        - "skill_gaps": List of {jd_chunk, best_resume_match, score}
    """
    if not resume_chunks or not jd_chunks:
        return {"matching_skills": [], "skill_gaps": []}

    # Build vector store of resume experience chunks
    resume_store = FAISSRAGStore()
    resume_store.add_texts(resume_chunks, [{"source": "resume", "index": i} for i in range(len(resume_chunks))])

    jd_evaluations = []

    # Compare each JD requirement chunk against the resume
    for jd_chunk in jd_chunks:
        top_matches = resume_store.query(jd_chunk, top_k=1)
        if top_matches:
            best_match = top_matches[0]
            jd_evaluations.append({
                "jd_chunk": jd_chunk,
                "best_resume_match": best_match["text"],
                "similarity_score": round(best_match["similarity_score"], 3)
            })

    # Sort JD chunks by similarity score
    # High score -> Resume strongly covers this JD requirement
    # Low score  -> Resume lacks coverage of this JD requirement (Skill Gap)
    sorted_by_score = sorted(jd_evaluations, key=lambda item: item["similarity_score"], reverse=True)

    # Top-k strongest matches
    matches = sorted_by_score[:top_k]
    
    # Top-k largest gaps (lowest similarity)
    gaps = sorted_by_score[-top_k:] if len(sorted_by_score) >= top_k else sorted_by_score

    return {
        "matching_skills": matches,
        "skill_gaps": gaps
    }
