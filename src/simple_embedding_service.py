# Save this as src/simple_embedding_service.py

import numpy as np
import faiss
from typing import List, Dict, Optional
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

class SimpleEmbeddingService:
    """Simple embedding service using TF-IDF + FAISS as backup"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = None
        
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create TF-IDF embeddings"""
        try:
            logger.info(f"Creating TF-IDF embeddings for {len(chunks)} chunks")
            
            # Extract text
            texts = [chunk['text'] for chunk in chunks]
            self.chunks = texts
            self.chunk_metadata = chunks
            
            # Create TF-IDF embeddings
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            self.embeddings = embeddings.astype('float32')
            
            logger.info(f"Created embeddings with shape: {self.embeddings.shape}")
            return self.embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS index"""
        try:
            logger.info("Building FAISS index...")
            
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Create index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            
            logger.info(f"FAISS index built with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}")
            raise
    
    def search_similar(self, query: str, k: int = 5, min_score: float = 0.1) -> List[Dict]:
        """Search similar chunks"""
        try:
            logger.info(f"Searching for: '{query}'")
            
            # Transform query
            query_vec = self.vectorizer.transform([query]).toarray().astype('float32')
            faiss.normalize_L2(query_vec)
            
            # Search
            scores, indices = self.index.search(query_vec, k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= min_score:
                    results.append({
                        'rank': i + 1,
                        'chunk_id': int(idx),
                        'text': self.chunks[idx],
                        'similarity_score': float(score),
                        'metadata': self.chunk_metadata[idx],
                        'word_count': self.chunk_metadata[idx]['word_count'],
                        'char_count': self.chunk_metadata[idx]['char_count']
                    })
            
            logger.info(f"Found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Hybrid search with keyword boosting"""
        results = self.search_similar(query, k=k*2, min_score=0.05)
        
        # Add keyword scoring
        query_words = set(query.lower().split())
        
        for result in results:
            text_words = set(result['text'].lower().split())
            overlap = len(query_words.intersection(text_words))
            keyword_score = overlap / len(query_words) if query_words else 0
            
            result['keyword_score'] = keyword_score
            result['combined_score'] = result['similarity_score'] * 0.6 + keyword_score * 0.4
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:k]
    
    def get_index_stats(self) -> Dict:
        """Get index statistics"""
        if self.index is None:
            return {'status': 'No index built'}
        
        return {
            'status': 'Ready',
            'total_chunks': self.index.ntotal,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'model_name': 'TF-IDF + FAISS',
            'index_type': 'Simple TF-IDF Embeddings'
        }