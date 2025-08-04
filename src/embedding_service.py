import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Handles text embeddings and semantic search using FAISS"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding service
        Args:
            model_name: HuggingFace sentence transformer model name
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.dimension = None
        
        # Initialize the model
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Test model to get dimension
            test_embedding = self.model.encode(["test"])
            self.dimension = test_embedding.shape[1]
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise Exception(f"Failed to load embedding model: {str(e)}")
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """
        Create embeddings for document chunks
        Args:
            chunks: List of chunk dictionaries with 'text' key
        Returns:
            numpy array of embeddings
        """
        try:
            logger.info(f"Creating embeddings for {len(chunks)} chunks")
            
            # Extract text from chunks
            texts = [chunk['text'] for chunk in chunks]
            
            # Store chunks and metadata
            self.chunks = texts
            self.chunk_metadata = chunks
            
            # Create embeddings in batches to avoid memory issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
                batch_embeddings = self.model.encode(
                    batch_texts,
                    batch_size=16,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            embeddings = np.vstack(all_embeddings)
            
            logger.info(f"Created embeddings with shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    def build_faiss_index(self, embeddings: np.ndarray):
        """
        Build FAISS index for semantic search
        Args:
            embeddings: numpy array of embeddings
        """
        try:
            logger.info("Building FAISS index...")
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Create FAISS index (Inner Product for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}")
            raise Exception(f"Failed to build FAISS index: {str(e)}")
    
    def search_similar(self, query: str, k: int = 5, min_score: float = 0.3) -> List[Dict]:
        """
        Search for similar chunks using semantic similarity
        Args:
            query: search query string
            k: number of results to return
            min_score: minimum similarity score threshold
        Returns:
            List of similar chunks with metadata
        """
        try:
            if self.index is None or self.model is None:
                raise Exception("Index or model not initialized")
            
            logger.info(f"Searching for: '{query}' (top {k} results)")
            
            # Create query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if score >= min_score:  # Filter by minimum score
                    result = {
                        'rank': i + 1,
                        'chunk_id': int(idx),
                        'text': self.chunks[idx],
                        'similarity_score': float(score),
                        'metadata': self.chunk_metadata[idx],
                        'word_count': self.chunk_metadata[idx]['word_count'],
                        'char_count': self.chunk_metadata[idx]['char_count']
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results above threshold {min_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            raise Exception(f"Failed to search: {str(e)}")
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching
        Args:
            query: search query string
            k: number of results to return
        Returns:
            List of results with combined scoring
        """
        try:
            # Get semantic search results
            semantic_results = self.search_similar(query, k=k*2, min_score=0.1)
            
            # Keyword boosting
            query_keywords = set(query.lower().split())
            
            for result in semantic_results:
                text_words = set(result['text'].lower().split())
                keyword_overlap = len(query_keywords.intersection(text_words))
                keyword_score = keyword_overlap / len(query_keywords) if query_keywords else 0
                
                # Combine semantic and keyword scores
                result['keyword_score'] = keyword_score
                result['combined_score'] = result['similarity_score'] * 0.7 + keyword_score * 0.3
            
            # Sort by combined score and return top k
            hybrid_results = sorted(semantic_results, key=lambda x: x['combined_score'], reverse=True)[:k]
            
            logger.info(f"Hybrid search returned {len(hybrid_results)} results")
            return hybrid_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return self.search_similar(query, k)  # Fallback to semantic search
    
    def save_index(self, filepath: str):
        """Save FAISS index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{filepath}.faiss")
            
            # Save metadata
            metadata = {
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension
            }
            
            with open(f"{filepath}.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Index saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
    
    def load_index(self, filepath: str):
        """Load FAISS index and metadata from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            # Load metadata
            with open(f"{filepath}.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.chunks = metadata['chunks']
            self.chunk_metadata = metadata['chunk_metadata']
            self.dimension = metadata['dimension']
            
            logger.info(f"Index loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
    
    def get_index_stats(self) -> Dict:
        """Get statistics about the current index"""
        if self.index is None:
            return {'status': 'No index built'}
        
        return {
            'status': 'Ready',
            'total_chunks': self.index.ntotal,
            'embedding_dimension': self.dimension,
            'model_name': self.model_name,
            'index_type': 'FAISS IndexFlatIP (Cosine Similarity)'
        }

# Test function for embedding service
def test_embedding_service():
    """Test the embedding service functionality"""
    # Sample chunks for testing
    test_chunks = [
        {
            'text': 'The policy covers medical expenses for hospitalization and surgery.',
            'word_count': 10,
            'char_count': 67
        },
        {
            'text': 'There is a waiting period of 36 months for pre-existing diseases.',
            'word_count': 12,
            'char_count': 65
        },
        {
            'text': 'Maternity expenses are covered after 24 months of continuous coverage.',
            'word_count': 11,
            'char_count': 70
        },
        {
            'text': 'The grace period for premium payment is 30 days after due date.',
            'word_count': 12,
            'char_count': 63
        }
    ]
    
    # Initialize service
    service = EmbeddingService()
    
    # Create embeddings
    embeddings = service.create_embeddings(test_chunks)
    
    # Build index
    service.build_faiss_index(embeddings)
    
    # Test searches
    test_queries = [
        "What is the waiting period for diseases?",
        "How long is the grace period for payment?",
        "Does the policy cover pregnancy expenses?"
    ]
    
    for query in test_queries:
        results = service.hybrid_search(query, k=2)
        print(f"\nQuery: {query}")
        for result in results:
            print(f"  Score: {result['combined_score']:.3f} - {result['text'][:60]}...")

if __name__ == "__main__":
    test_embedding_service()