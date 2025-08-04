from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import time
import asyncio
from contextlib import asynccontextmanager

# Import our services
from src.document_processor import DocumentProcessor
from src.embedding_service import EmbeddingService  
from src.llm_service import LLMService
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global services - will be initialized on startup
document_processor = None
embedding_service = None  
llm_service = None

# Authentication function
async def verify_hackathon_auth(authorization: str = Header(...)):
    """Verify hackathon authentication token"""
    expected_token = "Bearer d5f0de0bbd1d955a8dc5b316b80603e1fbdd44b65a1811a15afab5255ed5affc"
    if authorization != expected_token:
        logger.warning(f"Invalid auth token received: {authorization[:20]}...")
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global document_processor, embedding_service, llm_service
    
    logger.info("🚀 Starting HackRX LLM Query-Retrieval System...")
    
    try:
        # Initialize services
        document_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        embedding_service = EmbeddingService(Config.EMBEDDING_MODEL)
        llm_service = LLMService()
        
        # Test LLM connection
        if llm_service.test_api_connection():
            logger.info("✅ All services initialized successfully")
        else:
            logger.error("❌ LLM service connection failed")
            raise Exception("LLM service initialization failed")
            
    except Exception as e:
        logger.error(f"❌ Service initialization failed: {str(e)}")
        raise
    
    yield
    
    logger.info("🛑 Shutting down services...")

# Create FastAPI app
app = FastAPI(
    title="HackRX LLM Query-Retrieval System",
    description="Intelligent document processing and query answering system for insurance, legal, HR, and compliance domains",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., description="List of questions to answer")
    
    class Config:
        schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
                    "What is the waiting period for pre-existing diseases (PED) to be covered?",
                    "Does this policy cover maternity expenses, and what are the conditions?"
                ]
            }
        }

class QueryResponse(BaseModel):
    answers: List[str] = Field(..., description="List of answers corresponding to the questions")

# Health check endpoint (no auth required)
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "document_processor": "ready" if document_processor else "not initialized",
            "embedding_service": "ready" if embedding_service else "not initialized", 
            "llm_service": "ready" if llm_service else "not initialized"
        }
    }

# Main hackathon endpoint WITH AUTHENTICATION
@app.post("/hackrx/run", response_model=QueryResponse, dependencies=[Depends(verify_hackathon_auth)], tags=["HackRX"])
async def process_queries(request: QueryRequest):
    """
    Main endpoint for processing documents and answering questions
    
    This endpoint:
    1. Downloads and processes the PDF document
    2. Creates semantic embeddings for document chunks
    3. Searches for relevant information for each question
    4. Generates answers using LLM
    5. Returns structured JSON response
    
    Requires Authorization header: Bearer d5f0de0bbd1d955a8dc5b316b80603e1fbdd44b65a1811a15afab5255ed5affc
    """
    start_time = time.time()
    
    try:
        logger.info(f"📥 Processing authenticated request with {len(request.questions)} questions")
        logger.info(f"📄 Document URL: {request.documents[:100]}...")
        
        # Validate input
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        if len(request.questions) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 questions allowed")
        
        # Step 1: Process document
        logger.info("📖 Step 1: Processing document...")
        doc_result = document_processor.process_document(request.documents)
        
        if not doc_result['success']:
            logger.error(f"Document processing failed: {doc_result['error']}")
            raise HTTPException(
                status_code=422, 
                detail=f"Failed to process document: {doc_result['error']}"
            )
        
        logger.info(f"✅ Document processed: {doc_result['num_chunks']} chunks created")
        
        # Step 2: Create embeddings and search index
        logger.info("🧠 Step 2: Creating embeddings and search index...")
        try:
            embeddings = embedding_service.create_embeddings(doc_result['chunks'])
            embedding_service.build_faiss_index(embeddings)
            logger.info(f"✅ Search index created with {embeddings.shape[0]} vectors")
        except Exception as e:
            logger.error(f"Embedding creation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create search index: {str(e)}"
            )
        
        # Step 3: Process questions and generate answers
        logger.info("🤖 Step 3: Generating answers...")
        try:
            answers = llm_service.process_multiple_questions(request.questions, embedding_service)
            logger.info(f"✅ Generated {len(answers)} answers")
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate answers: {str(e)}"
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        logger.info(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        
        # Return response in exact hackathon format
        response = QueryResponse(answers=answers)
        
        logger.info("🎉 Authenticated request processed successfully!")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

# Test endpoint without auth for debugging
@app.post("/hackrx/run/test", response_model=QueryResponse, tags=["Testing"])
async def process_queries_test(request: QueryRequest):
    """Test endpoint without authentication - for debugging only"""
    start_time = time.time()
    
    try:
        logger.info(f"📥 Processing TEST request with {len(request.questions)} questions")
        
        # Process document
        doc_result = document_processor.process_document(request.documents)
        if not doc_result['success']:
            raise HTTPException(status_code=422, detail=doc_result['error'])
        
        # Create embeddings
        embeddings = embedding_service.create_embeddings(doc_result['chunks'])
        embedding_service.build_faiss_index(embeddings)
        
        # Generate answers
        answers = llm_service.process_multiple_questions(request.questions, embedding_service)
        
        processing_time = time.time() - start_time
        logger.info(f"⏱️  TEST processing time: {processing_time:.2f} seconds")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# System information endpoint
@app.get("/system/info", tags=["System"])
async def get_system_info():
    """Get system information and configuration"""
    return {
        "system": "HackRX LLM Query-Retrieval System",
        "version": "1.0.0",
        "config": {
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "embedding_model": Config.EMBEDDING_MODEL,
            "llm_model": Config.GROQ_MODEL,
            "max_chunks_for_context": Config.MAX_CHUNKS_FOR_CONTEXT
        },
        "endpoints": {
            "main": "/hackrx/run (requires auth)",
            "test": "/hackrx/run/test (no auth)",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HackRX LLM Query-Retrieval System",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "main": "/hackrx/run",
            "test": "/hackrx/run/test", 
            "health": "/health",
            "docs": "/docs",
            "system_info": "/system/info"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info"
    )