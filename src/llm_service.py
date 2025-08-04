import os
import json
import logging
from typing import List, Dict, Optional
from groq import Groq
from src.config import Config

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating answers using Groq API"""
    
    def __init__(self):
        """Initialize Groq client"""
        api_key = os.getenv("GROQ_API_KEY") or Config.GROQ_API_KEY
        
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables or config")
        
        self.client = Groq(api_key=api_key)
        self.model = Config.GROQ_MODEL
        
        logger.info(f"LLM Service initialized with model: {self.model}")
    
    def create_context_from_chunks(self, search_results: List[Dict], max_chunks: int = 3) -> str:
        """Create context string from search results"""
        if not search_results:
            return "No relevant information found in the document."
        
        context_parts = []
        for i, result in enumerate(search_results[:max_chunks], 1):
            score = result.get('similarity_score', 0)
            text = result.get('text', '').strip()
            
            context_parts.append(f"[Context {i} - Relevance: {score:.3f}]\n{text}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, question: str, search_results: List[Dict]) -> str:
        """Generate answer using Groq API"""
        try:
            # Create context from search results
            context = self.create_context_from_chunks(search_results, max_chunks=3)
            
            # Create prompt optimized for insurance/policy documents
            prompt = self._create_insurance_prompt(question, context)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert insurance policy analyst. Provide accurate, concise answers based only on the given context. If information is not in the context, say so clearly."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                model=self.model,
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=300,   # Reasonable limit for concise answers
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            
            logger.info(f"Generated answer for question: {question[:50]}...")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"I apologize, but I encountered an error while processing your question: {str(e)}"
    
    def _create_insurance_prompt(self, question: str, context: str) -> str:
        """Create optimized prompt for insurance policy questions"""
        
        prompt = f"""Based on the insurance policy information provided below, answer the question accurately and concisely.

POLICY CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context above
2. Be specific about amounts, time periods, conditions, and limitations
3. If the exact information isn't in the context, state "This specific information is not provided in the available policy text"
4. Keep the answer clear and direct
5. Include relevant policy details like waiting periods, coverage limits, or conditions

ANSWER:"""
        
        return prompt
    
    def generate_structured_answer(self, question: str, search_results: List[Dict]) -> Dict:
        """Generate structured answer with metadata"""
        try:
            # Generate the main answer
            answer = self.generate_answer(question, search_results)
            
            # Calculate confidence based on search results
            confidence = self._calculate_confidence(search_results)
            
            # Extract supporting evidence
            evidence = self._extract_evidence(search_results)
            
            return {
                "question": question,
                "answer": answer,
                "confidence": confidence,
                "supporting_evidence": evidence,
                "source_chunks_used": len(search_results),
                "max_relevance_score": max([r.get('similarity_score', 0) for r in search_results]) if search_results else 0
            }
            
        except Exception as e:
            logger.error(f"Error generating structured answer: {str(e)}")
            return {
                "question": question,
                "answer": f"Error processing question: {str(e)}",
                "confidence": "low",
                "supporting_evidence": [],
                "source_chunks_used": 0,
                "max_relevance_score": 0
            }
    
    def _calculate_confidence(self, search_results: List[Dict]) -> str:
        """Calculate confidence level based on search results"""
        if not search_results:
            return "low"
        
        max_score = max([r.get('similarity_score', 0) for r in search_results])
        avg_score = sum([r.get('similarity_score', 0) for r in search_results]) / len(search_results)
        
        if max_score >= 0.7 and avg_score >= 0.5:
            return "high"
        elif max_score >= 0.5 and avg_score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def _extract_evidence(self, search_results: List[Dict], max_evidence: int = 2) -> List[str]:
        """Extract key evidence snippets from search results"""
        evidence = []
        
        for result in search_results[:max_evidence]:
            text = result.get('text', '').strip()
            score = result.get('similarity_score', 0)
            
            # Extract the most relevant sentence or phrase
            sentences = text.split('. ')
            if sentences:
                # Take the first sentence as key evidence
                key_sentence = sentences[0].strip()
                if len(key_sentence) > 20:  # Only if meaningful length
                    evidence.append(f"{key_sentence} (relevance: {score:.3f})")
        
        return evidence
    
    def process_multiple_questions(self, questions: List[str], embedding_service) -> List[str]:
        """Process multiple questions and return answers in required format"""
        answers = []
        
        for i, question in enumerate(questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(questions)}: {question}")
                
                # Search for relevant chunks
                search_results = embedding_service.hybrid_search(question, k=5)
                
                # Generate answer
                answer = self.generate_answer(question, search_results)
                answers.append(answer)
                
                logger.info(f"Answer {i} generated successfully")
                
            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                answers.append(f"I apologize, but I couldn't process this question due to an error: {str(e)}")
        
        return answers
    
    def test_api_connection(self) -> bool:
        """Test if Groq API is working"""
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": "Say 'API test successful'"}],
                model=self.model,
                max_tokens=10
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"API test result: {result}")
            return "successful" in result.lower()
            
        except Exception as e:
            logger.error(f"API test failed: {str(e)}")
            return False

# Test function
def test_llm_service():
    """Test LLM service functionality"""
    try:
        # Initialize service
        llm_service = LLMService()
        
        # Test API connection
        print("🧪 Testing Groq API connection...")
        if llm_service.test_api_connection():
            print("✅ Groq API connection successful!")
        else:
            print("❌ Groq API connection failed!")
            return False
        
        # Test answer generation with sample data
        print("\n🧪 Testing answer generation...")
        
        sample_search_results = [
            {
                'text': 'Grace Period means the specified period of time, immediately following the premium due date during which premium payment may be made to renew or continue the policy without losing continuity benefits. A grace period of thirty days is provided.',
                'similarity_score': 0.85,
                'rank': 1
            },
            {
                'text': 'The policy shall be renewable at the option of the company. Premium payment must be made within the grace period to maintain coverage.',
                'similarity_score': 0.65,
                'rank': 2
            }
        ]
        
        test_question = "What is the grace period for premium payment?"
        answer = llm_service.generate_answer(test_question, sample_search_results)
        
        print(f"Question: {test_question}")
        print(f"Answer: {answer}")
        
        # Test structured answer
        structured = llm_service.generate_structured_answer(test_question, sample_search_results)
        print(f"\nStructured Answer:")
        print(f"Confidence: {structured['confidence']}")
        print(f"Evidence: {structured['supporting_evidence']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Service test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_llm_service()