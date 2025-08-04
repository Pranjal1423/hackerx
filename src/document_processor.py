import PyPDF2
import docx
import requests
import tempfile
import os
import re
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing for PDFs and DOCX files"""
    
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def download_document(self, url: str) -> str:
        """Download document from URL and save to temp file"""
        try:
            logger.info(f"Downloading document from: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Determine file type from URL or headers
            content_type = response.headers.get('content-type', '')
            if 'pdf' in content_type.lower() or url.lower().endswith('.pdf'):
                suffix = '.pdf'
            elif 'document' in content_type.lower() or url.lower().endswith('.docx'):
                suffix = '.docx'
            else:
                # Default to PDF
                suffix = '.pdf'
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name
                
            logger.info(f"Document downloaded to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise Exception(f"Failed to download document: {str(e)}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            text = ""
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    logger.warning("PDF is encrypted, attempting to decrypt")
                    pdf_reader.decrypt('')
                
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n--- Page {page_num + 1} ---\n"
                            text += page_text + "\n"
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            logger.info(f"Extracting text from DOCX: {docx_path}")
            doc = docx.Document(docx_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():  # Only add non-empty paragraphs
                    text += paragraph.text + "\n"
            
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise Exception(f"Failed to extract text from DOCX: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\'\"\/\$\%\&]', '', text)
        
        # Fix common OCR errors
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        
        # Remove excessive spaces
        text = ' '.join(text.split())
        
        return text.strip()
    
    def smart_chunk_text(self, text: str) -> List[Dict[str, any]]:
        """Create smart chunks with context preservation"""
        try:
            logger.info(f"Creating chunks from text of length: {len(text)}")
            
            # Clean the text first
            text = self.clean_text(text)
            
            # Split into sentences first
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            chunks = []
            current_chunk = ""
            current_sentences = []
            
            for i, sentence in enumerate(sentences):
                # Check if adding this sentence would exceed chunk size
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) <= self.chunk_size:
                    current_chunk = potential_chunk
                    current_sentences.append(sentence)
                else:
                    # Save current chunk if it's not empty
                    if current_chunk.strip():
                        chunk_info = {
                            'text': current_chunk.strip(),
                            'start_sentence': i - len(current_sentences),
                            'end_sentence': i - 1,
                            'word_count': len(current_chunk.split()),
                            'char_count': len(current_chunk)
                        }
                        chunks.append(chunk_info)
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0 and current_sentences:
                        # Take last few sentences for overlap
                        overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                        current_chunk = " ".join(overlap_sentences) + " " + sentence
                        current_sentences = overlap_sentences + [sentence]
                    else:
                        current_chunk = sentence
                        current_sentences = [sentence]
            
            # Add the last chunk
            if current_chunk.strip():
                chunk_info = {
                    'text': current_chunk.strip(),
                    'start_sentence': len(sentences) - len(current_sentences),
                    'end_sentence': len(sentences) - 1,
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk)
                }
                chunks.append(chunk_info)
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Log chunk statistics
            if chunks:
                avg_length = sum(chunk['char_count'] for chunk in chunks) / len(chunks)
                logger.info(f"Average chunk length: {avg_length:.1f} characters")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {str(e)}")
            raise Exception(f"Failed to create chunks: {str(e)}")
    
    def process_document(self, document_url: str) -> Dict[str, any]:
        """Main method to process document from URL"""
        temp_file_path = None
        
        try:
            # Download document
            temp_file_path = self.download_document(document_url)
            
            # Extract text based on file type
            if temp_file_path.endswith('.pdf'):
                text = self.extract_text_from_pdf(temp_file_path)
            elif temp_file_path.endswith('.docx'):
                text = self.extract_text_from_docx(temp_file_path)
            else:
                raise Exception("Unsupported file type")
            
            # Create chunks
            chunks = self.smart_chunk_text(text)
            
            # Return processed document info
            result = {
                'success': True,
                'original_text_length': len(text),
                'num_chunks': len(chunks),
                'chunks': chunks,
                'raw_text': text[:1000] + "..." if len(text) > 1000 else text  # Preview
            }
            
            logger.info(f"Document processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'chunks': [],
                'num_chunks': 0
            }
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info("Temporary file cleaned up")
                except Exception as e:
                    logger.warning(f"Could not clean up temp file: {str(e)}")

# Test function
def test_document_processor():
    """Test the document processor with sample URL"""
    processor = DocumentProcessor()
    
    # Test with the provided sample URL
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    result = processor.process_document(test_url)
    
    if result['success']:
        print(f"✅ Document processed successfully!")
        print(f"📄 Number of chunks: {result['num_chunks']}")
        print(f"📝 Original text length: {result['original_text_length']}")
        print(f"🔍 First chunk preview: {result['chunks'][0]['text'][:200]}...")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    test_document_processor()