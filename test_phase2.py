import sys
import os
sys.path.append('src')

from document_processor import DocumentProcessor

def test_document_processing():
    """Test document processing functionality"""
    print("🚀 Testing Document Processor - Phase 2")
    print("=" * 50)
    
    # Initialize processor
    processor = DocumentProcessor(chunk_size=400, chunk_overlap=50)
    
    # Test URL from hackathon requirements
    test_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    print("📥 Testing document download and processing...")
    result = processor.process_document(test_url)
    
    if result['success']:
        print("✅ SUCCESS: Document processed successfully!")
        print(f"📊 Statistics:")
        print(f"   - Original text length: {result['original_text_length']:,} characters")
        print(f"   - Number of chunks created: {result['num_chunks']}")
        
        if result['chunks']:
            # Show first few chunks
            print(f"\n📝 First 3 chunks preview:")
            for i, chunk in enumerate(result['chunks'][:3]):
                print(f"\n   Chunk {i+1}:")
                print(f"   - Length: {chunk['char_count']} chars, {chunk['word_count']} words")
                print(f"   - Preview: {chunk['text'][:150]}...")
        
        print(f"\n🎯 Phase 2 COMPLETE! Document processing working perfectly.")
        return True
        
    else:
        print(f"❌ FAILED: {result['error']}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Verify the URL is accessible")
        print("3. Make sure all packages are installed: pip install -r requirements.txt")
        return False

def test_chunking_strategy():
    """Test chunking with sample text"""
    print("\n🧪 Testing chunking strategy...")
    
    processor = DocumentProcessor(chunk_size=200, chunk_overlap=30)
    
    sample_text = """
    This is a sample insurance policy document. The policy covers various medical expenses.
    There is a waiting period of 36 months for pre-existing diseases. The grace period for premium payment is 30 days.
    Maternity expenses are covered after 24 months of continuous coverage. Cataract surgery has a waiting period of 2 years.
    The No Claim Discount is 5% of the base premium. Health check-ups are reimbursed every two years.
    """
    
    chunks = processor.smart_chunk_text(sample_text)
    
    print(f"   Created {len(chunks)} chunks from sample text")
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {chunk['char_count']} chars - {chunk['text'][:100]}...")

if __name__ == "__main__":
    success = test_document_processing()
    test_chunking_strategy()
    
    if success:
        print("\n🎉 Phase 2 Testing Complete - Ready for Phase 3!")
    else:
        print("\n⚠️  Fix Phase 2 issues before proceeding")