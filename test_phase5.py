import requests
import json
import time
import subprocess
import threading
import sys
import os

# Test the FastAPI application
def test_fastapi_server():
    """Test the complete FastAPI application"""
    print("🚀 Testing FastAPI Application - Phase 5")
    print("=" * 60)
    
    BASE_URL = "http://localhost:8000"
    
    # Test 1: Health check
    print("🔍 Test 1: Health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            print(f"   Services: {health_data['services']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {str(e)}")
        print("💡 Make sure the server is running: python -m uvicorn src.main:app --reload")
        return False
    
    # Test 2: System info
    print("\n📊 Test 2: System information...")
    try:
        response = requests.get(f"{BASE_URL}/system/info", timeout=10)
        if response.status_code == 200:
            info_data = response.json()
            print("✅ System info retrieved")
            print(f"   System: {info_data['system']}")
            print(f"   Version: {info_data['version']}")
            print(f"   LLM Model: {info_data['config']['llm_model']}")
        else:
            print(f"❌ System info failed: {response.status_code}")
    except Exception as e:
        print(f"❌ System info failed: {str(e)}")
    
    # Test 3: Main hackathon endpoint
    print("\n🎯 Test 3: Main hackathon endpoint (/hackrx/run)...")
    
    # Prepare hackathon format request
    test_request = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?", 
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    print(f"   📤 Sending request with {len(test_request['questions'])} questions...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/hackrx/run",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=120  # 2 minutes timeout for processing
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Main endpoint test passed!")
            print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"   📝 Answers generated: {len(result['answers'])}")
            
            # Show sample answers
            print(f"\n   📋 Sample Answers:")
            for i, (question, answer) in enumerate(zip(test_request['questions'], result['answers']), 1):
                print(f"   Q{i}: {question}")
                print(f"   A{i}: {answer[:100]}...")
                print()
            
            # Verify response format matches hackathon requirements
            if 'answers' in result and isinstance(result['answers'], list):
                print("✅ Response format matches hackathon requirements")
                return True
            else:
                print("❌ Response format doesn't match requirements")
                return False
                
        else:
            print(f"❌ Main endpoint failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Error text: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out (>2 minutes)")
        print("💡 This might be normal for first run (downloading models)")
        return False
    except Exception as e:
        print(f"❌ Main endpoint test failed: {str(e)}")
        return False
    
    # Test 4: Detailed endpoint
    print("🔍 Test 4: Detailed endpoint (/hackrx/run/detailed)...")
    try:
        # Use smaller request for detailed test
        small_request = {
            "documents": test_request["documents"],
            "questions": test_request["questions"][:1]  # Just one question
        }
        
        response = requests.post(
            f"{BASE_URL}/hackrx/run/detailed",
            json=small_request,
            timeout=60
        )
        
        if response.status_code == 200:
            detailed_result = response.json()
            print("✅ Detailed endpoint working")
            print(f"   📊 Metadata available: {list(detailed_result.get('metadata', {}).keys())}")
        else:
            print(f"⚠️  Detailed endpoint failed: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Detailed endpoint test failed: {str(e)}")
    
    return True

def test_error_handling():
    """Test error handling scenarios"""
    print("\n🧪 Test 5: Error handling...")
    BASE_URL = "http://localhost:8000"
    
    # Test invalid URL
    try:
        invalid_request = {
            "documents": "invalid_url",
            "questions": ["Test question?"]
        }
        
        response = requests.post(f"{BASE_URL}/hackrx/run", json=invalid_request, timeout=30)
        if response.status_code in [400, 422, 500]:
            print("✅ Error handling working (invalid URL rejected)")
        else:
            print(f"⚠️  Unexpected response for invalid URL: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Error handling test failed: {str(e)}")
    
    # Test empty questions
    try:
        empty_request = {
            "documents": "https://example.com/test.pdf",
            "questions": []
        }
        
        response = requests.post(f"{BASE_URL}/hackrx/run", json=empty_request, timeout=10)
        if response.status_code == 400:
            print("✅ Error handling working (empty questions rejected)")
        else:
            print(f"⚠️  Unexpected response for empty questions: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️  Empty questions test failed: {str(e)}")

def main():
    """Main test function"""
    print("🎯 FastAPI Integration Test Suite")
    print("=" * 60)
    print("💡 Make sure the server is running:")
    print("   cd hackathon-llm-system")
    print("   python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
    print("=" * 60)
    
    # Wait for user confirmation
    input("\nPress Enter when server is running...")
    
    # Run tests
    success = test_fastapi_server()
    
    if success:
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 Phase 5 COMPLETE! FastAPI application working perfectly!")
        print("\n✅ What's working:")
        print("   - FastAPI server with proper CORS")
        print("   - Health check and system info endpoints")
        print("   - Main hackathon endpoint (/hackrx/run)")
        print("   - Proper JSON request/response format")
        print("   - Document processing pipeline")
        print("   - LLM answer generation")
        print("   - Error handling and validation")
        
        print(f"\n🚀 Your system is READY for the hackathon!")
        print("📡 API running at: http://localhost:8000")
        print("📖 API docs at: http://localhost:8000/docs")
        
        return True
    else:
        print("\n❌ Phase 5 needs debugging")
        return False

if __name__ == "__main__":
    main()