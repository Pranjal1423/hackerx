import requests
import json
import time

def test_authenticated_endpoint():
    """Test the authenticated /hackrx/run endpoint"""
    print("🔐 Testing Authenticated Endpoint")
    print("=" * 50)
    
    # Server URL
    BASE_URL = "http://localhost:8000"
    ENDPOINT = f"{BASE_URL}/hackrx/run"
    
    # Required authentication header
    HEADERS = {
        "Content-Type": "application/json",
        "Authorization": "Bearer d5f0de0bbd1d955a8dc5b316b80603e1fbdd44b65a1811a15afab5255ed5affc"
    }
    
    # Test request
    test_data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment?",
            "What is the waiting period for pre-existing diseases?"
        ]
    }
    
    print(f"📡 Testing endpoint: {ENDPOINT}")
    print(f"📤 Sending {len(test_data['questions'])} questions...")
    
    try:
        # Send authenticated request
        start_time = time.time()
        response = requests.post(
            ENDPOINT,
            headers=HEADERS,
            json=test_data,
            timeout=120
        )
        processing_time = time.time() - start_time
        
        print(f"⏱️  Request took: {processing_time:.2f} seconds")
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS! Authentication working!")
            print(f"📝 Received {len(result['answers'])} answers")
            
            # Show answers
            for i, (question, answer) in enumerate(zip(test_data['questions'], result['answers']), 1):
                print(f"\n   Q{i}: {question}")
                print(f"   A{i}: {answer[:100]}...")
            
            print("\n🎉 Your endpoint is ready for hackathon submission!")
            return True
            
        elif response.status_code == 401:
            print("❌ AUTHENTICATION FAILED!")
            print("🔧 Check if your main.py has the auth function")
            return False
            
        else:
            print(f"❌ Request failed with status: {response.status_code}")
            try:
                error_data = response.json()
                print(f"📋 Error details: {error_data}")
            except:
                print(f"📋 Error text: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR!")
        print("💡 Make sure your server is running:")
        print("   python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ REQUEST TIMEOUT (>2 minutes)")
        print("💡 This might be normal for first run (model loading)")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def test_without_auth():
    """Test what happens without authentication"""
    print("\n🚫 Testing WITHOUT Authentication (should fail)")
    print("=" * 50)
    
    try:
        response = requests.post(
            "http://localhost:8000/hackrx/run",
            headers={"Content-Type": "application/json"},
            json={
                "documents": "https://example.com/test.pdf",
                "questions": ["Test question?"]
            },
            timeout=10
        )
        
        if response.status_code == 401:
            print("✅ GOOD! Authentication is required (401 Unauthorized)")
            return True
        else:
            print(f"⚠️  Unexpected status without auth: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing without auth: {str(e)}")
        return False

def test_health_check():
    """Test health check endpoint"""
    print("\n❤️  Testing Health Check")
    print("=" * 30)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health_data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🧪 Authentication & Endpoint Testing")
    print("=" * 60)
    
    # Test health first
    health_ok = test_health_check()
    if not health_ok:
        print("\n❌ Server not responding. Start server first!")
        return
    
    # Test without auth (should fail)
    test_without_auth()
    
    # Test with auth (should work)
    auth_ok = test_authenticated_endpoint()
    
    print("\n" + "=" * 60)
    if auth_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Your API is ready for hackathon submission")
        print("\n📋 Next steps:")
        print("1. Use ngrok to expose your server")
        print("2. Submit the ngrok URL + /hackrx/run")
        print("3. Include the auth header in your submission notes")
    else:
        print("❌ TESTS FAILED!")
        print("🔧 Fix the issues before submitting")

if __name__ == "__main__":
    main()