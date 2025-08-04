#!/usr/bin/env python3
"""
Server startup script for HackRX LLM Query-Retrieval System
"""

import sys
import os
import subprocess
import time

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("❌ .env file not found!")
        print("💡 Create .env file with your Groq API key:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        return False
    
    # Check if required packages are installed
    try:
        import fastapi
        import uvicorn
        import groq
        print("✅ Required packages found")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("💡 Install requirements: pip install -r requirements.txt")
        return False
    
    # Check if source files exist
    required_files = [
        'src/main.py',
        'src/document_processor.py', 
        'src/embedding_service.py',
        'src/llm_service.py',
        'src/config.py'
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Missing file: {file}")
            return False
    
    print("✅ Environment check passed")
    return True

def start_server(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI server"""
    
    if not check_environment():
        print("\n❌ Environment check failed. Fix issues before starting server.")
        return False
    
    print(f"\n🚀 Starting HackRX LLM Query-Retrieval System...")
    print(f"📡 Server will be available at: http://localhost:{port}")
    print(f"📖 API documentation at: http://localhost:{port}/docs")
    print(f"🔍 Health check at: http://localhost:{port}/health")
    print("\n⏳ Starting server (this may take a moment to load models)...")
    
    try:
        # Start uvicorn server
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "src.main:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info"
        ]
        
        if reload:
            cmd.append("--reload")
        
        print(f"🔧 Running command: {' '.join(cmd)}")
        print("=" * 60)
        
        # Run the server
        process = subprocess.run(cmd)
        return process.returncode == 0
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Failed to start server: {str(e)}")
        return False

def main():
    """Main function"""
    print("🎯 HackRX LLM Query-Retrieval System - Server Startup")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Start the HackRX API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    success = start_server(
        host=args.host,
        port=args.port, 
        reload=not args.no_reload
    )
    
    if success:
        print("\n✅ Server started successfully!")
    else:
        print("\n❌ Failed to start server")
        sys.exit(1)

if __name__ == "__main__":
    main()