#!/usr/bin/env python3
import sys
import os

def main():
    # Ensure current directory is on PYTHONPATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        import uvicorn
        # Import to fail early if server.py has import issues
        from analyzer.server import app
    except ImportError as e:
        print(f"Error: Required dependency missing ({e}).", file=sys.stderr)
        print("Please install requirements: pip install fastapi uvicorn python-multipart", file=sys.stderr)
        sys.exit(1)

    print("Starting LLM Training Data Quality Analyzer UI...")
    print("Navigate to http://127.0.0.1:8000/ in your browser.")
    uvicorn.run("analyzer.server:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
