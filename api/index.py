import os
import subprocess
import sys

def handler(event, context):
    """Vercel serverless function handler for Streamlit"""
    port = os.environ.get('PORT', '8501')
    
    result = subprocess.run(
        [sys.executable, '-m', 'streamlit', 'run', '../app.py',
         '--server.port', port,
         '--server.address', '0.0.0.0',
         '--server.headless', 'true'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/plain'},
            'body': f"Streamlit failed to start:\n{result.stderr}"
        }
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain'},
        'body': 'Streamlit server started'
    }