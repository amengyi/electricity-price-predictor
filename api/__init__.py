import os
import subprocess
import sys

def handler(event, context):
    """Vercel serverless function handler"""
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/plain'},
        'body': 'Electricity Price Predictor API'
    }

if __name__ == '__main__':
    port = os.environ.get('PORT', '8501')
    subprocess.run([
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.serverAddress', '0.0.0.0',
        '--browser.serverPort', port
    ])