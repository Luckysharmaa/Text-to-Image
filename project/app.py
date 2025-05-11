from flask import Flask, render_template, request, send_file
import requests
import io
from PIL import Image
import base64
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# Hugging Face API token
HUGGINGFACE_TOKEN= "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"

# Define the API URL and headers
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

def query(payload):
    try:
        logger.info(f"Sending request to API with prompt: {payload['inputs']}")
        # Add timeout to the request
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        
        # Log the response status and headers
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")
        
        # Check for successful response
        if response.status_code == 200:
            logger.info("Successfully received image data")
            return response.content
        else:
            error_msg = f"Error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return None
    except requests.exceptions.Timeout:
        error_msg = "Request timed out. The model might be loading. Please try again."
        logger.error(error_msg)
        return None
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error: {str(e)}"
        logger.error(error_msg)
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        logger.info(f"Received prompt: {prompt}")
        
        image_bytes = query({"inputs": prompt})
        
        if image_bytes:
            try:
                # Convert image to base64 for displaying in HTML
                image = Image.open(io.BytesIO(image_bytes))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                logger.info("Successfully converted image to base64")
                return render_template('index.html', image=img_str, prompt=prompt)
            except Exception as e:
                error_msg = f"Error processing image: {str(e)}"
                logger.error(error_msg)
                return render_template('index.html', error=error_msg, prompt=prompt)
        else:
            error_msg = "Failed to generate image. Please try again."
            logger.error(error_msg)
            return render_template('index.html', error=error_msg, prompt=prompt)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Ensure the templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)