from flask import Flask, render_template, request, send_file
import requests
import io
from PIL import Image
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Hugging Face API token (replace with your actual token)
HUGGINGFACE_TOKEN = "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"

# Define the API URL and headers
API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}


def query(payload):
    try:
        logger.info(f"Sending request to API with prompt: {payload['inputs']}")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        logger.info(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            logger.info("Successfully received image data")
            return response.content
        else:
            logger.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.Timeout:
        logger.error("Request timed out. The model might be loading.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}")
        return None


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        prompt = request.form['prompt']
        logger.info(f"Received prompt: {prompt}")

        image_bytes = query({"inputs": prompt})

        if image_bytes:
            try:
                image = Image.open(io.BytesIO(image_bytes))
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                return render_template('index.html', image=img_str, prompt=prompt)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return render_template('index.html', error=f"Error processing image: {str(e)}", prompt=prompt)
        else:
            return render_template('index.html', error="Failed to generate image. Please try again.", prompt=prompt)

    return render_template('index.html')


@app.route('/download')
def download_image():
    prompt = request.args.get('prompt')
    image_bytes = query({"inputs": prompt})
    if image_bytes:
        return send_file(io.BytesIO(image_bytes), mimetype='image/png', as_attachment=True,
                         download_name='generated_image.png')
    return "Image generation failed", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
