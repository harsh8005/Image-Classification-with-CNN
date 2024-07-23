from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the trained model
cnn = models.load_model('trained_cnn.h5')

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((32, 32))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    predictions = cnn.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return classes[predicted_class]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        image_bytes = file.read()
        prediction = predict_image(image_bytes)
        return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
