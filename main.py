from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define paths to the models and labels
MODEL_PATHS = {
    'leaf': {
        'model': "E:/Tanaman Herbal/model daun/mobilenetv2_model.tflite",
        'labels': "E:/Tanaman Herbal/model daun/labels.txt"
    },
    'fruit': {
        'model': "E:/Tanaman Herbal/model buah/mobilenetV2_model.tflite",
        'labels': "E:/Tanaman Herbal/model buah/labels.txt"
    },
    'rhizome': {
        'model': "E:/Tanaman Herbal/model rimpang/mobilenetv2_model.tflite",
        'labels': "E:/Tanaman Herbal/model rimpang/labels.txt"
    }
}

# Function to load the selected model and labels
def load_model_and_labels(model_name):
    model_info = MODEL_PATHS[model_name]
    interpreter = tf.lite.Interpreter(model_path=model_info['model'])
    interpreter.allocate_tensors()
    with open(model_info['labels'], 'r') as f:
        labels = f.read().splitlines()
    return interpreter, labels

# Function to preprocess the image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make predictions
def predict(image_path, model_name):
    interpreter, labels = load_model_and_labels(model_name)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data)
    confidence = output_data[0][predicted_index]
    
    return labels[predicted_index], confidence

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or 'model' not in request.form:
        return redirect(request.url)
    
    file = request.files['file']
    model_name = request.form['model']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        label, confidence = predict(filepath, model_name)
        
        return render_template('result.html', image=file.filename, label=label, confidence=confidence, model_name=model_name)

if __name__ == '__main__':
    app.run(debug=True)
