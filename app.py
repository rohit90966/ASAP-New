import os
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Path to the model
model_path = 'E:/VScode/Models/my_model.h5'
model = load_model(model_path)
print("Model input shape:", model.input_shape)

# Class labels (same as before)
class_labels = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Ensure the 'uploads' folder exists for image storage
if not os.path.exists('static/uploads'):
    os.makedirs('static/uploads')

@app.route('/')
def index():
    return render_template('index.html')  # Display the HTML form

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to the uploads directory
    img_path = os.path.join('static/uploads', file.filename)
    file.save(img_path)

    # Preprocess the image
    img = Image.open(img_path).convert("RGB")  # Ensure it's RGB
    img = img.resize((150, 150))  # Resize to 150x150 (required shape)
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        print("Model input shape:", model.input_shape)  # Debugging step

        # Make the prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)

        # Get class label from the class_labels dictionary
        predicted_class_name = class_labels[predicted_class]

        # Pass the prediction and image filename to the result template
        return render_template('result.html', 
                               predicted_class=predicted_class_name,
                               confidence=confidence*100,
                               image_filename=file.filename)

    except ValueError as e:
        return jsonify({
            'error': 'Model input shape mismatch. Ensure the model supports (150, 150, 3) input images.',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
