from flask import Flask, render_template, request,url_for
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
model = tf.keras.models.load_model('my_model.h5')
import numpy as np
from tensorflow.keras.preprocessing import image

def load_and_preprocess_image(img_path, target_size=(150, 150)):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

def get_disease_details(class_name):
    """Fetch detailed information about the disease."""
    disease_info = {
        '0': {"name": "No Disease", "description": "The eye appears healthy with no signs of disease."},
        '1': {"name": "Diabetic Retinopathy", "description": "Damage to the retina caused by high blood sugar levels."},
        '2': {"name": "Cataract", "description": "Clouding of the lens, leading to blurred or impaired vision."},
        '3': {"name": "Glaucoma", "description": "A group of eye conditions damaging the optic nerve, leading to vision loss."},
        '4': {"name": "Other Eye Conditions", "description": "Unclassified conditions requiring further analysis."}
    }
    return disease_info.get(class_name, {"name": "Unknown", "description": "No information available."})

def predict_disease(model, img_path, class_labels):
    """Predict disease for a given image and return detailed information."""
    # Preprocess the image
    img_array = load_and_preprocess_image(img_path)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_index]
    confidence = predictions[0][predicted_index] * 100  # Confidence in percentage

    # Fetch disease details
    disease_details = get_disease_details(predicted_class)
    disease_details["confidence"] = f"{confidence:.2f}%"
    disease_details["image"] = img_path

    return disease_details
# def preprocess_image(image):
#     image = image.convert('RGB')  # Ensure it's in RGB mode
#     image = image.resize((150, 150))  # Resize to match model input
#     image = np.array(image)
#     image = image.reshape((1, 150, 150, 3)).astype('float32') / 255.0  # Normalize
#     return image

@app.route('/')
def home():
    return render_template('index.html')
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', result="No file part in the request")

#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', result="No file selected")

#     try:
#         image = Image.open(file)
#         processed_image = preprocess_image(image)
#         prediction = model.predict(processed_image)
#         output = np.argmax(prediction, axis=1)[0]
#         result = f"Prediction: {output}"
#     except Exception as e:
#         result = f"An error occurred: {str(e)}"
    
#     return render_template('index.html', result=result)

# Load the trained model

# Define class labels
class_labels = ['0', '1', '2', '3', '4']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded!"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected!"}), 400

    # Save the uploaded file temporarily
    img_path = f"static/uploads/{file.filename}"
    file.save(img_path)

    # Predict disease
    prediction = predict_disease(model, img_path, class_labels)

    # Return result (can also render a template)
    return render_template('result.html', result=prediction,img_path = img_path)

if __name__ == '__main__':
    app.run(debug=True)