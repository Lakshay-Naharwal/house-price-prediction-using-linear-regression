from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import json
import os

app = Flask(__name__)

# Load model and metadata
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model', 'bangalore_house_price_model.pkl')
metadata_path = os.path.join(script_dir, 'model', 'metadata.pkl')
locations_path = os.path.join(script_dir, 'model', 'locations.json')

if os.path.exists(model_path) and os.path.exists(metadata_path):
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    all_columns = metadata['columns']
else:
    model = None
    all_columns = []

@app.route('/')
def index():
    if os.path.exists(locations_path):
        with open(locations_path, 'r') as f:
            locations = json.load(f)
    else:
        locations = []
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found'}), 500
    
    try:
        data = request.get_json()
        location = data.get('location')
        sqft = float(data.get('total_sqft', 0))
        bath = int(data.get('bath', 0))
        bhk = int(data.get('bhk', 0))

        # Prepare input vector
        x = np.zeros(len(all_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        
        try:
            loc_index = all_columns.index(location)
            x[loc_index] = 1
        except (ValueError, KeyError):
            pass # 'other' or unknown location
        
        # Predict
        prediction = model.predict([x])[0]
        
        return jsonify({
            'price': round(float(prediction), 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
