from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('../models/energy_consumption_model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract the input features
    solar_power = data['solarPower']
    solar_energy = data['solarEnergy']
    hour = data['hour']
    day = data['day']
    month = data['month']
    year = data['year']

    # Prepare the input array (features must be in the same order used during training)
    input_features = np.array([[solar_power, solar_energy, hour, day, month, year]])

    # Make the prediction using the loaded model
    prediction = model.predict(input_features)

    # Return the prediction as a JSON response
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
