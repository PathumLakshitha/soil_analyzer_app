from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import os
import pandas as pd

app = Flask(__name__)

# Load YOLOv8 model
model_path = "model/Tea_soil_research_YOLOv8n_best.pt"
model = YOLO(model_path)

# Load Excel file for bramble mix ratios
excel_path = "soil_bramble_mix_ratio_data.xlsx"
df = pd.read_excel(excel_path)

# Function to calculate the average bramble mix ratio
def get_bramble_mix_ratio(soil_type):
    averages = df.groupby("Soil Type")["Bramble Mix Ratio (%)"].mean()
    return averages.get(soil_type, "Undefined")

# pH Adjustment Function
def adjust_soil_ph(soil_type, current_ph):
    target_ph_min = 4.5
    target_ph_max = 5.5
    invalid_ph_min = 4.4  # Below this is invalid
    soil_volume_area = 0.85  # Square meters
    soil_height = 15  # Centimeters

    if current_ph < invalid_ph_min:
        return None, None, f"The entered pH value {current_ph} is invalid. Please enter a value of 4.4 or higher."

    if target_ph_min <= current_ph <= target_ph_max:
        return f"The soil pH is already suitable for tea plantation.", None, None

    if current_ph > target_ph_max:
        sulfur_instruction = (f"Mix 25-85 grams of sulfur powder per {soil_volume_area} square meter of soil "
                              f"thinned to a height of {soil_height} centimeters. "
                              "This soil should be used to fill nursery bags after three months.")

        aluminum_instruction = (f"Mix 225 grams of aluminum sulfate per {soil_volume_area} square meter of soil "
                                f"thinned to a height of {soil_height} centimeters. "
                                "This soil can be used to fill nursery bags after a week.")
        return sulfur_instruction, aluminum_instruction, None

    return f"The pH for {soil_type} is below the target range; no amendments are needed.", None, None

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Image file is required"}), 400

    file = request.files['file']
    pH_value = request.form.get('pH_value', None)  # pH value is optional

    # Save uploaded image
    upload_folder = "static/uploads"
    os.makedirs(upload_folder, exist_ok=True)
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)

    # Make a prediction using the YOLO model
    results = model.predict(file_path, conf=0.5)
    
    # Extracting prediction probabilities and classes
    predictions = results[0].names  # Extract soil type names from prediction results
    prediction_probs = results[0].probs  # Extract prediction probabilities
    
    # Print predictions and probabilities
    print("Predictions:", predictions)
    print("Prediction Probabilities:", prediction_probs)

    # Get the index of the highest prediction probability
    max_prob_idx = prediction_probs.top1
    confidence_score = prediction_probs.top1conf.item()

    soil_type = predictions[max_prob_idx] if predictions else "Unknown"

    # Get the bramble mix ratio
    bramble_mix_ratio = get_bramble_mix_ratio(soil_type)

    # Check if pH value is provided
    pH_adjustment_sulfur = None
    pH_adjustment_aluminum = None
    error_message = None
    if pH_value:
        try:
            pH_value = float(pH_value)  # Convert pH value to float
            pH_adjustment_sulfur, pH_adjustment_aluminum, error_message = adjust_soil_ph(soil_type, pH_value)
        except ValueError:
            return jsonify({"error": "Invalid pH value format"}), 400

    if error_message:
        return jsonify({"error": error_message}), 400

    return jsonify({
        "soil_type": soil_type,
        "confidence_score": confidence_score,
        "bramble_mix_ratio": bramble_mix_ratio,
        "pH_adjustment_sulfur": pH_adjustment_sulfur,
        "pH_adjustment_aluminum": pH_adjustment_aluminum,
        "file_path": file_path
    })


if __name__ == '__main__':
    app.run(debug=True)
