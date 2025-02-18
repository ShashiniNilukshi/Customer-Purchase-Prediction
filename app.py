from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('Customer_Purchase_Prediction_Model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

purchase_label = {'Yes': 1, 'No': 0}
marital_status_label = {'Married': 0, 'Single': 1}
education_level_label = {'Bachelor': 0, 'Master': 1, 'PhD': 2}
gender_label = {'Female': 0, 'Male': 1}
customer_segment_label = {'High-value': 0, 'Low-value': 1}

@app.route('/')
def home():
    return "Welcome to Customer Purchase Prediction API"

@app.route('/prediction', methods=['POST'])
def predict():
    data = request.get_json()

    # Normalize key names to accept both lowercase and capitalized versions
    purchase = data.get('purchase') or data.get('Purchase')
    marital_status = data.get('marital_status') or data.get('Marital_Status')
    education_level = data.get('education_level') or data.get('Education_Level')
    gender = data.get('gender') or data.get('Gender')
    customer_segment = data.get('customer_segment') or data.get('Customer_Segment')
    income = data.get('Income') or data.get('income')
    age = data.get('Age') or data.get('age')

    # Validate missing data
    if None in (purchase, marital_status, education_level, gender, customer_segment, income, age):
        return jsonify({'error': 'Missing input data'}), 400

    # Validate that Income and Age are numeric
    try:
        income = float(income)  # Convert to float
        age = int(age)  # Convert to integer
    except (ValueError, TypeError):
        return jsonify({'error': 'Income and Age must be valid numbers'}), 400

    # Validate categorical inputs
    valid_values = {
        "purchase": list(purchase_label.keys()),
        "marital_status": list(marital_status_label.keys()),
        "education_level": list(education_level_label.keys()),
        "gender": list(gender_label.keys()),
        "customer_segment": list(customer_segment_label.keys())
    }

    invalid_fields = []
    for key, valid_options in valid_values.items():
        if data.get(key) not in valid_options:
            invalid_fields.append(f"{key} must be one of {valid_options}")

    if invalid_fields:
        return jsonify({'error': 'Invalid categorical data', 'details': invalid_fields}), 400

    # Encode categorical features
    purchase_encoded = purchase_label.get(purchase)
    marital_status_encoded = marital_status_label.get(marital_status)
    education_level_encoded = education_level_label.get(education_level)
    gender_encoded = gender_label.get(gender)
    customer_segment_encoded = customer_segment_label.get(customer_segment)

    # Prepare input data
    input_data = np.array([[purchase_encoded, marital_status_encoded, education_level_encoded,
                            gender_encoded, customer_segment_encoded, income, age]])

    # Make prediction
    prediction = model.predict(input_data)

    return jsonify({'Purchase': prediction[0]})  # Ensure JSON serializable output

if __name__ == '__main__':
    app.run(debug=True)
