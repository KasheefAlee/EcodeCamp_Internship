from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
loaded_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    pclass = int(request.form['pclass'])
    sex = request.form['sex'].lower()
    age = int(request.form['age'])
    sibsp = int(request.form['sibsp'])
    fare = float(request.form['fare'])
    embarked = request.form['embarked'].lower()

    print("Received Input Data:")  # Debugging
    print(f"pclass: {pclass}, sex: {sex}, age: {age}, sibsp: {sibsp}, fare: {fare}, embarked: {embarked}")

    # Convert to numeric values for prediction
    sex_numeric = 0 if sex == 'female' else 1
    embarked_numeric = {'cherbourg': 0, 'queenstown': 1, 'southampton': 2}.get(embarked, -1)

    # Ensure valid inputs
    if embarked_numeric == -1:
        return "Invalid embarkation port input. Please enter 'Cherbourg', 'Queenstown', or 'Southampton'."

    # Prepare data for the model
    input_array = np.array([pclass, sex_numeric, age, sibsp, fare, embarked_numeric]).reshape(1, -1)

    # Make prediction
    prediction = loaded_model.predict(input_array)

    result = "Prediction: Survived" if prediction[0] == 1 else "Prediction: Did not survive"

    # Prepare input data for display
    input_data = {
        'pclass': pclass,
        'sex': 'Female' if sex_numeric == 0 else 'Male',
        'age': age,
        'sibsp': sibsp,
        'fare': fare,
        'embarked': 'Cherbourg' if embarked_numeric == 0 else 'Queenstown' if embarked_numeric == 1 else 'Southampton'
    }

    return render_template('result.html', prediction_text=result, input_data=input_data)

if __name__ == "__main__":
    app.run(debug=True)