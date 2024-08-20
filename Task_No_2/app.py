# Import Flask and load the model
from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

# Create a Flask app
app = Flask(__name__)

# Load the saved model
model = load_model('C:\\Users\\Kasheef_Alee\\Desktop\\Task_No_2\\lstm_model.h5')

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    stock_data = request.form['stock_data']

    try:
        # Fetch the last 60 days of stock data
        data = yf.download(stock_data, start='2023-06-01', end='2024-01-01')
        if data.empty:
            return render_template('index.html', prediction=None, error="No data found for ticker.")

        close_prices = data['Close'].values.reshape(-1, 1)

        # Preprocess the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)
        test_data = scaled_data[-60:]
        test_data = np.reshape(test_data, (1, test_data.shape[0], 1))

        # Make the prediction
        prediction = model.predict(test_data)
        prediction = scaler.inverse_transform(prediction)

        # Return the prediction to the user
        return render_template('index.html', prediction=prediction[0][0], error=None)
    
    except Exception as e:
        return render_template('index.html', prediction=None, error=str(e))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
