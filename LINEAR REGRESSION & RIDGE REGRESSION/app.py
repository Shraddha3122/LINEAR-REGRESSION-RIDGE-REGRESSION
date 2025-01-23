from flask import Flask, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

app = Flask(__name__)

# Load training and testing data
train_data = pd.read_csv('D:/WebiSoftTech/LINEAR REGRESSION & RIDGE REGRESSION/test.csv')
try:
    test_data = pd.read_csv('D:/WebiSoftTech/LINEAR REGRESSION & RIDGE REGRESSION/test.csv')
except FileNotFoundError:
    test_data = None

# Ensure data is in the expected format
if not {'Weight', 'Height'}.issubset(train_data.columns):
    raise ValueError("Train data must contain 'Weight' and 'Height' columns.")
if test_data is not None and not {'Weight', 'Height'}.issubset(test_data.columns):
    raise ValueError("Test data must contain 'Weight' and 'Height' columns.")

# Prepare training data
X_train = train_data[['Weight']]
y_train = train_data['Height']

# Fit Simple Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

if test_data is not None:
    # Prepare testing data
    X_test = test_data[['Weight']]
    y_test = test_data['Height']

    # Fit Ridge Regression
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_test, y_test)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict height based on weight."""
    data = request.json
    weight = data.get('weight')
    if weight is None:
        return jsonify({"error": "Weight is required."}), 400

    # Predict using Simple Linear Regression
    prediction = linear_model.predict([[weight]])[0]
    return jsonify({"weight": weight, "predicted_height": prediction})

if __name__ == '__main__':
    app.run(debug=True)
