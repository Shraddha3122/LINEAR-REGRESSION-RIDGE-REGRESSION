# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('D:/WebiSoftTech/LINEAR REGRESSION & RIDGE REGRESSION/Boston_House/boston_houses.csv')

# Select only 'RM' and 'Price' columns
data = data[['RM', 'Price']]

# Prepare the Data
# Split the data into features and target variable
X = data[['RM']]  # Features
y = data['Price']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Ridge Regression Model
# Create a Ridge regression model
ridge_model = Ridge(alpha=1.0)  # You can adjust alpha as needed

# Fit the model on the training data
ridge_model.fit(X_train, y_train)

# Create a Flask Application
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the number of rooms from the request
    data = request.get_json()
    rooms = data['rooms']
    
    # Predict the price using the fitted model
    predicted_price = ridge_model.predict([[rooms]])
    
    # Return the predicted price as a JSON response
    return jsonify({'predicted_price': predicted_price[0]})

# Run the Flask Application
if __name__ == '__main__':
    app.run(debug=True)