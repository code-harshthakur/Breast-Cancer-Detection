from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Initialize the Flask application
app = Flask(__name__)

# Initialize Standard Scaler
scaler = StandardScaler()

# Generate synthetic training data
X_train = np.random.rand(100, 2)
y_train = np.random.randint(2, size=100)

# Train the scaler
scaler.fit(X_train)

# Initialize and compile a Keras model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        
        # Create a feature array and scale it
        features = np.array([[feature1, feature2]])
        features_scaled = scaler.transform(features)
        
        # Make a prediction
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction = 'Benign' if prediction_prob < 0.5 else 'Malignant'

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
