# Importing required libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense

# Load and preprocess the Iris dataset
def load_and_preprocess_data():
    iris = load_iris()
    X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
    y = iris.target.reshape(-1, 1)  # Labels (Iris species, reshaped to 2D)

    # One-Hot Encode the target labels
    encoder = OneHotEncoder(sparse_output=False)
    y_encoded = encoder.fit_transform(y)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler, encoder  # Return scaler and encoder for future use

# Build the neural network model
def build_model(input_shape, output_units):
    model = Sequential()
    
    # Define the input layer and hidden layers
    model.add(Input(shape=input_shape))
    model.add(Dense(units=64, activation='relu'))  # First hidden layer
    model.add(Dense(units=64, activation='relu'))  # Second hidden layer

    # Output layer (number of units = number of classes, softmax activation)
    model.add(Dense(units=output_units, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Main function to load data, build model, train, and evaluate
def main():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, scaler, encoder = load_and_preprocess_data()

    # Build model
    model = build_model(input_shape=(X_train.shape[1],), output_units=y_train.shape[1])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=2)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test loss: {loss:.4f}')
    print(f'Test accuracy: {accuracy:.4f}')

    # Return necessary elements for future use
    return model, scaler, encoder

# Predict on new data
def predict_new_data(model, scaler, encoder, new_data):
    # Standardize the new data using the same scaler fitted on the training data
    new_data_scaled = scaler.transform(new_data)

    # Make predictions
    predictions = model.predict(new_data_scaled)

    # Convert predictions from one-hot encoded format back to class labels
    predicted_classes = encoder.inverse_transform(predictions)

    print(f'Predictions (one-hot encoded):\n{predictions}')
    print(f'Predicted class labels:\n{predicted_classes}')

# Execute the main function
if __name__ == "__main__":
    model, scaler, encoder = main()

    # Example new data points for prediction
    new_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example data

    # Predict on new data
    predict_new_data(model, scaler, encoder, new_data)
