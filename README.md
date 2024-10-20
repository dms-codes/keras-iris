

# Iris Classification with Keras Neural Network

This project demonstrates how to build and train a neural network model using the Keras library to classify the Iris dataset. It includes data preprocessing, model building, training, evaluation, and making predictions on new data.

## Dataset

The model is trained on the Iris dataset, which contains 150 samples of iris flowers. Each sample includes four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The task is to classify these samples into one of three species:
- Setosa
- Versicolor
- Virginica

## Project Structure

- **main.py**: Main script for loading the dataset, building and training the model, evaluating it, and making predictions.
- **README.md**: This documentation file.

## Requirements

- Python 3.x
- Required libraries:
  - `numpy`
  - `scikit-learn`
  - `tensorflow` / `keras`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

**Note**: Create a `requirements.txt` file with the following content:

```txt
numpy
scikit-learn
tensorflow
```

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/keras-iris-classification.git
   cd keras-iris-classification
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the main script:

   ```bash
   python main.py
   ```

4. The model will be trained on the Iris dataset and evaluated on a test set. You will see the test loss and accuracy in the terminal output.

5. The script will also demonstrate how to make predictions on new data. It will print both the one-hot encoded predictions and the predicted class labels.

### Example Prediction

In the script, the following sample data is used for prediction:

```python
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
```

The prediction results will show the predicted class for this input.

## Model Architecture

The model consists of:
- Input layer
- Two hidden layers with 64 neurons each and ReLU activation
- Output layer with softmax activation for classification into three classes

The model is compiled using the Adam optimizer and categorical crossentropy loss.

## Evaluation

After training, the model is evaluated on a test dataset. The test loss and accuracy will be printed.

Example output:
```
Test loss: 0.1245
Test accuracy: 0.9333
```

## Contributing

Feel free to submit issues or pull requests if you'd like to improve the project.

---

