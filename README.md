Step-by-Step Guide: RBFN Classifier with PyTorch on Iris Dataset
1. Introduction
This guide explains how to build a Radial Basis Function Network (RBFN) using PyTorch to classify the well-known Iris dataset. The project includes dataset preparation, preprocessing, model definition, training, and evaluation. The Iris dataset contains 3 classes of 50 instances each, with four features per instance.
2. Setup and Imports
We begin by importing required libraries:

- torch, torch.nn, torch.optim: for building and training the neural network
- pandas: for loading and handling the dataset
- sklearn.model_selection: for splitting the dataset
- sklearn.preprocessing: for scaling features

3. Dataset Preparation
We load the Iris dataset using pandas, factorize the target variable (converting string labels to numeric), and then scale the features:

- Use `pd.read_csv` to load the dataset.
- Use `pd.factorize` to convert string labels into numerical classes.
- Use `StandardScaler` to normalize the features (mean=0, std=1) for better performance with RBF.

4. Splitting the Dataset
Split the data into training and testing sets using `train_test_split`. 70% of the data is used for training and 30% for testing.
5. Convert to PyTorch Tensors
The model requires input in the form of PyTorch tensors. A helper function `to_tensor` is used to convert NumPy arrays to tensors.
6. RBF Kernel and Network Definition

The RBF kernel is computed using the squared Euclidean distance between the input and the centers.

The RBFN model consists of:
- A learnable set of RBF centers (randomly initialized)
- A learnable `beta` parameter controlling the width of the Gaussian kernel
- A linear layer that maps RBF outputs to class scores

7. Training the Model

- Loss function: `CrossEntropyLoss` suitable for multi-class classification
- Optimizer: Adam optimizer with learning rate of 0.01
- Training loop: Runs for 100 epochs, where each epoch performs:
  - Forward pass
  - Loss computation
  - Backward pass
  - Optimizer step

8. Model Evaluation

After training, the model is evaluated on the test data.
- Predictions are obtained using the trained model.
- Accuracy is computed by comparing predicted and actual class labels.

9. Sample Output

Epoch [100/100], Loss: 0.0493
Accuracy: 96.67%

10. Summary

This project provides a simple and educational example of implementing an RBFN using PyTorch. It demonstrates key steps such as data preprocessing, model construction, training, and evaluation. RBFNs are useful for modeling local receptive fields and are especially intuitive for small datasets like Iris.

Requirements
- torch
- pandas
- scikit-learn
Execution
Ensure `iris.data` is in your directory, then run:
python main.py

