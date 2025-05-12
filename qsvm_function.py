# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:01:18 2025

@author: 810624TJ
"""


from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_aer import AerSimulator
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt



def train_quantum_svc(X_train_prefiltered, X_test_prefiltered, y_train, y_test, 
                      num_features=4, 
                      train_size=5000, 
                      test_size=2500, 
                      C=10):
    '''
    Trains a Quantum Support Vector Classifier (QSVC) using the Qiskit Machine Learning library. The function 
    selects the top N most important features, creates a quantum feature map, and fits the QSVC model to a 
    smaller training set for computational efficiency.

    Parameters
    ----------
    X_train_prefiltered : numpy.ndarray
        The filtered training feature set, reduced to the most important features.
    X_test_prefiltered : numpy.ndarray
        The filtered test feature set, reduced to the most important features.
    y_train : numpy.ndarray
        The target labels for the training set.
    y_test : numpy.ndarray
        The target labels for the test set.
    num_features : int, optional
        The number of top features to use for quantum classification (default is 4).
    train_size : int, optional
        The number of samples to use for training the QSVC (default is 5000).
    test_size : int, optional
        The number of samples to use for testing the QSVC (default is 2500).
    C : float, optional
        The regularization parameter for the QSVC (default is 10).

    Returns
    -------
    qsvc : QSVC
        The trained Quantum Support Vector Classifier.
    y_pred_q : numpy.ndarray
        The predicted labels for the test set.
    misclassified : numpy.ndarray
        The misclassified samples from the test set.

    Notes
    -----
    - Uses a linear entanglement feature map (ZZFeatureMap) with a single repetition.
    - The AerSimulator is configured to utilize up to 31 parallel threads for faster simulation.
    - Includes a confusion matrix plot for the test set predictions.

    Example
    -------
    >>> qsvc, y_pred_q, misclassified = train_quantum_svc(X_train_prefiltered, X_test_prefiltered, y_train, y_test)
    >>> print(f"Number of misclassified samples: {len(misclassified)}")
    '''
    
    # Select the top N features
    X_train_q = X_train_prefiltered[:, :num_features]
    X_test_q = X_test_prefiltered[:, :num_features]

    # Use a smaller subset for quantum training
    X_train_q_small = X_train_q[:train_size]
    y_train_q_small = y_train[:train_size]
    X_test_q_small = X_test_q[:test_size]
    y_test_q_small = y_test[:test_size]

    # Create the feature map and quantum kernel
    simulator = AerSimulator(max_parallel_threads=31)
    feature_map = ZZFeatureMap(feature_dimension=num_features, reps=1, entanglement='linear')
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map, quantum_instance=simulator)

    # Train the QSVC model
    qsvc = QSVC(quantum_kernel=quantum_kernel, C=C)
    qsvc.fit(X_train_q_small, y_train_q_small)

    # Make predictions
    y_pred_q = qsvc.predict(X_test_q_small)

    # Print the classification report
    print("\nClassification Report:")
    print(classification_report(y_test_q_small, y_pred_q))

    # Identify misclassified samples
    misclassified = X_test_q_small[y_pred_q != y_test_q_small]
    print(f"\nNumber of misclassified samples: {len(misclassified)}")

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test_q_small, y_pred_q, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return qsvc, y_pred_q, misclassified

