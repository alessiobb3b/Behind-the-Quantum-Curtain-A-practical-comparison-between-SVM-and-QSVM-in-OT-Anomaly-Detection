# -*- coding: utf-8 -*-
"""
Created on Mon May 12 21:18:13 2025

@author: 810624TJ
"""

from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def train_classical_svc(X_train_prefiltered, X_test_prefiltered, y_train, y_test, num_features=4, train_size=5000, test_size=2500, C=10):
    '''
    Trains a classical Support Vector Classifier (SVC) using the scikit-learn library. The function 
    uses the same feature set, training size, and regularization parameter as the quantum SVC to 
    provide a fair comparison.

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
        The number of top features to use for classification (default is 4).
    train_size : int, optional
        The number of samples to use for training the SVC (default is 5000).
    test_size : int, optional
        The number of samples to use for testing the SVC (default is 2500).
    C : float, optional
        The regularization parameter for the SVC (default is 10).

    Returns
    -------
    svc : SVC
        The trained classical Support Vector Classifier.
    y_pred_svc : numpy.ndarray
        The predicted labels for the test set.
    misclassified : numpy.ndarray
        The misclassified samples from the test set.

    Notes
    -----
    - Uses a linear kernel by default for a direct comparison with quantum SVM models.
    - The function includes a confusion matrix plot for the test set predictions.
    - No hyperparameter tuning (e.g., grid search) is applied to match the quantum SVC setup.

    Example
    -------
    >>> svc, y_pred_svc, misclassified = train_classical_svc(X_train_prefiltered, X_test_prefiltered, y_train, y_test)
    >>> print(f"Number of misclassified samples: {len(misclassified)}")
    '''
    
    # Select the top N features
    X_train_c = X_train_prefiltered[:, :num_features]
    X_test_c = X_test_prefiltered[:, :num_features]

    # Use a smaller subset for training
    X_train_c_small = X_train_c[:train_size]
    y_train_c_small = y_train[:train_size]
    X_test_c_small = X_test_c[:test_size]
    y_test_c_small = y_test[:test_size]

    # Train the SVC model
    svc = SVC(kernel='linear', C=C)
    svc.fit(X_train_c_small, y_train_c_small)

    # Make predictions
    y_pred_svc = svc.predict(X_test_c_small)

    # Print the classification report
    print("\nClassification Report (Classical SVC):")
    print(classification_report(y_test_c_small, y_pred_svc))

    # Identify misclassified samples
    misclassified = X_test_c_small[y_pred_svc != y_test_c_small]
    print(f"\nNumber of misclassified samples: {len(misclassified)}")

    # Plot the confusion matrix
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_test_c_small, y_pred_svc, cmap='Blues')
    plt.title("Confusion Matrix (Classical SVC)")
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return svc, y_pred_svc, misclassified

