# -*- coding: utf-8 -*-
"""
Created on Mon May 12 20:41:28 2025

@author: 810624TJ
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel


def scale_features(df, target_col='Label', test_size=0.3, random_state=42):
    '''
    Scales the numerical features in the input DataFrame using the RobustScaler, which centers 
    the data around the median and scales it based on the interquartile range (IQR). This approach 
    is particularly effective for datasets containing outliers, as it reduces their influence on the 
    scaling process.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing both features and the target column.
    target_col : str, optional
        The name of the target column (default is 'Label').
    test_size : float, optional
        The proportion of the dataset to include in the test split (default is 0.3).
    random_state : int, optional
        The seed used by the random number generator for reproducibility (default is 42).

    Returns
    -------
    X_train_scaled : pandas.DataFrame
        The scaled training features, with binary columns unscaled.
    X_test_scaled : pandas.DataFrame
        The scaled test features, with binary columns unscaled.
    y_train : pandas.Series
        The target labels for the training set.
    y_test : pandas.Series
        The target labels for the test set.

    Notes
    -----
    - Only numerical columns are scaled, while binary columns remain unchanged.
    - The scaling is based on the training set to avoid data leakage.
    - The function preserves the original binary features.

    Example
    -------
    >>> X_train_scaled, X_test_scaled, y_train, y_test = scale_features(df)
    >>> print(X_train_scaled.head())
    >>> print(y_train.head())
    '''
    
    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify binary and numerical columns
    binary_cols = X.columns[X.nunique() == 2].tolist()
    numeric_cols = [col for col in X.columns if col not in binary_cols]
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Apply scaling only to numerical features
    scaler = RobustScaler()
    X_train_scaled_numeric = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled_numeric = scaler.transform(X_test[numeric_cols])
    
    # Reconstruct scaled DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled_numeric, columns=numeric_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled_numeric, columns=numeric_cols, index=X_test.index)
    
    # Add the original binary columns (not scaled)
    X_train_scaled[binary_cols] = X_train[binary_cols]
    X_test_scaled[binary_cols] = X_test[binary_cols]
    
    return X_train_scaled, X_test_scaled, y_train, y_test






def feature_selection(X_train, y_train, n_estimators=150, min_samples_leaf=4, random_state=42):
    '''
    Performs feature selection to simplify the SVM model while maintaining accuracy and interpretability.
    This approach uses a hybrid, literature-validated method that combines multiple complementary steps:

    1. Feature Importance Estimation with Random Forest

       Trains a Random Forest model on the scaled data. This ensemble algorithm (based on multiple decision 
       trees) allows us to estimate the importance of each feature. The importance is calculated based on 
       the average reduction in Gini impurity for each feature across all the trees in the forest. Features 
       that contribute significantly to reducing impurity are considered more important.

       - Each tree is trained on a bootstrap sample of the dataset (a randomly selected subset with replacement).
       - Splits at each node are made using random subsets of features, contributing to the final prediction.
       - Feature importance is the average impurity reduction gained by splitting on that feature across all trees.

    2. Prefiltering with SelectFromModel

       Uses the SelectFromModel class from scikit-learn to remove less significant features based on the 
       importance scores calculated by the Random Forest. This prefiltering step reduces the dimensionality 
       of the dataset, typically eliminating around half of the features, thereby improving efficiency and 
       reducing the risk of overfitting. Only features with an importance score above the median are retained.

    Parameters
    ----------
    X_train : pandas.DataFrame
        The scaled training feature set.
    y_train : pandas.Series
        The target labels for the training set.
    n_estimators : int, optional
        The number of trees in the Random Forest (default is 150).
    min_samples_leaf : int, optional
        The minimum number of samples required to be at a leaf node (default is 4).
    random_state : int, optional
        The seed used by the random number generator for reproducibility (default is 42).

    Returns
    -------
    model : RandomForestClassifier
        The trained Random Forest model with the feature importance scores.
    selector : SelectFromModel
        The SelectFromModel object used to filter the features.

    Notes
    -----
    - The Random Forest is trained with balanced class weights to account for potential class imbalances.
    - The minimum leaf size is set to 4 to prevent overly aggressive splitting on less informative features.
    - The feature selection step retains only the most relevant features based on the importance scores.

    Example
    -------
    >>> X_train_selected, y_train_selected, model, selector = feature_selection(X_train_scaled, y_train)
    >>> print(X_train_selected.shape)
    >>> print(selector.get_support())
    '''
    
    # Train a Random Forest with 150 trees
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        verbose=2
    )
    
    # Fit the model to the training data
    rf.fit(X_train, y_train)
    
    # Select features based on importance scores
    selector = SelectFromModel(rf, threshold='median', prefit=True)
    X_train_selected = selector.transform(X_train)
    
    return X_train_selected, y_train, rf, selector





def select_important_features(rf, X_train_scaled, X_test_scaled, threshold='median', top_n=60, prefit=True):
    '''
    Selects the most important features from a scaled training set using a pre-trained Random Forest model. 
    The function filters features based on importance scores and provides a visual summary of the selected features.

    Parameters
    ----------
    rf : RandomForestClassifier
        The pre-trained Random Forest model.
    X_train_scaled : pandas.DataFrame
        The scaled training feature set.
    X_test_scaled : pandas.DataFrame
        The scaled test feature set.
    threshold : str or float, optional
        The importance threshold for feature selection. Default is 'median', which selects the top 50% of features.
    top_n : int, optional
        The number of top features to display in the importance plot (default is 60).
    prefit : bool, optional
        If True, assumes the model is already trained (default is True).

    Returns
    -------
    X_train_selected : numpy.ndarray
        The filtered training feature set, reduced to the most important features.
    X_test_selected : numpy.ndarray
        The filtered test feature set, reduced to the most important features.
    selected_features : list
        The list of feature names that were selected.

    Notes
    -----
    - The function uses SelectFromModel to filter features based on importance scores.
    - If prefit=True, the function assumes the Random Forest model has already been trained.
    - The importance scores are visualized in a bar plot to highlight the most influential features.

    Example
    -------
    >>> X_train_selected, X_test_selected, selected_features = select_important_features(rf, X_train_scaled, X_test_scaled)
    >>> print(f"Selected {len(selected_features)} features:")
    >>> print(selected_features)
    '''
    
    # Calculate raw importance scores
    raw_importances = rf.feature_importances_
    perc_importances = 100 * raw_importances / np.sum(raw_importances)

    # Build the importance DataFrame
    feature_imp = pd.DataFrame({
        'feature': X_train_scaled.columns,
        'importance': raw_importances,
        'importance_pct': perc_importances
    }).sort_values('importance', ascending=False)
    
    # Add a formatted percentage column for readability
    feature_imp['importance_%'] = feature_imp['importance_pct'].round(2).astype(str) + '%'
    
    # Plot the top N important features
    plt.figure(figsize=(12, 6))
    plt.title("Top {} Feature Importances (Random Forest)".format(top_n))
    plt.bar(x=feature_imp['feature'].head(top_n), height=feature_imp['importance_pct'].head(top_n))
    plt.ylabel('Importance (%)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
    # Select features based on the threshold
    sfm = SelectFromModel(rf, threshold=threshold, prefit=prefit)
    X_train_selected = sfm.transform(X_train_scaled)
    X_test_selected = sfm.transform(X_test_scaled)

    # Get the names of the selected features
    selected_features = X_train_scaled.columns[sfm.get_support()].tolist()

    # Print the reduction in features
    print(f"Features reduced from {X_train_scaled.shape[1]} to {X_train_selected.shape[1]} (above the median).")

    return X_train_selected, X_test_selected, selected_features


