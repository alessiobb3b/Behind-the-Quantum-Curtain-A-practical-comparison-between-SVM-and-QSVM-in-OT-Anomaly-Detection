# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:45:55 2025

@author: 810624TJ
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_data(PATH_RAWDATA):

    '''
    Imports and concatenates CSV files present in a specified directory, excluding those with 
    significantly imbalanced classes and a specific problematic file. The function excludes files 
    with an attack/benign ratio less than 1/3, considering this as a sign of excessive imbalance.
    
    Parameters
    ----------
    PATH_RAWDATA : str
        The path to the directory containing the CSV files to be imported. This should be a string
        specifying the absolute or relative path to the current working directory.

    Returns
    -------
    final_df : pandas.DataFrame
        The resulting DataFrame from concatenating all valid CSV files. Column names are normalized 
        to remove any leading or trailing whitespace. If no files are included, the function returns 
        an empty DataFrame.

    Notes
    ----
    - Checks that each file contains a 'Label' (or ' Label' with leading space) column for class identification.
    - Prints information about included and excluded files for tracking.
    - In case of a file reading error, the file is skipped, and the error is printed.

    Example
    -------
    >>> PATH_RAWDATA = './TrafficLabelling'
    >>> final_df = import_data(PATH_RAWDATA)
    >>> print(final_df.head())
    '''
       
    list_csv = []
    for FILE in os.listdir(PATH_RAWDATA):
        # Exclude the problematic file
        if FILE.endswith(".csv") and FILE != "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv":  
            list_csv.append(os.path.join(PATH_RAWDATA, FILE))
    list_df = []

    for csv_file in list_csv:
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            df.columns = df.columns.str.strip()

            label_col = 'Label' if 'Label' in df.columns else ' Label'
            if label_col in df.columns:
                label_counts = df[label_col].value_counts()
                benign_count = label_counts.get('BENIGN', 0)
                attack_count = label_counts.sum() - benign_count

                print(f"\n {os.path.basename(csv_file)}:")
                print(label_counts)

                if benign_count > 0 and (attack_count / benign_count) < (1/3):
                    print("Excluded (too imbalanced relative to BENIGN)")
                    continue  # salta questo file
                    
                print("Included in the dataset")
            else:
                print(f"\n 'Label' column not found in {os.path.basename(csv_file)}")

            list_df.append(df)

        except Exception as e:
            print(f" Error in {csv_file}: {e}")

    # Concatenazione dei file validi
    final_df = pd.concat(list_df, ignore_index=True)
    final_df.columns = final_df.columns.str.strip()
    return final_df




def clean_dataset(df):
    '''
    Cleans a DataFrame by removing columns containing only zeros, rows with missing values (NaN), 
    and rows with infinite values (inf, -inf). Prints a summary of the modifications made to the dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be cleaned.

    Returns
    -------
    df : pandas.DataFrame
        The cleaned DataFrame, without irrelevant columns or problematic values.

    Note
    ----
    - Removes columns that contain only zeros.
    - Eliminates rows containing NaN and infinite values.
    - Replaces inf and -inf with NaN before removing rows.
    - Prints a summary of the removed columns and rows for tracking.

     Example
    -------
    >>> df = pd.read_csv("dataset.csv")
    >>> df_cleaned = clean_dataset(df)
    >>> print(df_cleaned.head())
    '''
    # Print initial dataset size and feature names
    print("Initial dataset size:", df.shape)
    print("\nFeature Names:", list(df.columns))
    
    # Initialize flags for problematic columns
    flag_all_zero_cols = []
    flag_null_counts = {}
    inf_count = {}

    # 1. Identify columns with only zeros
    for col in df.columns:
        unique_vals = df[col].unique().tolist()
        if set(unique_vals) == [0] or unique_vals == [0]:
            flag_all_zero_cols.append(col)

    # 2. Identify columns with missing values (NaN)
    flag_null_counts = df.isna().sum()
    flag_null_counts = {col: count for col, count in flag_null_counts.items() if count > 0}

    # 3. Identify columns with inf or -inf values (only numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        count_inf = np.isinf(df[col]).sum()
        if count_inf > 0:
            inf_count[col] = count_inf

    # Remove columns with only zeros
    if flag_all_zero_cols:
        print("\nThe following columns contain only zeros and have been removed:")
        for col in flag_all_zero_cols:
            print(f" - {col}")
        df.drop(columns=flag_all_zero_cols, inplace=True)
    else:
        print("\nNo columns contain only zeros.")

    # Remove rows with NaN values
    if flag_null_counts:
        print("\nColumns with NaN values:")
        for col, count in flag_null_counts.items():
            print(f" - {col}: {count} NaN values")
        righe_nan = df.isna().sum().sum()
        df.dropna(inplace=True)
        print(f"Removed {righe_nan} NaN cells and their corresponding rows.")
    else:
        print("\nNo columns contain NaN values.")

    # Replace inf and -inf with NaN and remove rows
    if inf_count:
        print("\nColumns with inf or -inf values:")
        for col, count in inf_count.items():
            print(f" - {col}: {count} inf values")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        righe_nan = df.isna().sum().sum()
        df.dropna(inplace=True)
        print(f"Removed {righe_nan} inf and -inf cells and their corresponding rows.")
    else:
        print("\nNo columns contain inf or -inf values.")

    # Print final dataset size
    print("\nNew dataset size:", df.shape)

    return df


def transform_ip_to_frequencies(df):
    '''
    Transforms the input DataFrame by adding columns representing the frequency of occurrence for 
    source and destination IP addresses per minute. This function normalizes timestamps and groups 
    data into one-minute intervals, simplifying the temporal analysis of network flows.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame must contain at least the columns 'Timestamp', 'Source IP', and 'Destination IP'.

    Returns
    -------
    df : pandas.DataFrame
        The transformed DataFrame, with the original columns 'Minute', 'Source IP', and 'Destination IP' removed.
        The DataFrame is indexed by 'Timestamp'.

    Notes
    -----
    - Timestamps are converted to datetime format and sorted chronologically.
    - Rows with invalid timestamps (NaT) are removed.
    - IP address counts are calculated for one-minute intervals.
    - The original 'Minute', 'Source IP', and 'Destination IP' columns are dropped after processing.

    Example
    -------
    >>> df_transformed = transform_ip_to_frequencies(df)
    >>> print(df_transformed.head())
    '''
    
    df = df.copy()

    # Convert the 'Timestamp' column to datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed', dayfirst=True, errors='coerce')
    # Remove rows with invalid timestamps (NaT)
    df.dropna(subset=['Timestamp'], inplace=True)
    # Sort the DataFrame chronologically by 'Timestamp'
    df.sort_values('Timestamp', inplace=True)

    # Create a "Minute" column for time grouping
    df["Minute"] = df["Timestamp"].dt.floor("min")

    # Calculate source IP frequency per minute
    df_source_counts = (
        df.groupby(["Source IP", "Minute"])
          .size()
          .rename("Source_ip_count")
          .reset_index()
    )

    # Calculate destination IP frequency per minute
    df_destination_counts = (
        df.groupby(["Destination IP", "Minute"])
          .size()
          .rename("Destination_ip_count")
          .reset_index()
    )

    # Merge the calculated frequencies
    # Source IP
    df = df.merge(df_source_counts, on=["Source IP", "Minute"], how="left")
    # Destination IP
    df = df.merge(df_destination_counts, on=["Destination IP", "Minute"], how="left")
    # Drop original columns
    df.drop(columns=["Minute", "Source IP", "Destination IP"], inplace=True)
    # Set the 'Timestamp' column as the index
    df.set_index("Timestamp", inplace=True)

    return df


def drop_flow_id(df):
    '''
    Removes the 'Flow ID' column from the input DataFrame. This is typically used to eliminate 
    unique identifiers that are not relevant for machine learning models or statistical analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame from which the 'Flow ID' column should be removed.

    Returns
    -------
    df : pandas.DataFrame
        The modified DataFrame without the 'Flow ID' column.

    Notes
    -----
    - This function assumes the 'Flow ID' column is present in the DataFrame.
    - If the column is not present, a KeyError will be raised.

    Example
    -------
    >>> df_cleaned = drop_flow_id(df)
    >>> print(df_cleaned.head())
    '''
    df.drop(columns=["Flow ID"], inplace=True)
    return df



def categorize_ports(df):
    '''
    Categorizes the source and destination ports in a DataFrame based on their range and applies 
    one-hot encoding to simplify network traffic analysis. Ports are classified into three categories:
    - Well-known (0-1023)
    - Registered (1024-49151)
    - Ephemeral (49152-65535)

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame must contain the columns 'Source Port' and 'Destination Port'.

    Returns
    -------
    df : pandas.DataFrame
        The transformed DataFrame, with the original 'Source Port' and 'Destination Port' columns removed 
        and new categorized columns added.

    Notes
    -----
    - The original 'Source Port' and 'Destination Port' columns are removed after categorization.
    - One-hot encoding is applied to the resulting categories for both source and destination ports.
    - Column names are renamed for improved readability.

    Example
    -------
    >>> df_categorized = categorize_ports(df)
    >>> print(df_categorized.head())
    '''
    
    df = df.copy()

    # Internal function to categorize ports
    def categorize_port(port):
        if port <= 1023:
            return "well_known"
        elif port <= 49151:
            return "registered"
        else:
            return "ephemeral"

    # Apply the internal function
    df['Source_Port_Category'] = df['Source Port'].apply(categorize_port)
    df['Destination_Port_Category'] = df['Destination Port'].apply(categorize_port)
    # Remove the original columns
    df.drop(columns=['Source Port', 'Destination Port'], inplace=True)
    # Apply one-hot encoding to the categorized columns
    df = pd.get_dummies(df, columns=['Source_Port_Category', 'Destination_Port_Category'], dtype=int)
    # Rename columns for better readability
    df.rename(columns={
        'Source_Port_Category_well_known': 'sourceport_wellknown',
        'Source_Port_Category_registered': 'sourceport_registered',
        'Source_Port_Category_ephemeral': 'sourceport_ephemeral',
        'Destination_Port_Category_well_known': 'destinationport_wellknown',
        'Destination_Port_Category_registered': 'destinationport_registered',
        'Destination_Port_Category_ephemeral': 'destinationport_ephemeral'
    }, inplace=True)

    return df



def encode_protocol(df):
    '''
    Applies one-hot encoding to the 'Protocol' column, converting the numerical protocol values 
    into binary columns. This transformation makes the protocol information more explicit for 
    machine learning models.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame must contain the 'Protocol' column.

    Returns
    -------
    df : pandas.DataFrame
        The transformed DataFrame with additional binary columns for each protocol found. 
        The original 'Protocol' column is removed.

    Notes
    -----
    - The new binary columns represent common network protocols:
        - 'protocol_6' -> 'TCP'
        - 'protocol_17' -> 'UDP'
        - 'protocol_0' -> 'HOPOPT'
    - The original 'Protocol' column is automatically removed from the DataFrame.
    - Column names are renamed for better readability.

    Example
    -------
    >>> df_encoded = encode_protocol(df)
    >>> print(df_encoded.head())
    '''
    
    df = df.copy()
    # Apply one-hot encoding to the 'Protocol' column
    df = pd.get_dummies(df, columns=['Protocol'], prefix='protocol', dtype=int)
    # Rename the resulting columns for clarity
    df.rename(columns={
        'protocol_6': 'TCP',
        'protocol_17': 'UDP',
        'protocol_0': 'HOPOPT'
    }, inplace=True)
    return df



def transform_label(df):
    '''
    Transforms the 'Label' column in a DataFrame into a binary format suitable for supervised 
    machine learning models (e.g., SVM). The function maps the values in the 'Label' column as follows:

    - 0: Benign traffic ('BENIGN')
    - 1: Malicious traffic (any value other than 'BENIGN')

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame must contain a 'Label' column.

    Returns
    -------
    df : pandas.DataFrame
        The transformed DataFrame, with the 'Label' column converted to a binary classification.

    Notes
    -----
    - Checks for the presence of a 'Label' or ' Label' (with leading space) column before proceeding.
    - Displays a pie chart of the class distribution after the transformation.
    - Prints a warning message if the 'Label' column is not found.

    Example
    -------
    >>> df = pd.DataFrame({
    ...     'Label': ['BENIGN', 'Attack', 'BENIGN', 'Malware', 'DDoS']
    ... })
    >>> df_binary = transform_label(df)
    >>> print(df_binary.head())
    '''
    
    # Identify the correct 'Label' column
    target_col = 'Label' if 'Label' in df.columns else ' Label'
    
    # Check if the 'Label' column is present
    if target_col in df.columns:
        # Apply binary transformation
        df[target_col] = df[target_col].apply(lambda x: 0 if x == 'BENIGN' else 1)

        # Plot the label distribution
        plt.figure(figsize=(6, 6))
        df[target_col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        plt.title('Label Distribution')
        plt.ylabel('')
        plt.tight_layout()
        plt.show()
    else:
        print("'Label' column not found in the DataFrame.")
    
    return df



def reset_index(df):
    '''
    Resets the index of the input DataFrame, removing the existing index and replacing it with a 
    default integer-based index. This can be useful after merging, concatenating, or filtering 
    operations that result in non-sequential or non-unique index values.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame whose index should be reset.

    Returns
    -------
    df : pandas.DataFrame
        The DataFrame with a reset index. The old index is dropped, and the default integer index is applied.

    Notes
    -----
    - The original index is permanently removed.
    - The operation is non-destructive to the DataFrame's data, affecting only the index.
    - This function returns a copy of the DataFrame with the updated index.

    Example
    -------
    >>> df_reset = reset_index(df)
    >>> print(df_reset)
    '''
    
    return df.reset_index(drop=True)

