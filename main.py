# -*- coding: utf-8 -*-
"""
Created on Mon May 12 19:56:44 2025

@author: 810624TJ
"""

import preprocessing_function as pre_fx
import feature_selection_function as fs_fx
import qsvm_function as qsvm_fx
import svm_function as svm_fx

PATH_RAWDATA = 'insert_path_csv'

final_df = pre_fx.import_data(PATH_RAWDATA)
final_df = pre_fx.clean_dataset(final_df)
df_final = pre_fx.transform_ip_to_frequencies(final_df)
df_final = pre_fx.drop_flow_id(df_final)
df_final = pre_fx.categorize_ports(df_final)
df_final = pre_fx.encode_protocol(df_final)
df_final = pre_fx.transform_label(df_final)
df_final = pre_fx.reset_index(df_final)

X_train_scaled, X_test_scaled, y_train, y_test = fs_fx.scale_features(df_final)
X_train_selected, y_train_selected, rf_model, selector = fs_fx.feature_selection(X_train_scaled, y_train)
X_train_filtered, X_test_filtered, selected_features = fs_fx.select_important_features(rf_model, X_train_scaled, X_test_scaled)


qsvc, y_pred_q, misclassified = qsvm_fx.train_quantum_svc(X_train_filtered, X_test_filtered, y_train, y_test)

svc, y_pred_svc, misclassified = svm_fx.train_classical_svc(X_train_filtered, X_test_filtered, y_train, y_test)
