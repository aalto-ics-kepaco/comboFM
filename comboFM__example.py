import sys, os
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from tffm.models import TFFMRegressor
from utils import concatenate_features, standardize


def main():
    
    # Experiment: 1) new_dose-response_matrix_entries, 2) new_dose-response_matrices, 3) new_drug_combinations
    experiment = "new_dose-response_matrix_entries"
    # Outer test fold to run
    outer_fold = 1
    
    seed = 123 # Random seed
    n_epochs = 50 # Number of epochs 
    learning_rate=0.001 # Learning rate of the optimizer
    batch_size = 1024 # Batch size
    init_std=0.01 # Initial standard deviation
    input_type='sparse' # Input type: 'sparse' or 'dense'
    order = 5 # Order of the factorization machine (comboFM)
    reg = 10**4 # Regularization parameter
    rank = 50 # Rank of the factorization

    print('GPU available:')
    print(tf.test.is_gpu_available())
    
    # Features in position 1: Drug A - Drug B
    features_tensor_1 = ("drug1_concentration__one-hot_encoding.csv", 
                         "drug2_concentration__one-hot_encoding.csv", 
                         "drug1__one-hot_encoding.csv", 
                         "drug2__one-hot_encoding.csv", 
                         "cell_lines__one-hot_encoding.csv")
    features_auxiliary_1 = ("drug1_drug2_concentration__values.csv", 
                            "drug1__estate_fingerprints.csv", 
                            "drug2__estate_fingerprints.csv", 
                            "cell_lines__gene_expression.csv")
    X_tensor_1 = concatenate_features(features_tensor_1)
    X_auxiliary_1 = concatenate_features(features_auxiliary_1)
    X_1 = np.concatenate((X_tensor_1, X_auxiliary_1), axis = 1)
    
    # Features in position 2: Drug B - Drug A
    features_tensor_2 = ("drug2_concentration__one-hot_encoding.csv", 
                         "drug1_concentration__one-hot_encoding.csv", 
                         "drug2__one-hot_encoding.csv", 
                         "drug1__one-hot_encoding.csv", 
                         "cell_lines__one-hot_encoding.csv")
    features_auxiliary_2 =("drug2_drug1_concentration__values.csv", 
                           "drug2__estate_fingerprints.csv", 
                           "drug1__estate_fingerprints.csv", 
                           "cell_lines__gene_expression.csv")
    X_tensor_2 = concatenate_features(features_tensor_2)
    X_auxiliary_2 = concatenate_features(features_auxiliary_2)
    X_2 = np.concatenate((X_tensor_2, X_auxiliary_2), axis = 1)
    
    # Concatenate the features from both positions vertically
    X = np.concatenate((X_1, X_2), axis=0)
    print('Dataset shape: {}'.format(X.shape))
    print('Non-zeros rate: {:.05f}'.format(np.mean(X != 0)))
    print('Number of one-hot encoding features: {}'.format(X_tensor_1.shape[1]))
    print('Number of auxiliary features: {}'.format(X_auxiliary_1.shape[1]))
    i_aux = X_tensor_1.shape[1]
    del X_tensor_1, X_auxiliary_1, X_tensor_2, X_auxiliary_2, X_1, X_2
    
    # Read responses
    y  = np.loadtxt("data/responses.csv", delimiter = ",", skiprows = 1)
    y = np.concatenate((y, y), axis=0)
    
    # Read cross-validation folds and divide the data
    te_idx = np.loadtxt('cross-validation_folds/%s/test_idx_outer_fold-%d.txt'%(experiment, outer_fold)).astype(int)
    tr_idx = np.loadtxt('cross-validation_folds/%s/train_idx_outer_fold-%d.txt'%(experiment, outer_fold)).astype(int)
    
    X_tr, X_te, y_tr, y_te = X[tr_idx,:], X[te_idx,:], y[tr_idx], y[te_idx]

    print('Training set shape: {}'.format(X_tr.shape))
    print('Test set shape: {}'.format(X_te.shape))
    
    # Standardize, i_aux is denotes the index from which the auxiliary descriptors to be standardized start (one-hot encodings should not be standardized)
    X_tr, X_te = standardize(X_tr, X_te, i_aux)
    
    if input_type == 'sparse':
        X_tr = sp.csr_matrix(X_tr)
        X_te = sp.csr_matrix(X_te)
    
    model = TFFMRegressor(
        order = order,
        rank = rank,
        n_epochs = n_epochs,
        optimizer=tf.train.AdamOptimizer(learning_rate = learning_rate),
        batch_size = batch_size,
        init_std=init_std,
        reg = reg,
        input_type = input_type,
        seed = seed
    )
    
    # Train the model
    model.fit(X_tr, y_tr, show_progress=True)
    
    # Predict
    y_pred_te = model.predict(X_te)

    # Evaluate performance
    RMSE = np.sqrt(mean_squared_error(y_te, y_pred_te))
    RPearson = np.corrcoef(y_te, y_pred_te)[0,1]
    RSpearman,_ = spearmanr(y_te, y_pred_te)
    
    print("RMSE: %f\nPearson correlation: %f\nSpearman correlation: %f"%(RMSE, RPearson, RSpearman))
    
main()

