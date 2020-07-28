import sys
sys.path.append("..")
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
from tffm import TFFMRegressor
from utils import concatenate_features, standardize


def main():
    
    seed = 123 # Random seed
    data_dir = "../validation_data_train/"
    
    n_epochs = 200 # Number of epochs
    learning_rate=0.001 # Learning rate of the optimizer
    batch_size = 1024 # Batch size
    init_std = 0.01 # Initial standard deviation
    input_type = 'sparse' # Input type: 'sparse' or 'dense'
    reg = 10**4 # Regularization parameter
    rank = 100 # Rank of the factorization
    order = 5 # comboFM order
      
    print('GPU available:')
    print(tf.test.is_gpu_available())
    
    ### Training data forr validation experiment
    
     # Features in position 1: Drug A - Drug B
    features_tensor_1 = ("drug1_concentration__one-hot_encoding.csv", "drug2_concentration__one-hot_encoding.csv", "drug1__one-hot_encoding.csv", "drug2__one-hot_encoding.csv", "cell_lines__one-hot_encoding.csv")
    features_auxiliary_1 = ("drug1_drug2_concentration__values.csv", "drug1__estate_fingerprints.csv", "drug2__estate_fingerprints.csv", "cell_lines__gene_expression.csv")
    X_tensor_1 = concatenate_features(data_dir, features_tensor_1)
    X_auxiliary_1 = concatenate_features(data_dir, features_auxiliary_1)
    X_1 = np.concatenate((X_tensor_1, X_auxiliary_1), axis = 1)
    
    # Features in position 2: Drug B - Drug A
    features_tensor_2 = ("drug2_concentration__one-hot_encoding.csv", "drug1_concentration__one-hot_encoding.csv", "drug2__one-hot_encoding.csv", "drug1__one-hot_encoding.csv", "cell_lines__one-hot_encoding.csv")
    features_auxiliary_2 =("drug2_drug1_concentration__values.csv", "drug2__estate_fingerprints.csv", "drug1__estate_fingerprints.csv", "cell_lines__gene_expression.csv")
    X_tensor_2 = concatenate_features(data_dir, features_tensor_2)
    X_auxiliary_2 = concatenate_features(data_dir, features_auxiliary_2)
    X_2 = np.concatenate((X_tensor_2, X_auxiliary_2), axis = 1)
    
    # Concatenate the features from both positions vertically
    X_tr = np.concatenate((X_1, X_2), axis=0)
    print('Dataset shape: {}'.format(X_tr.shape))
    print('Non-zeros rate: {:.05f}'.format(np.mean(X_tr != 0)))
    print('Number of one-hot encoding features: {}'.format(X_tensor_1.shape[1]))
    print('Number of auxiliary features: {}'.format(X_auxiliary_1.shape[1]))
    i_aux = X_tensor_1.shape[1]
    del X_tensor_1, X_auxiliary_1, X_tensor_2, X_auxiliary_2, X_1, X_2
    
    # Read responses
    y_tr  = np.loadtxt("../validation_data_train/responses.csv", delimiter = ",", skiprows = 1)
    y_tr = np.concatenate((y_tr, y_tr), axis=0)
    
    
    ### Validation data
    
    # Validation set features
    data_dir = "../validation_data/"
    X_tensor_1 = concatenate_features(data_dir, features_tensor_1)
    X_auxiliary_1 = concatenate_features(data_dir, features_auxiliary_1)
    X_val = np.concatenate((X_tensor_1, X_auxiliary_1), axis = 1)
    
    print('Validation dataset shape: {}'.format(X_val.shape))
    print('Non-zeros rate: {:.05f}'.format(np.mean(X_val != 0)))
    print('Number of one-hot encoding features: {}'.format(X_tensor_1.shape[1]))
    print('Number of auxiliary features: {}'.format(X_auxiliary_1.shape[1]))
    
    i_aux = X_tensor_1.shape[1]
    del X_tensor_1, X_auxiliary_1

    X_tr, X_val = standardize(X_tr, X_val, i_aux)
    
    if input_type == 'sparse':
        X_tr = sp.csr_matrix(X_tr)
        X_val = sp.csr_matrix(X_val)
    
    model = TFFMRegressor(
        order=order,
        rank=rank,
        n_epochs = n_epochs,
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        batch_size = batch_size,
        init_std=init_std,
        reg=reg,
        input_type=input_type,
        seed=seed
    )

    # Train the model
    model.fit(X_tr, y_tr, show_progress=True)
    
    # Predict
    y_pred_val = model.predict(X_val)
    

    np.savetxt("results/validation_set_predictions.txt", y_pred_val)    
    
main()

