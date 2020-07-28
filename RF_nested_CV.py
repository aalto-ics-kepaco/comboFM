import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from sys import argv
from utils import concatenate_features, standardize
from sklearn.ensemble import RandomForestRegressor


def main(argv):
    
    seed = 123 # Random seed
    data_dir = "../data/"
    
    nfolds_outer = 10 # Number of folds in the outer loop
    nfolds_inner = 5 # Number of folds in the inner loop
    
    n_trees_params = [32, 64, 128, 512] # Number of trees: to be optimized
    max_features_params = [0.25, 0.5, 0.75, 1.0] # Max features (fraction of n): to be optimized
    
    # Experiment: 1) new_dose-response_matrix_entries, 2) new_dose-response_matrices, 3) new_drug_combinations"""
    experiment = argv[2]
   
    id_in = int(argv[1]) 
    print("\nJob ID: %d" %id_in)
        
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
    X = np.concatenate((X_1, X_2), axis=0)
    print('Dataset shape: {}'.format(X.shape))
    print('Non-zeros rate: {:.05f}'.format(np.mean(X != 0)))
    print('Number of one-hot encoding features: {}'.format(X_tensor_1.shape[1]))
    print('Number of auxiliary features: {}'.format(X_auxiliary_1.shape[1]))
    i_aux = X_tensor_1.shape[1]
    del X_tensor_1, X_auxiliary_1, X_tensor_2, X_auxiliary_2, X_1, X_2
    
    # Read responses
    y  = np.loadtxt("../data/responses.csv", delimiter = ",", skiprows = 1)
    y = np.concatenate((y, y), axis=0)
    
    inner_folds = list(range(1, nfolds_inner+1))
    outer_folds = list(range(1, nfolds_outer+1))
    
    outer_fold = outer_folds[id_in]
    te_idx = np.loadtxt('../cross-validation_folds/%s/test_idx_outer_fold-%d.txt'%(experiment, outer_fold)).astype(int)
    tr_idx = np.loadtxt('../cross-validation_folds/%s/train_idx_outer_fold-%d.txt'%(experiment, outer_fold)).astype(int)
    
    X_tr, X_te, y_tr, y_te = X[tr_idx,:], X[te_idx,:], y[tr_idx], y[te_idx]

    print('Training set shape: {}'.format(X_tr.shape))
    print('Test set shape: {}'.format(X_te.shape))
    
    
    CV_RMSE_n_trees = np.zeros([len(n_trees_params), nfolds_inner])
    CV_RPearson_n_trees = np.zeros([len(n_trees_params), nfolds_inner])
    CV_RSpearman_n_trees = np.zeros([len(n_trees_params), nfolds_inner])
     
    max_features = 1.0 # Fix max features to 100% while optimizing number of trees
     
    for n_trees_i in range(len(n_trees_params)):
         
        n_trees = n_trees_params[n_trees_i]
         
        for inner_fold in inner_folds:
            print("INNER FOLD: %d" %inner_fold)
            print("n_trees: %d" %n_trees)
            print("max_features: %d" %max_features)
             
            te_idx_CV = np.loadtxt('../cross-validation_folds/%s/test_idx_outer_fold-%d_inner_fold-%d.txt'%(experiment, outer_fold, inner_fold)).astype(int)
            tr_idx_CV = np.loadtxt('../cross-validation_folds/%s/train_idx_outer_fold-%d_inner_fold-%d.txt'%(experiment, outer_fold, inner_fold)).astype(int)
            X_tr_CV, X_te_CV, y_tr_CV, y_te_CV = X[tr_idx_CV,:], X[te_idx_CV,:], y[tr_idx_CV], y[te_idx_CV]
           
    
            X_tr_CV, X_te_CV, y_tr_CV, y_te_CV = X[tr_idx_CV,:], X[te_idx_CV,:], y[tr_idx_CV], y[te_idx_CV]
            X_tr_CV, X_te_CV = standardize(X_tr_CV, X_te_CV, i_aux)
           
            model = RandomForestRegressor(
                n_estimators = n_trees,
                max_features = max_features,
                random_state = seed,
                n_jobs = 12
            )
            
            # Train the model
            model.fit(X_tr_CV, y_tr_CV)
             
            # Predict
            y_pred_te_CV = model.predict(X_te_CV)
            
            # Evaluate performance
            RMSE = np.sqrt(mean_squared_error(y_te_CV, y_pred_te_CV))
            CV_RMSE_n_trees[n_trees_i, inner_fold-1] = RMSE
            RPearson = np.corrcoef(y_te_CV, y_pred_te_CV)[0,1]
            CV_RPearson_n_trees[n_trees_i, inner_fold-1] = RPearson
            RSpearman,_ = spearmanr(y_te_CV, y_pred_te_CV)
            CV_RSpearman_n_trees[n_trees_i, inner_fold-1] = RSpearman
             
            print("RMSE: %f\nR_pearson: %f\nR_spearman: %f"%(RMSE, RPearson, RSpearman))
     
    CV_avg_n_trees = np.mean(CV_RPearson_n_trees, axis=1)
    n_trees_i= np.where(CV_avg_n_trees == np.max(CV_avg_n_trees))[0]
    n_trees = n_trees_params[int(n_trees_i)]
    np.savetxt('results_rf/%s/outer_fold-%d_n_trees_CV_avg_RPearson.txt'%(experiment,outer_fold), CV_avg_n_trees)
     
    CV_RMSE_max_features = np.zeros([len(max_features_params), nfolds_inner])
    CV_RPearson_max_features = np.zeros([len(max_features_params), nfolds_inner])
    CV_RSpearman_max_features = np.zeros([len(max_features_params), nfolds_inner])
    
    for max_features_i in range(len(max_features_params)):
        
        max_features = max_features_params[max_features_i]
        
        for inner_fold in inner_folds:
            
            print("INNER FOLD: %d" %inner_fold)
            print("max_features: %d" %max_features)
            print("n_trees: %d" %n_trees)
            
            te_idx_CV = np.loadtxt('../cross-validation_folds/%s/test_idx_outer_fold-%d_inner_fold-%d.txt'%(experiment, outer_fold, inner_fold)).astype(int)
            tr_idx_CV = np.loadtxt('../cross-validation_folds/%s/train_idx_outer_fold-%d_inner_fold-%d.txt'%(experiment, outer_fold, inner_fold)).astype(int)

            X_tr_CV, X_te_CV, y_tr_CV, y_te_CV = X[tr_idx_CV,:], X[te_idx_CV,:], y[tr_idx_CV], y[te_idx_CV]
            X_tr_CV, X_te_CV = standardize(X_tr_CV, X_te_CV, i_aux)
          
            
            model = RandomForestRegressor(
                n_estimators = n_trees,
                max_features = max_features,
                random_state = seed,
                n_jobs = 12
            )
            
            # Train the model
            model.fit(X_tr_CV, y_tr_CV)
            
            # Predict
            y_pred_te_CV = model.predict(X_te_CV)
            
            #  Evaluate performance
            RMSE = np.sqrt(mean_squared_error(y_te_CV, y_pred_te_CV))
            CV_RMSE_max_features[max_features_i, inner_fold-1] = RMSE
            RPearson = np.corrcoef(y_te_CV, y_pred_te_CV)[0,1]
            CV_RPearson_max_features[max_features_i, inner_fold-1] = RPearson
            RSpearman,_ = spearmanr(y_te_CV, y_pred_te_CV)
            CV_RSpearman_max_features[max_features_i, inner_fold-1] = RSpearman
              
            print("RMSE: %f\nR_pearson: %f\nR_spearman: %f"%(RMSE, RPearson, RSpearman))


    CV_avg_max_features = np.mean(CV_RPearson_max_features, axis=1)
    max_features_i = np.where(CV_avg_max_features == np.max(CV_avg_max_features))[0]
    max_features = max_features_params[int(max_features_i)]
    
    np.savetxt('results_rf/%s/outer_fold-%d_max_features_CV_avg_RPearson.txt'%(experiment,outer_fold), CV_avg_max_features)

    X_tr, X_te = standardize(X_tr, X_te, i_aux)
    
    model = RandomForestRegressor(
                n_estimators = n_trees,
                max_features = max_features,
                random_state = seed,
                n_jobs = 12
            )

    # Train the model
    model.fit(X_tr, y_tr)
    
    # Predict
    y_pred_te = model.predict(X_te)

    np.savetxt("results_rf/%s/outer-fold-%d_y_test_order_max_features-%f_n_trees-%d_%s.txt"%(experiment, outer_fold, max_features, n_trees, experiment), y_te)
    np.savetxt("results_rf/%s/outer-fold-%d_y_pred_order_max_features-%f_n_trees-%d_%s.txt"%(experiment, outer_fold,  max_features, n_trees, experiment), y_pred_te)

main(argv)

