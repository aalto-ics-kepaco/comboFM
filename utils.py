import sys
sys.path.append("..")
import numpy as np
from sklearn.preprocessing import StandardScaler

def concatenate_features(feature_filenames):
    """
        This function concatenates the feature matrices in the files listed
        in feature_filenames.
        
        Input:
        feature_filenames: list of the filenames of the features to be included
        
        Output:
        X: final feature matrix (non-standardized)
        """
    print('Reading file: %s'%feature_filenames[0])
    X = np.loadtxt("data/"  + feature_filenames[0] + ".txt") # read the first feature file
    for i in range(1,len(feature_filenames)):
        filename = feature_filenames[i]
        print('Reading file: %s'%feature_filenames[i])
        X_i = np.loadtxt("data/"  + filename + ".txt")
        X = np.concatenate((X, X_i), axis=1)
    print('... done!')
    return X


def standardize(X_train, X_test, i_aux):
    """
        This function standardizes features (z-score transformation) using
        the mean and standard deviation computed on the train set.
        Standardization is performed only on the auxiliary descriptors,
        i.e. exluding the one-hot encoded features.
        
        Input:
        X_train: Train set features (one-hot encoding of the tensor structure + auxiliary descriptors)
        X_test: Test set features (one-hot encoding of the tensor structure + auxiliary descriptors)
        i_aux: Index from which the auxiliary descriptors start to be standardized
               (to exclude the one-hot encodings from standardization)
               
        Output:
        X_train: Scaled train set features
        X_test: Scaled test set features
    """
    
    scaler = StandardScaler()
    scaler.fit(X_train[:,i_aux:])
    X_train[:,i_aux:]=scaler.transform(X_train[:,i_aux:])
    X_test[:,i_aux:]=scaler.transform(X_test[:,i_aux:])
    
    return X_train, X_test
