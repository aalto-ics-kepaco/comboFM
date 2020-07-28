# comboFM: leveraging multi-way interactions for systematic prediction of drug combination effects

## Overview

comboFM is a machine learning framework for predicting the responses of drug combinations in pre-clinical studies, such as those based on cell lines or patient-derived cells, implemented in Python. 

The data used in the experiments is available on: https://zenodo.org/record/3782333#.XyA-xx1RXyw.

## Instructions

*comboFM__nested_CV.py* contains the script for running the full nested cross-validation (CV). The script takes a number identifying the outer CV loop as an input argument, which allows to parallelize the computations using array jobs.  One should also pass the name of the prediction scenario as an input argument, which has to be one of the following options: 
1. new_dose-response_matrix_entries: imputing missing entries in otherwise known dose-response matrices.
2. new_dose-response_matrices: making predictions for completely held out dose-response matrices, such that the drug pair has still been observed in other cell lines.
3. new_drug_combinations: making predictions for completely held out drug combinations, such that the individual drugs are still be observed individually in other combinations.
In all prediction settings, it is assumed that the monotherapy responses of single drugs in a combination are known (i.e. these are included the training set). We recommend using GPUs for more efficient computations.

*comboFM__example.py* contains an example of running comboFM for one outer cross-validation fold with fixed parameters to enable testing without having to run the full nested cross-validation procedure. 

*comboFM_validation.py* contains the script for running an experiment on the validation dataset consisting of the remaining NCI-ALMANAC data not used in the method development. 

*RF__nested_CV.py* contains the script for running the full nested cross-validation (CV) procedure with rarndom forerst.

*cross-validation_folds* stores the cross-validation folds for 10x5 nested CV used in the experiments (10 outer folds, 5 inner folds) for the three prediction settings described above. The folds are designed for duplicated training data in which both of the drugs in a combination are included in both positions such that the symmetry of the drug combinations is taken into account (Drug A - Drug B, Drug B - Drug A), i.e. this informs the algorithm that the combination of drug A with drug B should be considered the same as the combination of drug B with drug A. 

*preprocessing* contains R scripts that were used in preprocessing and extracting the features.

*experimental_validation_data* contains the results from experimental validation described in the article (also available on Zenodo). 

## Dependencies

- numpy
- scikit-learn
- scipy
- tqdm
- tensorflow 1.0+

comboFM also requires installation of TensorFlow-based factorization machine (Mikhail Trofimov and Alexander Novikov. TFFM: TensorFlow implementation of an arbitrary order Factorization Machine, 2016. https://github.com/geffy/tffm). TFFM can be installed e.g. by pip install tffm. 

## Citing comboFM

comboFM is described in the following article:

Julkunen, H.; Cichonska, A.; Gautam, P.; Szedmak, S.; Douat, J.; Pahikkala, T.; Aittokallio, T. and Rousu, J. comboFM: leveraging multi-way interactions for systematic prediction of drug combination effects.

