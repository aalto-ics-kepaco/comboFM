# comboFM


## Overview

comboFM is a machine learning framework for predicting drug combination dose-response matrices, implemented in Python. Given the predicted dose-response matrices, one can subsequently quantify the drug combination synergy scores. 

The computations can be parallelized using array jobs.  comboFM_nestedCV.py runs each outer cross validation (CV) loop as a separate array job. The script takes a number identifying the outer CV loop as an input. In the comboFM publication, we used 10x5 nested CV (10 outer folds, 5 inner folds). Hence, the corresponding bash script should contain #SBATCH --array=0-9. One can also pass the name of the prediction scenario as an input argument, which has to be one of the following options: 1) new_dose-response_entries, 2) new_dose-response_matrices or 3) new_drug_combinations. We recommend using GPUs for more efficient computations.

## Dependencies

- numpy
- scikit-learn
- scipy
- tqdm
- tensorflow 1.0+

comboFM also requires installation of TensorFlow-based factorization machine [1], which can be installed e.g. by pip install tffm. 


## Citing comboFM

comboFM is described in the following article:
…

## Licence

## References 

[1] Mikhail Trofimov and Alexander Novikov. TFFM: TensorFlow implementation of an arbitrary order Factorization Machine, 2016. https://github.com/geffy/tffm.
