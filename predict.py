import glob
import h5py
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import sys
from scipy.stats import pearsonr

"""
hi. this is going to run ridge regression, you basically just need:

working_dir: where you want your results saved, does not have to exist, we will make it
feature_path: path to the actual features, should be a subject x features npy array
phenotype_path: path to a csv with the to-be-predicted values
phenotype_name: str of name of column in phenotype_path you want
control_path: path to csv, regress these features from the features within each fold
"""
folds = 10
working_dir = sys.argv[1]
os.makedirs(working_dir,exist_ok=True)
feature_path = sys.argv[2] 
phenotype_path = sys.argv[3] 
phenotype_name = sys.argv[4] #'thompson_PC1'
control_path = sys.argv[5]

"""
load the subject measures you want to control for
"""
phenotypes_control = pd.read_csv(control_path)
"""
load in all the feature weights
"""
features = np.load(feature_path).astype(np.float16)
"""
this is adapted from pennlinckit.utils.predict
"""
model_cv = KFold(folds)
targets = pd.read_csv(phenotype_path)[phenotype_name].values.astype(np.float16)
np.save('/{0}/{1}_targets.npy'.format(working_dir,phenotype_name),targets)
assert targets.shape[0] == features.shape[0]
prediction = np.zeros((targets.shape))
coefs = np.zeros((folds,features.shape[1]))
alphas = np.zeros((folds))
fold = 0
for train, test in model_cv.split(np.arange(targets.shape[0])):
    x_train,y_train,x_test,y_test = features[train].copy(),targets[train].copy(),features[test].copy(),targets[test].copy() #split up the data by train/test
    nuisance_model = LinearRegression() #make the nuisance model object
    nuisance_model.fit(phenotypes_control.values[train],x_train) #fit the nuisance_model to training data
    x_train = x_train - nuisance_model.predict(phenotypes_control.values[train]) #remove nuisance from training data
    x_test = x_test - nuisance_model.predict(phenotypes_control.values[test]) #remove nuisance from test data
    m = RidgeCV(cv=10,alphas=(1,10,100,500,1000,5000,10000,15000)) #make the actual ridge model object, adding some super high reg strengths because we have so many features
    m.fit(x_train,y_train) # fit the ridge model
    coefs[fold] = m.coef_.astype(np.float16)
    alphas[fold] = m.alpha_
    prediction[test] = m.predict(x_test) #apply the trained model to the test data, save the result
    np.save('/{0}/{1}_prediction.npy'.format(working_dir,phenotype_name),prediction)
    np.save('/{0}/{1}_coefs.npy'.format(working_dir,phenotype_name),coefs)
    np.save('/{0}/{1}_alphas.npy'.format(working_dir,phenotype_name),alphas)
    fold = fold + 1 
