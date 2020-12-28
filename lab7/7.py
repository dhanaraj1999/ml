import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
#Read the cleveland heart disease dataset
attributes = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
'ca', 'thal', 'heartdisease']
#Read Cleveland Heart dicease data
heartDisease = pd.read_csv('HeartDisease.csv', names = attributes)
heartDisease = heartDisease.replace('?', np.nan)
# Display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())
print('\nAttributes and datatypes')
print(heartDisease.dtypes)
#Model the Bayesian Network
model = BayesianModel( [ ('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'),
('exang', 'trestbps'), ('trestbps', 'heartdisease'), ('fbs', 'heartdisease'), ('heartdisease', 'restecg'),
('heartdisease', 'thalach'), ('heartdisease', 'chol') ] )
# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning Conditional Probability Distributions using Maximum LikelihoodEstimators...');
model.fit(heartDisease, estimator = MaximumLikelihoodEstimator)
#Inferencing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)
print('\nComputing the probability of Heart disease given age = 28')
q = HeartDisease_infer.query(variables = ['heartdisease'], evidence = {'age': 28})
print(q['heartdisease'])
print('\nComputing the probability of Heart disease given chol = 100')
q = HeartDisease_infer.query(variables = ['heartdisease'], evidence = {'chol': 100})
print(q['heartdisease'])