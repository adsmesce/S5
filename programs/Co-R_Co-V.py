import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Heart_disease_cleveland_new.csv')

features = ["thal","ca","exang","oldpeak","cp"]
new_data = data[features]

corr_matrix = new_data.corr()
print('\n\nCorrelation matrix:')
print(corr_matrix)

cov_matrix = np.cov(new_data.T)
print('\n\nCovariance matrix:')
print(cov_matrix)

top_5_corr = corr_matrix['cp'].sort_values(ascending=False)[1:6]

print('\n\nTop 5 attributes closely related to the predicted attribute:')
print(top_5_corr)

