import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv(r"lois_continuous.csv")
print(df.head(10))

cond = 'Conductivity 25C continuous'
temp = 'Temperature water continuous'
df = df.dropna(subset=[cond, temp])
x = df[[temp]]  
y = df[cond]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
clf = LinearRegression()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

plt.scatter(x, y, label='Data points',color='black')
plt.plot(x_test, y_pred, color='yellow', label='Linear Regression')
plt.title('Linear Regression')
plt.xlabel(temp)
plt.ylabel(cond)
plt.legend()
plt.show() 

mse = mean_squared_error(y_test, y_pred)
print('RMSE :', np.sqrt(mse))

