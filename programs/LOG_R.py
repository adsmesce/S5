import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("Heart_disease_cleveland_new.csv")
print(df)

x = df.drop("target", axis = 1)
y = df["target"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0,stratify=y)

model =LogisticRegression()
model.fit(x_train,y_train)
training_data_predictions = model.predict(x_train)
test_data_predictions = model.predict(x_test)
print("the accuracy score on training data is:", accuracy_score(training_data_predictions,y_train))
print("the accuracy score on test data is:",accuracy_score(test_data_predictions,y_test))

importance=model.coef_[0]
x_train = np.arange(0,len(x_train),1)
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train, training_data_predictions,color="red" ,linewidth=1)
plt.title('logistic regression(test set)')
plt.xlabel('data')
plt.ylabel('condition')
plt.show()
