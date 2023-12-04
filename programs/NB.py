import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv("glass.csv")
print(df.head())

features =['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
x = df[features]
y = df.Type
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = .3,random_state = 100)

clf = GaussianNB()
clf = clf.fit(xtrain,ytrain)
y_predict = clf.predict(xtest)


result_matrix = confusion_matrix(ytest,y_predict)
print("\n confusion matrix : \n",result_matrix)
result = classification_report(ytest,y_predict)
print(result)
accuracy = accuracy_score(ytest,y_predict)
print("\nAccuracy : ",accuracy*100)
