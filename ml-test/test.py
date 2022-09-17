import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 

from sklearn.model_selection import train_test_split  

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler  

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

with open('dataset-new.csv', 'r') as csv_file:
    names = ['Recipe_Name','Review_Count','Author','Prepare_Time','Cook_Time','Total_Time','Ingredients','Directions']
    dataset = pd.read_csv(csv_file , names=names)
   
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])




X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, 4].values  
   
   
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

classifier = KNeighborsClassifier(n_neighbors=5)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))

