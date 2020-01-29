#Implementazione SVM in python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

bankdata = pd.read_csv("/Users/user/Desktop/bill_authentication.csv")
bankdata.head()

#To divide the data into attributes and labels, execute the following code:
#In the first line of the script above, all the columns of the bankdata dataframe are being stored in the X variable except the "Class" column, 
#which is the label column. The drop() method drops this column. In the second line, only the class column is being stored in the y variable. 
#At this point of time X variable contains attributes while y variable contains corresponding labels.
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']

#the model_selection library of the Scikit-Learn library contains the train_test_split method that allows us to seamlessly divide data into training and test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#In the case of a simple SVM we simply set this parameter as "linear" since simple SVMs can only classify linearly separable data.
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear') #Polynomial Kernel (poly); Gaussian K. (rbf); Sigmoid K. (sigmoid)
svclassifier.fit(X_train, y_train)

#To make predictions, the predict method of the SVC class is used. 
y_pred = svclassifier.predict(X_test)

#Evaluating the Algorithm Precision, Recall, F1 e confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))