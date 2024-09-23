# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1. Import the required packages and print the present data.

step 2. Find the null and duplicate values. 

step 3. Using logistic regression find the predicted values of accuracy , confusion matrices. 

step 4. Display the results.

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Rohith T
RegisterNumber: 212223040173
import pandas as pd
data = pd.read_csv("C:/Users/admin/Desktop/ML/Placement_Data.csv")
print(data.head())
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
lr =LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
ypred=lr.predict(x_test)
ypred
from sklearn.metrics import accuracy_score, classification_report
accuracy= accuracy_score(y_test, ypred)
accuracy
classification_report1= classification_report(y_test, ypred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-09-16 103700](https://github.com/user-attachments/assets/e38a747e-b7ec-4ed2-b79e-ea3c430ce947)
![Screenshot 2024-09-16 103530](https://github.com/user-attachments/assets/f7695986-f3f4-4f10-819b-e2c4a395b3a4)
![Screenshot 2024-09-16 103520](https://github.com/user-attachments/assets/aca43fa9-7d15-4ce4-9a44-7e6e8865472b)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
