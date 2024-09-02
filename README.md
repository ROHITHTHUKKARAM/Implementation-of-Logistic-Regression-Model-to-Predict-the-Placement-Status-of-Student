# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the Standard Libraries
2. Set Variables for Assigning Dataset Values
3. Import Logistic Regression from sklearn
4. Assign the Points for Representing in the Graph
5. Predict the Placement Status by Using the Logistic Regression Model
6. Compare the Graphs and Evaluate the Model's Performance

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Rohith T
RegisterNumber: 212223040173

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
data = fetch_california_housing()
x= data.data[:, :3]
y= np.column_stack((data.target, data.data[:, 6]))
x_train,x_test,y_train,y_test= train_test_split(x,y, test_size=0.2,random_state=42)
scaler_x= StandardScaler()
scaler_y= StandardScaler()
x_train= scaler_x.fit_transform(x_train)
x_test= scaler_x.transform(x_test)
y_train= scaler_y.fit_transform(y_train)
y_test= scaler_y.transform(y_test)
sgd= SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd= MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)
y_pred= multi_output_sgd.predict(x_test)
y_pred= scaler_y.inverse_transform(y_pred)
y_test= scaler_y.inverse_transform(y_test)
mse= mean_squared_error(y_test,y_pred)
print("Mean squared error:",mse)
print("\npredicition\n",y_pred[:5])

```

## Output:
![Screenshot 2024-09-02 111855](https://github.com/user-attachments/assets/876f574b-7d16-49ba-b46a-d37539b73dae)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
