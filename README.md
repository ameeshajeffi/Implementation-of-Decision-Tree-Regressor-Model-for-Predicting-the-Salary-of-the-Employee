# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AMEESHA JEFFI J
RegisterNumber: 212223220007
*/
```

```python
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## Data.Head():

![279486017-17219e1b-9545-45e2-bb45-dd459016cbf9](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/306b5d23-8237-49c3-9969-4ec85ce79c8f)

## Data.info():

![279486035-6c499887-944a-476d-b365-f406cc541e6f](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/3509a2dd-7cd3-4015-a1c8-cf16248bc31a)

## isnull() and sum():

![279486050-e97ab81f-f8b9-4813-83de-327da3214afe](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/f2345bd5-a76a-4755-a767-bda6ac71e4db)

## Data.Head() for salary:

![279486068-ffc344dd-39b6-4370-9282-468f4642736c](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/d16fb1be-4812-44a4-a1ba-378f213f8c55)

## MSE Value:

![279486086-d063c559-f82f-4a52-b1fd-74c153c7d36e](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/19eee10d-0478-4771-890d-0ad61efd8951)

## r2 Value:

![279486100-2956ebf4-c1b2-4a45-9365-21f67717ebc4](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/93f1ad42-3735-413e-9c17-5d9ca2703600)

## Data Prediction:

![279486119-516cbe0b-9937-4dd6-a5a8-1ac01a6673eb](https://github.com/ameeshajeffi/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150773598/3d8ff9d9-a384-4f8a-8de2-64bdfeaa48bb)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
