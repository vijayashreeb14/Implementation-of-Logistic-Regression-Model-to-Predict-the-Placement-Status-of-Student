# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1..Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results.

## Program:


Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIJAYASHREE B
RegisterNumber: 212223040238  
~~~
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~

## Output:

![image](https://github.com/user-attachments/assets/ec7ae59c-cc55-4678-abd5-8e43b6692e6f)

![image](https://github.com/user-attachments/assets/b23977fd-8135-447e-8cbb-2e5fa15c0ddc)

Y-PREDICTION ARRAY:

![image](https://github.com/user-attachments/assets/8e884bea-17d8-4d41-9f0c-8b883eea73a3)

ACCURACY LEVEL:

![image](https://github.com/user-attachments/assets/649ad18b-ed8b-4c2b-be17-ec4628e85e07)

CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/4ed7303a-1746-436e-93b0-3355810c7466)

PREDICTION OF LR:
![image](https://github.com/user-attachments/assets/69a042cb-4f54-4e6e-8822-83fe7187be0c)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
