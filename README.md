# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the standard libraries.
2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3. Import LabelEncoder and encode the dataset.
4. Import LogisticRegression from sklearn and apply the model on the dataset.
5. Predict the values of array.
6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7. Apply new unknown value
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Parveen Fathima M 
RegisterNumber: 212219040103  
*/
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Original data(first five columns):
![Screenshot (13)](https://user-images.githubusercontent.com/87666371/174466991-5f0c9874-9dd3-49c4-99a1-af8dd01297f4.png)

## Data after dropping unwanted columns(first five):
![Screenshot (14)](https://user-images.githubusercontent.com/87666371/174467010-8ba28cda-d4d2-4034-9544-e1c4f55ec8de.png)

## Checking the presence of null values:
![Screenshot (16)](https://user-images.githubusercontent.com/87666371/174467031-37d4b3a5-6925-48d2-9861-815a9946cdca.png)

## Checking the presence of duplicated values:
![Screenshot (38)](https://user-images.githubusercontent.com/87666371/174467628-9ed278e5-8d5e-41a1-b0d1-d18287904034.png)

## Data after Encoding:
![Screenshot (23)](https://user-images.githubusercontent.com/87666371/174467085-722dd946-8929-45fa-b932-a3b6b10acc58.png)

## X Data:
![Screenshot (25)](https://user-images.githubusercontent.com/87666371/174467105-e9ccfe2f-d22f-4875-9b1e-33ed8530cbc4.png)

## Y Data:
![Screenshot (27)](https://user-images.githubusercontent.com/87666371/174467134-263c4c8e-4ae0-48d2-8e9a-37abc254e1fb.png)

## Predicted Values:
![Screenshot (29)](https://user-images.githubusercontent.com/87666371/174467242-11d0755d-3df7-4c91-853b-405fbe6be2e8.png)

## Accuracy Score:
![Screenshot (33)](https://user-images.githubusercontent.com/87666371/174467383-25f203db-f99c-452f-95d9-7856df71d398.png)

## Confusion Matrix:
![Screenshot (34)](https://user-images.githubusercontent.com/87666371/174467412-70563b86-c118-43d0-ad00-e03d7c6f2291.png)
 
## Classification Report:
![Screenshot (31)](https://user-images.githubusercontent.com/87666371/174467314-07e0e6d4-7d99-4750-a8e5-a07bb8b62409.png)
 
## Predicting output from Regression Model:
![Screenshot (36)](https://user-images.githubusercontent.com/87666371/174467446-4c547a6c-4aa1-46ff-a9cb-711ba12d02f5.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
