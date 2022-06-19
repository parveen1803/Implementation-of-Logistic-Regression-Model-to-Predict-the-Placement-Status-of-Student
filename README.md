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
![original data](https://user-images.githubusercontent.com/87666371/174467914-8638c3d3-7d75-49c6-b150-d2fec4c7d9ab.png)

## Data after dropping unwanted columns(first five):
![dataafterdropping](https://user-images.githubusercontent.com/87666371/174467921-26b10910-a52b-4073-89eb-1494ac2fa052.png)

## Checking the presence of null values:
![presencevalue](https://user-images.githubusercontent.com/87666371/174467928-eb9e212e-051b-4e05-9bc2-d1debf1bf6e2.png)

## Checking the presence of duplicated values:
![duplicaredvalue](https://user-images.githubusercontent.com/87666371/174467938-4b0c1a6d-c1ee-4f2a-a461-d922524c7eec.png)

## Data after Encoding:
![dataaftercoding](https://user-images.githubusercontent.com/87666371/174467947-af4afd87-789a-4d3c-99ac-912412d7b742.png)

## X Data:
![xdata](https://user-images.githubusercontent.com/87666371/174467960-34fb2f1b-adc9-43da-894f-6ebcb52182a5.png)

## Y Data:
![ydata](https://user-images.githubusercontent.com/87666371/174467981-e31fb1ab-c655-47ef-bc51-d5e57a5de06d.png)

## Predicted Values:
![predicted](https://user-images.githubusercontent.com/87666371/174467803-4104cda4-4b4f-4ba8-b7c2-cef18e734bd0.png)

## Accuracy Score:
![accuracyscore](https://user-images.githubusercontent.com/87666371/174467989-89175b84-8049-4717-aac9-7a4849002895.png)

## Confusion Matrix:
 ![confusionmatrix](https://user-images.githubusercontent.com/87666371/174468000-3d7c273c-8cfe-4f99-96e0-73334aeb8fe9.png)

## Classification Report:
![classificationmatrix](https://user-images.githubusercontent.com/87666371/174468013-6cdff41d-4622-4a19-80f9-b9f83378ca1a.png)
 
## Predicting output from Regression Model:
![outputRM](https://user-images.githubusercontent.com/87666371/174468020-2e428160-d8b7-4e78-b2f7-7ddc4a789ed8.png) 

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
